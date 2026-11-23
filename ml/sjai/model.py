"""SJ Foundation AI - decoder-only Transformer (RoPE, RMSNorm, SwiGLU).
Developer: Sahil Jadhav. Random init; no third-party pretrained weights."""
import math, dataclasses
import torch, torch.nn as nn, torch.nn.functional as F


@dataclasses.dataclass
class Config:
    vocab_size: int = 12000
    n_layer: int = 6
    n_head: int = 6
    d_model: int = 288
    d_ff: int = 768
    max_seq: int = 256
    rope_base: float = 10000.0
    tie_embeddings: bool = True


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


def rope_cache(T, hd, base):
    inv = 1.0 / (base ** (torch.arange(0, hd, 2).float() / hd))
    f = torch.outer(torch.arange(T).float(), inv)
    return f.cos(), f.sin()


def apply_rope(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    T = x.shape[-2]
    c, s = cos[:T], sin[:T]
    return torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), -1).flatten(-2)


class Attn(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.h, self.hd = c.n_head, c.d_model // c.n_head
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model, bias=False)
        self.o = nn.Linear(c.d_model, c.d_model, bias=False)

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(y.transpose(1, 2).reshape(B, T, D))


class SwiGLU(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.g = nn.Linear(c.d_model, c.d_ff, bias=False)
        self.u = nn.Linear(c.d_model, c.d_ff, bias=False)
        self.d = nn.Linear(c.d_ff, c.d_model, bias=False)

    def forward(self, x):
        return self.d(F.silu(self.g(x)) * self.u(x))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1, self.a = RMSNorm(c.d_model), Attn(c)
        self.n2, self.m = RMSNorm(c.d_model), SwiGLU(c)

    def forward(self, x, cos, sin):
        x = x + self.a(self.n1(x), cos, sin)
        return x + self.m(self.n2(x))


class SJModel(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.emb = nn.Embedding(c.vocab_size, c.d_model)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.norm = RMSNorm(c.d_model)
        self.head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        if c.tie_embeddings:
            self.head.weight = self.emb.weight
        cos, sin = rope_cache(c.max_seq, c.d_model // c.n_head, c.rope_base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)
        for n, p in self.named_parameters():  # scaled residual-projection init
            if n.endswith("a.o.weight") or n.endswith("m.d.weight"):
                nn.init.normal_(p, 0, 0.02 / math.sqrt(2 * c.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0, 0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())  # tied weights counted once

    def forward(self, idx, targets=None):
        x = self.emb(idx)
        for b in self.blocks:
            x = b(x, self.cos, self.sin)
        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new=64, temperature=0.8, top_k=40, eos=None):
        for _ in range(max_new):
            lg, _ = self(idx[:, -self.c.max_seq:])
            lg = lg[:, -1] / max(temperature, 1e-5)
            if top_k:
                v, _ = lg.topk(min(top_k, lg.size(-1)))
                lg[lg < v[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(F.softmax(lg, -1), 1)
            idx = torch.cat([idx, nxt], 1)
            if eos is not None and nxt.item() == eos:
                break
        return idx
