"""Deployment export + pure-numpy reference inference engine and BPE tokenizer.

SJW1 format:  b"SJW1" | u32 header_len | header JSON (utf-8) | data section.
Header: {"config": {...}, "quant": "int8-rowwise", "tensors": {name: {dtype, shape, offset, nbytes}},
         "data_sha256": ...}.  Linear/embedding weights are int8 with fp32 per-row scales (stored
as a separate "<name>.scale" tensor); norm weights are fp32.  Little-endian throughout.
The Kotlin engine and the numpy reference below implement the same maths."""
import hashlib, json, struct, sys, unicodedata
import numpy as np
import torch

MAGIC = b"SJW1"


def quantize_rows(w):
    w = w.astype(np.float32)
    scale = np.maximum(np.abs(w).max(axis=1), 1e-8) / 127.0
    q = np.clip(np.round(w / scale[:, None]), -127, 127).astype(np.int8)
    return q, scale.astype(np.float32)


def export_weights(pt_path, out_path, quantize=True):
    ck = torch.load(pt_path, map_location="cpu", weights_only=False)
    sd = {k: v.numpy() for k, v in ck["model"].items()}
    tensors, chunks, off = {}, [], 0

    def add(name, arr, dtype):
        nonlocal off
        b = np.ascontiguousarray(arr).tobytes()
        tensors[name] = {"dtype": dtype, "shape": list(arr.shape), "offset": off, "nbytes": len(b)}
        chunks.append(b); off += len(b)

    for k, v in sd.items():
        if k in ("head.weight",):  # tied to emb.weight
            continue
        is_matrix = v.ndim == 2
        if is_matrix and quantize:
            q, s = quantize_rows(v); add(k, q, "int8"); add(k + ".scale", s, "float32")
        else:
            add(k, v.astype(np.float32), "float32")
    data = b"".join(chunks)
    header = {"config": ck["config"], "quant": "int8-rowwise" if quantize else "fp32", "tensors": tensors,
              "data_sha256": hashlib.sha256(data).hexdigest(), "format": "SJW1"}
    hb = json.dumps(header).encode("utf-8")
    with open(out_path, "wb") as f:
        f.write(MAGIC + struct.pack("<I", len(hb)) + hb + data)
    return header


def load_weights(path):
    raw = open(path, "rb").read()
    if raw[:4] != MAGIC:
        raise ValueError("bad magic")
    (hl,) = struct.unpack("<I", raw[4:8]); header = json.loads(raw[8:8 + hl]); data = raw[8 + hl:]
    if hashlib.sha256(data).hexdigest() != header["data_sha256"]:
        raise ValueError("model file integrity check failed")
    W = {}
    for n, t in header["tensors"].items():
        dt = np.int8 if t["dtype"] == "int8" else np.float32
        W[n] = np.frombuffer(data, dt, count=int(np.prod(t["shape"])), offset=t["offset"]).reshape(t["shape"])
    return header["config"], W


def _deq(W, n):
    w = W[n]
    return w.astype(np.float32) * W[n + ".scale"][:, None] if w.dtype == np.int8 else w


class RefEngine:
    """Single-sequence engine with KV cache (mirrors the Kotlin implementation)."""

    def __init__(self, path):
        self.c, W = load_weights(path)
        c = self.c
        self.emb = _deq(W, "emb.weight")
        self.L = []
        for i in range(c["n_layer"]):
            g = lambda s: _deq(W, f"blocks.{i}.{s}")
            self.L.append(dict(n1=g("n1.w"), qkv=g("a.qkv.weight"), o=g("a.o.weight"),
                               n2=g("n2.w"), g=g("m.g.weight"), u=g("m.u.weight"), d=g("m.d.weight")))
        self.nf = _deq(W, "norm.w")
        hd = c["d_model"] // c["n_head"]; self.hd = hd
        inv = 1.0 / (c["rope_base"] ** (np.arange(0, hd, 2, dtype=np.float32) / hd))
        f = np.outer(np.arange(c["max_seq"], dtype=np.float32), inv)
        self.cos, self.sin = np.cos(f), np.sin(f)
        self.reset()

    def reset(self):
        self.k = [[] for _ in self.L]; self.v = [[] for _ in self.L]; self.pos = 0

    @staticmethod
    def _rms(x, w): return x / np.sqrt((x * x).mean() + 1e-6) * w

    def _rope(self, x, pos):  # x: (h, hd)
        x1, x2 = x[:, ::2], x[:, 1::2]; c, s = self.cos[pos], self.sin[pos]
        out = np.empty_like(x); out[:, ::2] = x1 * c - x2 * s; out[:, 1::2] = x1 * s + x2 * c
        return out

    def step(self, tok):
        c = self.c; D, H, hd = c["d_model"], c["n_head"], self.hd
        if self.pos >= c["max_seq"]:
            raise ValueError("context full")
        x = self.emb[tok].copy()
        for i, L in enumerate(self.L):
            qkv = L["qkv"] @ self._rms(x, L["n1"])
            q = self._rope(qkv[:D].reshape(H, hd), self.pos); k = self._rope(qkv[D:2 * D].reshape(H, hd), self.pos)
            self.k[i].append(k); self.v[i].append(qkv[2 * D:].reshape(H, hd))
            K = np.stack(self.k[i]); V = np.stack(self.v[i])  # (t,H,hd)
            sc = np.einsum("hd,thd->ht", q, K) / np.sqrt(hd)
            sc = np.exp(sc - sc.max(1, keepdims=True)); sc /= sc.sum(1, keepdims=True)
            x = x + L["o"] @ np.einsum("ht,thd->hd", sc, V).reshape(D)
            h = self._rms(x, L["n2"]); gt = L["g"] @ h
            x = x + L["d"] @ ((gt / (1 + np.exp(-gt))) * (L["u"] @ h))
        self.pos += 1
        return self.emb @ self._rms(x, self.nf)  # tied output head


# ---------------- tokenizer (SentencePiece-BPE compatible) ----------------
def export_tokenizer(sp, out_path):
    pieces = [{"p": sp.id_to_piece(i), "s": sp.get_score(i), "t": ("c" if sp.is_control(i) else "u" if sp.is_unknown(i)
              else "b" if sp.is_byte(i) else "n")} for i in range(sp.get_piece_size())]
    json.dump({"format": "SJTOK1", "pieces": pieces}, open(out_path, "w", encoding="utf-8"), ensure_ascii=False)


class BPETokenizer:
    def __init__(self, path):
        d = json.load(open(path, encoding="utf-8")); self.pieces = d["pieces"]
        self.id = {p["p"]: i for i, p in enumerate(self.pieces) if p["t"] == "n"}
        self.score = {p["p"]: p["s"] for p in self.pieces if p["t"] == "n"}
        self.byte_id = {int(p["p"][3:5], 16): i for i, p in enumerate(self.pieces) if p["t"] == "b"}
        self.unk = next(i for i, p in enumerate(self.pieces) if p["t"] == "u")
        self.eos = next(i for i, p in enumerate(self.pieces) if p["p"] == "</s>")

    def encode(self, text):
        text = " ".join(unicodedata.normalize("NFKC", text).split())
        if not text:
            return []
        syms = list("\u2581" + text.replace(" ", "\u2581"))
        while True:
            best, bi = None, -1
            for i in range(len(syms) - 1):
                m = syms[i] + syms[i + 1]
                if m in self.score and (best is None or self.score[m] > best):
                    best, bi = self.score[m], i
            if bi < 0:
                break
            syms[bi:bi + 2] = [syms[bi] + syms[bi + 1]]
        out = []
        for s in syms:
            if s in self.id:
                out.append(self.id[s])
            else:
                out += [self.byte_id.get(b, self.unk) for b in s.encode("utf-8")]
        return out

    def decode(self, ids):
        buf = bytearray()
        for i in ids:
            p = self.pieces[i]
            if p["t"] == "b":
                buf.append(int(p["p"][3:5], 16))
            elif p["t"] == "n":
                buf += p["p"].encode("utf-8")
        s = buf.decode("utf-8", errors="replace").replace("\u2581", " ")
        return s[1:] if s.startswith(" ") else s


if __name__ == "__main__":
    pt, out = sys.argv[1], sys.argv[2]
    h = export_weights(pt, out); print("exported", out, h["quant"])
