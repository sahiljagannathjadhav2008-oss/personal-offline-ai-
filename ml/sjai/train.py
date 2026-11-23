"""Training engine: next-token prediction, AdamW, warmup+cosine LR, grad clipping,
gradient accumulation, atomic checkpoints with SHA-256 sidecars, resume, validation."""
import argparse, dataclasses, hashlib, json, math, os, pathlib, time
import torch
import sentencepiece as spm
from .model import Config, SJModel


def sha256(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def encode_corpus(sp, path):
    ids = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            ids += sp.encode(line) + [sp.eos_id()]
    return torch.tensor(ids, dtype=torch.long)


def get_batch(data, bs, T, g):
    ix = torch.randint(0, len(data) - T - 1, (bs,), generator=g)
    x = torch.stack([data[i:i + T] for i in ix])
    y = torch.stack([data[i + 1:i + T + 1] for i in ix])
    return x, y


def lr_at(step, total, base, warm, min_ratio=0.1):
    if step < warm:
        return base * (step + 1) / warm
    p = (step - warm) / max(1, total - warm)
    return base * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p)))


@torch.no_grad()
def evaluate(model, data, bs, T, batches=8, seed=0):
    model.eval(); g = torch.Generator().manual_seed(seed); tot = 0.0
    for _ in range(batches):
        x, y = get_batch(data, bs, T, g); tot += model(x, y)[1].item()
    model.train(); l = tot / batches
    return l, math.exp(l)


def save_ckpt(path, model, opt, step, cfg, extra):
    path = pathlib.Path(path); tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                "config": dataclasses.asdict(cfg), "extra": extra}, tmp)
    os.replace(tmp, path)
    path.with_suffix(".sha256").write_text(sha256(path))


def load_ckpt(path):
    path = pathlib.Path(path)
    side = path.with_suffix(".sha256")
    if side.exists() and side.read_text().strip() != sha256(path):
        raise RuntimeError(f"checkpoint integrity check failed: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def main(a=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/processed"); ap.add_argument("--tok", default="../artifacts/sj_tokenizer.model")
    ap.add_argument("--out", default="../artifacts/run"); ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--bs", type=int, default=8); ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--seq", type=int, default=128); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=30); ap.add_argument("--eval_every", type=int, default=25)
    ap.add_argument("--ckpt_every", type=int, default=50); ap.add_argument("--resume", action="store_true")
    ap.add_argument("--layers", type=int, default=7); ap.add_argument("--dmodel", type=int, default=320)
    ap.add_argument("--heads", type=int, default=5); ap.add_argument("--dff", type=int, default=864)
    ap.add_argument("--seed", type=int, default=1337); ap.add_argument("--patience", type=int, default=4)
    args = ap.parse_args(a)
    torch.manual_seed(args.seed); g = torch.Generator().manual_seed(args.seed)
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sp = spm.SentencePieceProcessor(model_file=args.tok)
    tr = encode_corpus(sp, pathlib.Path(args.data) / "train.txt")
    va = encode_corpus(sp, pathlib.Path(args.data) / "val.txt")
    cfg = Config(vocab_size=sp.get_piece_size(), n_layer=args.layers, n_head=args.heads,
                 d_model=args.dmodel, d_ff=args.dff, max_seq=args.seq)
    model = SJModel(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    step = 0; ck = out / "last.pt"
    if args.resume and ck.exists():
        s = load_ckpt(ck); model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); step = s["step"]
        print(f"resumed from step {step}")
    print(f"params={model.num_params():,} train_tokens={len(tr):,} val_tokens={len(va):,} device=cpu")
    log = open(out / "train_log.jsonl", "a"); t0 = time.time(); model.train()
    best, bad = float("inf"), 0
    while step < args.steps:
        for gp in opt.param_groups:
            gp["lr"] = lr_at(step, args.steps, args.lr, args.warmup)
        opt.zero_grad(set_to_none=True); tl = 0.0
        for _ in range(args.accum):
            x, y = get_batch(tr, args.bs, args.seq, g)
            loss = model(x, y)[1] / args.accum; loss.backward(); tl += loss.item()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        opt.step(); step += 1
        rec = {"step": step, "loss": round(tl, 4), "grad_norm": round(gn, 3), "lr": opt.param_groups[0]["lr"],
               "elapsed_s": round(time.time() - t0, 1)}
        if step % args.eval_every == 0 or step == args.steps:
            vl, pp = evaluate(model, va, min(args.bs, 4), args.seq)
            rec.update(val_loss=round(vl, 4), val_ppl=round(pp, 2))
            if vl < best - 1e-4:
                best, bad = vl, 0
                torch.save({"model": model.state_dict(), "config": dataclasses.asdict(cfg), "step": step, "val_loss": vl}, out / "best.pt")
            else:
                bad += 1
        log.write(json.dumps(rec) + "\n"); log.flush()
        if step % 10 == 0 or "val_loss" in rec:
            print(rec, flush=True)
        if step % args.ckpt_every == 0 or step == args.steps:
            save_ckpt(ck, model, opt, step, cfg, {"dataset": str(args.data), "tokenizer_sha256": sha256(args.tok)})
        if bad >= args.patience:
            print(f"early stop at step {step}: best val_loss={best:.4f}", flush=True); break
    torch.save({"model": model.state_dict(), "config": dataclasses.asdict(cfg)}, out / "model_fp32.pt")
    print("done")


if __name__ == "__main__":
    main()
