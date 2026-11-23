# SJ Foundation AI

An independent small language model (≈9.8 M parameters) trained **from random initialization** by this project,
plus an offline Android app that runs it. Developer: **Sahil Jadhav**.

> **Honest status.** This is an experimental base model trained on a very small original corpus (~5,000 tokens).
> It can continue short Marathi/Hindi/English sentences and is **not** a capable assistant. See `docs/MODEL_CARD.md`.
> The Android app has **not been compiled or run yet** — the first GitHub Actions run is the first real build.
> See `docs/TEST_REPORT.md` for exactly what was and was not verified.

## Layout
| Path | What |
|---|---|
| `ml/sjai/model.py` | Decoder-only Transformer (RoPE, RMSNorm, SwiGLU, tied embeddings) |
| `ml/sjai/corpus_gen.py`, `pipeline.py` | Original CC0 corpus + dataset pipeline (licence, lang-id, clean, PII filter, dedupe, split, manifest) |
| `ml/sjai/train.py` | AdamW, warmup+cosine, grad clipping/accumulation, checksummed checkpoints, resume, early stopping |
| `ml/sjai/export.py` | INT8 SJW1 export, reference engine, SentencePiece-compatible tokenizer |
| `ml/tests/` | 25 pytest tests |
| `android/` | Kotlin/Compose app, pure-Kotlin INT8 inference engine, Gradle wrapper |
| `android/tools/` | Plain-JVM verification harness for the Kotlin engine (sandbox use) |
| `artifacts/` | Tokenizer, best checkpoint, release model/tokenizer/vectors, training log |
| `.github/workflows/build.yml` | Python tests, Android unit tests, signed release APK, APK checks, artifact upload |

## Build the APK
Push to GitHub. The workflow builds `SJ-Foundation-AI-release-apk` (no Android Studio needed). See `docs/BUILD.md`.

## Retrain
`cd ml && python -m sjai.corpus_gen ../data/raw && python -m sjai.pipeline ../data/raw ../data/processed && python -m sjai.train`
See `docs/TRAINING.md`. Scaling to a real model requires a much larger, properly licensed corpus.
