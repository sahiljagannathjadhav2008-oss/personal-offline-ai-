# Test report (generated in the development sandbox: 1 CPU core, 3.9 GB RAM, no GPU, no Android SDK)
| Area | Result | Notes |
|---|---|---|
| Python: model init/param count/forward/causality/grads/weight update/loss decrease/generation | **PASS** (pytest) | 25 tests total |
| Python: tokenizer mr/hi/en/unicode/special/unusual input | **PASS** | |
| Python: checkpoint save/load/corruption/resume | **PASS** | |
| Python: pipeline PII/dedupe/licence | **PASS** | |
| Python: INT8 export vs PyTorch; tokenizer vs SentencePiece | **PASS** | int8 max logit diff 0.026 |
| Training executed | **DONE** | CPU, 400-step run early-stopped at 175; best step 75, val ppl 3.11 |
| Kotlin engine/tokenizer/policy/input-guard (plain JVM, kotlinc 2.0.21, org.json shim) | **PASS** 26/26 | engine logits = Python reference; ~52 tok/s on 1 sandbox core |
| Android compile, APK build | **NOT RUN** | Maven/Google hosts unreachable in sandbox; first CI run will be the first compile |
| JUnit tests in `app/src/test` | **NOT RUN** | written; mirror the plain-JVM checks above |
| Instrumented: Keystore encryption, storage deletion, device snapshot, no-INTERNET | **NOT RUN** | need emulator/device |
| UI, startup, streaming, offline behaviour on device | **NOT RUN** | |
| Tecno Spark 7T / POCO X7 Pro real-device tests | **NOT RUN** | no hardware; only policy logic tested with simulated device values |
| GPU/NNAPI acceleration | **NOT IMPLEMENTED** | CPU only |
| GitHub Actions workflow | **NOT RUN** | written, never executed |
