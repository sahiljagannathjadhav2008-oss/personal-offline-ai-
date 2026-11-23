# Security and privacy (what is actually implemented)
| Feature | Implementation | Verified |
|---|---|---|
| Offline | Manifest declares **no permissions**; no network code | JUnit manifest test + CI `aapt2 dump permissions` check (CI: pending first run) |
| Encrypted conversations | AES-256-GCM, non-exportable Android Keystore key, one file per conversation (`data/Security.kt`, `Repository.kt`) | Instrumented test written — **NOT RUN** (needs emulator/device) |
| No cloud backup | `allowBackup=false`, data-extraction rules exclude all | JUnit manifest test |
| Deletion | per-conversation delete, delete-all-conversations, delete-all-local-data (also deletes the Keystore key) | Instrumented test — **NOT RUN** |
| Model integrity | SHA-256 over weights checked on every load; corrupt/bad-magic files rejected | Kotlin checks PASS (sandbox JVM) |
| Checkpoint integrity | SHA-256 sidecar checked on load | pytest PASS |
| Input validation | control characters stripped, 1000-char cap, prompt truncated to context | Kotlin checks PASS |
| Prompt injection / tools | No tool or command execution exists; output is displayed as plain text | by design |
| Training use of chats | Conversations are never used for training; no telemetry | by design |
**Not implemented:** accounts/passwords (so no Argon2id), secure backups/export. Release APKs built in CI are signed with a throwaway CI key; create your own keystore for distribution.
