# Build
1. Create a GitHub repository and push this folder. 2. Open **Actions → Build SJ Foundation AI**. 3. Download the `SJ-Foundation-AI-release-apk` artifact.
The workflow installs JDK 17 + Android SDK 35, runs Python tests and Android unit tests, builds the release APK (signed with a CI-generated key), checks the APK exists, contains the model and requests no permissions, and uploads it. Run it manually with "run_instrumented" to execute emulator tests.
Gradle 8.10.2, AGP 8.7.2, Kotlin 2.0.21, Compose BOM 2024.10.01, minSdk 26.
