# Dataset
- **Source:** `ml/sjai/corpus_gen.py` — 24 hand-written paragraphs (8 per language), identity sentences, and grammar templates with small vocabularies. Written for this project, **CC0-1.0**. No scraped or third-party data.
- **Pipeline (`pipeline.py`):** licence allow-list → NFC clean → PII filter (email, Indian mobile, 12-digit IDs, URLs) → quality filter → dedupe → shuffle/split (10% val) → SHA-256 manifest (`data/processed/MANIFEST.json`, version `e85cd6d170e3`).
- **Result:** 21,660 raw records → 503 unique kept (templates collapse heavily) → 453 train / 50 val.
- **Caveats:** very small and repetitive; validation texts come from the same templates as training; the tokenizer was trained on the pre-split train file before the final split, so validation is slightly leaked. Marathi/Hindi sentences were written by the project author's assistant and have not been reviewed by a native speaker.
- **Language ID** is script-based (Devanagari vs Latin vs mixed); it cannot tell Marathi from Hindi.
