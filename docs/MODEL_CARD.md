# Model card — SJ Foundation AI base LM v0.1
- **Developer:** Sahil Jadhav. **Weights:** random init, trained by this project; no third-party pretrained weights.
- **Architecture:** 7 layers, 5 heads, d_model 320, d_ff 864, context 128, RoPE, RMSNorm, SwiGLU, tied embeddings. **9,791,680 parameters** (counted in code).
- **Tokenizer:** SentencePiece BPE with byte fallback, **3,480 tokens** (12,000 was requested, but the corpus is too small to support more).
- **Training:** next-token prediction, CPU only (1 core), AdamW lr 1e-3, batch 16×128 tokens. Best checkpoint = step 75 (val loss 1.1357, perplexity 3.11); training stopped early at step 175 because validation loss rose (overfitting).
- **Data:** ≈500 unique original texts (~5k tokens), hand-written paragraphs + grammar templates, CC0. See `DATASET.md`.
- **Deployed:** INT8 row-wise weights (~9.9 MB); logits differ from FP32 by ≤0.03 on the sample checked.
- **Capabilities that exist:** continues short simple sentences in Marathi, Hindi, English; developer identity is answered by app logic, not the model.
- **Limitations:** no general knowledge, not instruction-tuned, repetitive/incorrect output, tiny context, validation set is small and in-domain (perplexity is optimistic). The prompt is treated as text to continue, not as a chat instruction.
- **Not implemented:** vision, audio, tools, agents, GPU/NPU acceleration (modular interfaces only).
