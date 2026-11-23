import pathlib, numpy as np, pytest, torch
import sentencepiece as spm
from sjai.export import *
from sjai.model import Config, SJModel

ROOT = pathlib.Path(__file__).resolve().parents[2]
REL = ROOT / "artifacts" / "release"
SAMPLES = ["आज हवामान छान आहे.", "मुझे चाय पसंद है.", "Who created SJ Foundation AI?", "मी Pune ला जातो, 25 मिनिटे",
           "😀 ₹500", "क्ष त्र ज्ञ श्री", "  extra   spaces  ", "Hello, World!"]

def test_fp32_export_matches_torch(tmp_path):
    torch.manual_seed(0); cfg = Config(vocab_size=64, n_layer=2, n_head=2, d_model=32, d_ff=48, max_seq=16)
    m = SJModel(cfg).eval(); torch.save({"model": m.state_dict(), "config": vars(cfg)}, tmp_path / "m.pt")
    export_weights(tmp_path / "m.pt", tmp_path / "m.sjw", quantize=False)
    ids = list(range(10))
    with torch.no_grad(): ref = m(torch.tensor([ids]))[0][0].numpy()
    e = RefEngine(tmp_path / "m.sjw"); out = np.stack([e.step(t) for t in ids])
    assert np.abs(out - ref).max() < 1e-4

def test_corrupt_model_rejected(tmp_path):
    raw = (REL / "sj_model_int8.sjw").read_bytes(); p = tmp_path / "bad.sjw"
    p.write_bytes(raw[:-10] + b"0" * 10)
    with pytest.raises(ValueError): load_weights(p)

def test_int8_close_to_torch_on_trained_model():
    ck = torch.load(ROOT / "artifacts/run/best.pt", weights_only=False)
    m = SJModel(Config(**ck["config"])); m.load_state_dict(ck["model"]); m.eval()
    tk = BPETokenizer(REL / "sj_tokenizer.json"); ids = tk.encode(SAMPLES[2])
    with torch.no_grad(): ref = m(torch.tensor([ids]))[0][0].numpy()
    e = RefEngine(REL / "sj_model_int8.sjw"); out = np.stack([e.step(t) for t in ids])
    assert np.abs(out - ref).max() < 0.1 and (out.argmax(1) == ref.argmax(1)).mean() >= 0.8

def test_tokenizer_matches_sentencepiece():
    sp = spm.SentencePieceProcessor(model_file=str(ROOT / "artifacts/sj_tokenizer.model")); tk = BPETokenizer(REL / "sj_tokenizer.json")
    for s in SAMPLES + [l.strip() for l in open(ROOT / "data/processed/val.txt", encoding="utf-8") if l.strip()]:
        assert tk.encode(s) == sp.encode(s), s
    for s in SAMPLES: assert tk.decode(tk.encode(s)) == " ".join(s.split())

def test_context_limit_enforced():
    e = RefEngine(REL / "sj_model_int8.sjw")
    for _ in range(e.c["max_seq"]): e.step(5)
    with pytest.raises(ValueError): e.step(5)
