import json, pathlib, torch, pytest
import sentencepiece as spm
from sjai.model import Config, SJModel
from sjai import train as T, pipeline as P

ROOT = pathlib.Path(__file__).resolve().parents[2]
TOK = ROOT / "artifacts" / "sj_tokenizer.model"
tiny = Config(vocab_size=200, n_layer=2, n_head=2, d_model=32, d_ff=64, max_seq=32)

def test_param_count_matches_formula():
    m = SJModel(tiny)
    c = tiny
    expect = c.vocab_size*c.d_model + c.n_layer*(4*c.d_model**2 + 3*c.d_model*c.d_ff + 2*c.d_model) + c.d_model
    assert m.num_params() == expect

def test_default_config_is_about_10M():
    n = SJModel(Config(vocab_size=3480, n_layer=7, n_head=5, d_model=320, d_ff=864, max_seq=128)).num_params()
    assert 9_000_000 < n < 11_000_000

def test_forward_shape_and_causality():
    torch.manual_seed(0); m = SJModel(tiny).eval()
    x = torch.randint(0, 200, (2, 16)); a, _ = m(x)
    assert a.shape == (2, 16, 200)
    y = x.clone(); y[:, -1] = (y[:, -1] + 1) % 200
    b, _ = m(y)
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-5)  # future token must not leak

def test_gradients_valid_and_weights_update():
    torch.manual_seed(0); m = SJModel(tiny); before = {k: v.clone() for k, v in m.state_dict().items()}
    x = torch.randint(0, 200, (4, 16)); _, loss = m(x[:, :-1], x[:, 1:]); loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    torch.optim.AdamW(m.parameters(), lr=1e-2).step()
    assert any(not torch.equal(before[k], v) for k, v in m.state_dict().items())

def test_loss_decreases_on_small_sample():
    torch.manual_seed(0); m = SJModel(tiny); opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    x = torch.randint(0, 200, (8, 17)); first = None
    for _ in range(60):
        _, l = m(x[:, :-1], x[:, 1:]); opt.zero_grad(); l.backward(); opt.step(); first = first or l.item()
    assert l.item() < first * 0.7

def test_generation_runs():
    m = SJModel(tiny).eval(); out = m.generate(torch.tensor([[1, 2, 3]]), max_new=10)
    assert out.shape == (1, 13)

def test_checkpoint_save_load_resume_and_integrity(tmp_path):
    m = SJModel(tiny); opt = torch.optim.AdamW(m.parameters())
    p = tmp_path / "c.pt"; T.save_ckpt(p, m, opt, 7, tiny, {})
    s = T.load_ckpt(p); assert s["step"] == 7
    m2 = SJModel(tiny); m2.load_state_dict(s["model"])
    assert all(torch.equal(a, b) for a, b in zip(m.state_dict().values(), m2.state_dict().values()))
    p.write_bytes(p.read_bytes()[:-5] + b"xxxxx")
    with pytest.raises(RuntimeError): T.load_ckpt(p)

def test_lr_schedule():
    assert T.lr_at(0, 100, 1.0, 10) < T.lr_at(9, 100, 1.0, 10) <= 1.0
    assert T.lr_at(100, 100, 1.0, 10) == pytest.approx(0.1, abs=1e-6)

@pytest.fixture(scope="module")
def sp(): return spm.SentencePieceProcessor(model_file=str(TOK))

@pytest.mark.parametrize("s", ["आज हवामान छान आहे.", "मुझे चाय पसंद है.", "I like tea.",
    "मी Pune ला जातो, 25 मिनिटे", "क्ष त्र ज्ञ श्री द्य", "😀 ₹500 \u200d", ""])
def test_tokenizer_roundtrip(sp, s):
    assert sp.decode(sp.encode(s)) == s

def test_tokenizer_special_tokens(sp):
    assert (sp.pad_id(), sp.unk_id(), sp.bos_id(), sp.eos_id()) == (0, 1, 2, 3)

def test_devanagari_not_byte_exploded(sp):
    s = "आज हवामान छान आहे."; assert len(sp.encode(s)) < len(s.encode("utf-8")) / 2

def test_pipeline_filters_pii_and_dups(tmp_path):
    raw = tmp_path / "raw"; raw.mkdir()
    (raw / "SOURCES.json").write_text(json.dumps({"license": "CC0-1.0"}))
    rows = ["Hello world this is fine text.", "Hello world this is fine text.", "mail me at a@b.com please ok",
            "call 9876543210 now please ok", "आज हवामान छान आहे."]
    (raw / "raw_corpus.jsonl").write_text("\n".join(json.dumps({"text": r, "source": "t"}, ensure_ascii=False) for r in rows), encoding="utf-8")
    man = P.run(raw, tmp_path / "o", val_frac=0.3)
    assert man["stats"]["pii"] == 2 and man["stats"]["dup"] == 1 and man["stats"]["kept"] == 2

def test_pipeline_rejects_bad_license(tmp_path):
    raw = tmp_path / "raw"; raw.mkdir()
    (raw / "SOURCES.json").write_text(json.dumps({"license": "proprietary"})); (raw / "raw_corpus.jsonl").write_text("")
    with pytest.raises(SystemExit): P.run(raw, tmp_path / "o")

def test_training_resume_continues_from_checkpoint(tmp_path, capsys):
    base = ["--data", str(ROOT / "data/processed"), "--tok", str(TOK), "--out", str(tmp_path), "--layers", "1",
            "--dmodel", "32", "--heads", "2", "--dff", "48", "--seq", "32", "--bs", "2", "--accum", "1",
            "--eval_every", "2", "--ckpt_every", "2", "--warmup", "1", "--patience", "99"]
    T.main(base + ["--steps", "4"])
    assert T.load_ckpt(tmp_path / "last.pt")["step"] == 4
    T.main(base + ["--steps", "6", "--resume"])
    out = capsys.readouterr().out
    assert "resumed from step 4" in out
    assert T.load_ckpt(tmp_path / "last.pt")["step"] == 6
