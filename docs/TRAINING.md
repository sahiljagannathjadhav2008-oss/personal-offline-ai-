# Training
```
cd ml
pip install torch sentencepiece numpy pytest
python -m sjai.corpus_gen ../data/raw
python -m sjai.pipeline ../data/raw ../data/processed
python -m sjai.train --steps 400 --out ../artifacts/run          # add --resume to continue
python -c "from sjai.export import *; export_weights('../artifacts/run/best.pt','../artifacts/release/sj_model_int8.sjw')"
```
Tokenizer: `sentencepiece` BPE, `byte_fallback=True`, ids pad=0 unk=1 bos=2 eos=3, then `export_tokenizer`.
To scale up: supply a larger licensed corpus in `data/raw/raw_corpus.jsonl` + `SOURCES.json` (licence must be on the allow-list in `pipeline.py`), raise `--layers/--dmodel`, and train on a machine with more compute. Re-export, regenerate `test_vectors.json`, and update the model card.
