"""Dataset pipeline: source/licence check -> language id -> clean -> PII filter ->
quality filter -> dedupe -> train/val split -> versioned manifest."""
import hashlib, json, pathlib, random, re, sys, unicodedata

ALLOWED_LICENSES = {"CC0-1.0", "public-domain", "CC-BY-4.0", "MIT", "Apache-2.0"}
PII = [re.compile(p) for p in (
    r"[\w.+-]+@[\w-]+\.[\w.-]+",            # email
    r"(?<!\d)(?:\+?91[- ]?)?[6-9]\d{9}(?!\d)",  # Indian mobile
    r"(?<!\d)\d{4}\s?\d{4}\s?\d{4}(?!\d)",     # 12-digit ID-like numbers
    r"https?://\S+",                          # URLs
)]


def script_counts(t):
    dev = sum(1 for c in t if "\u0900" <= c <= "\u097F")
    lat = sum(1 for c in t if c.isascii() and c.isalpha())
    return dev, lat


def detect_lang(t):
    dev, lat = script_counts(t)
    if dev + lat == 0:
        return "unk"
    if dev and lat and min(dev, lat) / (dev + lat) > 0.15:
        return "mixed"
    return "dev" if dev > lat else "en"


def clean(t):
    t = unicodedata.normalize("NFC", t)
    t = re.sub(r"[\u200b\u200e\u200f\ufeff]", "", t)  # keep ZWJ/ZWNJ (\u200c\u200d) for conjuncts
    t = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", t)
    return re.sub(r"[ \t]+", " ", t).strip()


def has_pii(t):
    return any(p.search(t) for p in PII)


def quality_ok(t):
    if not (8 <= len(t) <= 4000):
        return False
    if detect_lang(t) == "unk":
        return False
    letters = sum(c.isalpha() or "\u0900" <= c <= "\u097F" for c in t)
    return letters / len(t) > 0.5


def run(raw_dir, out_dir, license_id=None, val_frac=0.1, seed=1):
    raw, out = pathlib.Path(raw_dir), pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((raw / "SOURCES.json").read_text(encoding="utf-8"))
    lic = license_id or meta["license"]
    if lic not in ALLOWED_LICENSES:
        raise SystemExit(f"licence {lic!r} not allowed for training data")
    stats = dict(read=0, pii=0, quality=0, dup=0, kept=0)
    seen, kept = set(), []
    for line in open(raw / "raw_corpus.jsonl", encoding="utf-8"):
        stats["read"] += 1
        r = json.loads(line)
        t = clean(r["text"])
        if has_pii(t):
            stats["pii"] += 1; continue
        if not quality_ok(t):
            stats["quality"] += 1; continue
        h = hashlib.sha1(re.sub(r"\W+", "", t.lower()).encode()).hexdigest()
        if h in seen:
            stats["dup"] += 1; continue
        seen.add(h); kept.append({"text": t, "lang": detect_lang(t), "source": r["source"]})
    random.Random(seed).shuffle(kept)
    nv = max(1, int(len(kept) * val_frac))
    splits = {"val": kept[:nv], "train": kept[nv:]}
    digests = {}
    for name, rows in splits.items():
        p = out / f"{name}.txt"
        p.write_text("\n".join(r["text"] for r in rows) + "\n", encoding="utf-8")
        digests[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    stats["kept"] = len(kept)
    langs = {}
    for r in kept:
        langs[r["lang"]] = langs.get(r["lang"], 0) + 1
    ver = hashlib.sha256(json.dumps(digests, sort_keys=True).encode()).hexdigest()[:12]
    manifest = {"dataset_version": ver, "license": lic, "stats": stats, "lang_docs": langs,
                "sha256": digests, "train_docs": len(splits["train"]), "val_docs": len(splits["val"]),
                "source_info": meta}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"version": ver, **stats, "train": len(splits["train"]), "val": len(splits["val"])}))
    return manifest


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
