#!/usr/bin/env python
"""Build the tiny-broca Brazilian Portuguese text corpus.

Assembles ~9k sentences in two batches that mirror the original tiny-voca
dataset (which every sentence is read by both voices):

* batch1: conversational sentences (tiny_broca.converse, a seeded combinatorial
  generator standing in for the LLM-written sentences of the original) plus
  real Common Voice 19 Portuguese sentences filtered to Brazilian Portuguese.
  Split into train/validation/test by sentence.
* batch2 (train only): more Common Voice sentences plus number/date/money
  templates (tiny_broca.templates).

Common Voice is CC0; the generator output is original work (Apache-2.0).

Usage:
    python scripts/build_corpus.py --cv-dir /path/to/common_voice_19_0 \
        --out-dir data/corpus --seed 20261117

The CV dir must contain transcript/pt/{train,dev,test}.tsv (the fsicoli
mirror of Mozilla Common Voice 19). Output: parquet files
data/corpus/batch{1,2}/{split}.parquet with columns id, batch, split, text,
source.
"""

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tiny_broca import converse, templates  # noqa: E402

# Sentences of the original batch1 (4,905) and batch2 (4,079); keep the same
# totals so the model sees the same amount of material.
BATCH1_CONVERSE = 3500
BATCH1_CV = 4905 - BATCH1_CONVERSE
BATCH2_CV = 3000
BATCH2_TEMPLATES = 4079 - BATCH2_CV


def load_cv_sentences(cv_dir: Path, split: str):
    rows = list(csv.DictReader(open(cv_dir / f"transcript/pt/{split}.tsv", encoding="utf8"),
                               delimiter="\t"))
    out = []
    for r in rows:
        variant = (r.get("variant") or "").lower()
        if "portugal" in variant or "galego" in variant or "durien" in variant:
            continue  # explicit non-Brazilian
        text = (r.get("sentence") or "").strip()
        words = [w for w in re.split(r"\s+", text) if w]
        if len(words) < 3 or not re.search(r"[A-Za-zÀ-ÿ]", text):
            continue  # fragments, brand names, foreign text
        if not (8 <= len(text) <= 160):
            continue
        out.append(text)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv-dir", required=True)
    ap.add_argument("--out-dir", default="data/corpus")
    ap.add_argument("--seed", type=int, default=20261117)
    args = ap.parse_args()
    cv_dir = Path(args.cv_dir)
    out_dir = Path(args.out_dir)
    rng = random.Random(args.seed)

    cv = {s: load_cv_sentences(cv_dir, s) for s in ("train", "dev", "test")}
    for s, rows in cv.items():
        rng.shuffle(rows)
        print(f"cv {s}: {len(rows)} pt-BR sentences")

    # batch1
    conv = converse.generate(BATCH1_CONVERSE, seed=args.seed)
    b1 = [("converse", s) for s in conv]
    cv_pool = []
    for s in ("train", "dev", "test"):
        cv_pool += [("cv", t) for t in cv[s][: BATCH1_CV // 3 + (1 if s != "test" else 0)]]
    rng.shuffle(cv_pool)
    b1 += cv_pool[:BATCH1_CV]
    rng.shuffle(b1)

    # batch2
    rest = []
    used = set(t for _, t in b1)
    for s in ("train", "dev", "test"):
        rest += [("cv", t) for t in cv[s] if t not in used]
    rng.shuffle(rest)
    b2 = rest[:BATCH2_CV]
    b2 += [("templates", t) for t in templates.generate(BATCH2_TEMPLATES, seed=args.seed)]
    rng.shuffle(b2)

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        def write(rows, batch, split):
            path = out_dir / f"batch{batch}" / f"{split}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            table = pa.table({
                "id": [f"broca{batch:02d}_{i:05d}" for i in range(len(rows))],
                "batch": [batch] * len(rows),
                "split": [split] * len(rows),
                "text": [t for _, t in rows],
                "source": [s for s, _ in rows],
            })
            pq.write_table(table, path)

        # Split batch1 by sentence: 83% / 7.5% / 9.5% (same as original).
        n = len(b1)
        cuts = (int(n * 0.83), int(n * 0.905))
        parts = [(b1[:cuts[0]], "train"), (b1[cuts[0]:cuts[1]], "val"), (b1[cuts[1]:], "test")]
        for rows, split in parts:
            write(rows, 1, split)
            print(f"batch1 {split}: {len(rows)} sentences")
        write(b2, 2, "train")
        print(f"batch2 train: {len(b2)} sentences")
    except ImportError:
        # Fallback so the corpus can be built without pyarrow.
        for batch, rows in ((1, b1), (2, b2)):
            path = out_dir / f"batch{batch}" / "sentences.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf8") as f:
                for i, (source, text) in enumerate(rows):
                    f.write(json.dumps({"id": f"broca{batch:02d}_{i:05d}",
                                        "batch": batch, "text": text, "source": source},
                                       ensure_ascii=False) + "\n")
            print(f"wrote {path}")

    stats = {
        "seed": args.seed,
        "batch1": len(b1),
        "batch2": len(b2),
        "cv_per_split": {s: len(v) for s, v in cv.items()},
    }
    (out_dir / "corpus_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
