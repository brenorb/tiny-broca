#!/usr/bin/env python
"""Build the v1 corpus batches from the v0 corpus.

batch5 holds the v0 sentences whose audio must be generated again:

* every sentence whose spoken form differs from its written form (digits,
  money, dates, symbols, acronyms, loanwords). v0's teacher read the written
  text, and an audit of its clips found it often read amounts digit by digit
  or garbled them ("R$ 7.586,29" -> "resatras 58862919"), so the audio and the
  student's tokens disagreed; v1's teacher reads ``spoken_form()``;
* every sentence with a template grammar error ("vou à a escola", "na o
  aeroporto", "sobiu", "Moro no Rua"), with the text corrected.

The superseded v0 sentences are written to data/corpus/v1_superseded.txt as
``b<batch>_<split>_<position>`` keys (corpus ids repeat across splits), so
that the dataset assembly leaves their v0 clips out.

batch4 (new sentences for the v1 coverage gaps) is converted from
data/corpus/batch4/sentences.jsonl when it exists: a stable hash of the text
assigns 90% of each category to train and 5% each to validation and test.

    python scripts/build_v1_corpus.py
"""
import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from tiny_broca.text import spoken_form

_PREPOSITIONS = {"na": "n", "no": "n", "nas": "n", "nos": "n", "da": "d", "do": "d", "das": "d", "dos": "d",
                 "pela": "pel", "pelo": "pel", "pelas": "pel", "pelos": "pel"}
_TAILS = ("faz sentido", "não é?", "viu?", "da última vez", "com certeza", "beleza?", "sabe?", "né?", "hein?")
# v0 template sentences too broken to keep: a CPF with one-digit groups.
_DROP = re.compile(r"\b\d\.\d\.\d-\d{2}\b")


def fix_template_text(text):
    """Correct the v0 conversational template errors."""
    def contract(m):
        prep, article = m.group(1), m.group(2)
        head = _PREPOSITIONS[prep.lower()]
        word = head + article.lower()
        return word.capitalize() if prep[0].isupper() else word
    text = re.sub(r"\b([Nn]a|[Nn]o|[Nn]as|[Nn]os|[Dd]a|[Dd]o|[Dd]as|[Dd]os|[Pp]ela|[Pp]elo|[Pp]elas|[Pp]elos) (a|o|as|os)\b",
                  contract, text)
    text = re.sub(r"\bà a\b", "à", text)
    text = re.sub(r"\bà o\b", "ao", text)
    text = re.sub(r"\bà as\b", "às", text)
    text = re.sub(r"\bà os\b", "aos", text)
    text = re.sub(r"\b([Ss])obiu\b", r"\1ubiu", text)
    text = re.sub(r"\b([Nn])o (Rua|Avenida|Praça|Travessa|Alameda|Estrada)\b", r"\1a \2", text)
    text = re.sub(r"\b([Oo]) o\b", r"\1", text)
    text = re.sub(r"\b([Aa]) a\b", r"\1", text)
    for tail in _TAILS:
        question = tail.endswith("?") or tail in ("faz sentido",)
        ending = "" if tail.endswith("?") else ("?" if question else ".")
        text = re.sub(r"(\w) " + re.escape(tail) + r"$", r"\1, " + tail + ending, text)
    return text


def fold(text):
    return re.sub(r"[^\w]+", " ", text.lower()).strip()


def write_split_files(rows, directory, batch):
    directory.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([pa.field("id", pa.string()), pa.field("batch", pa.int64()), pa.field("split", pa.string()),
                        pa.field("text", pa.string()), pa.field("source", pa.string())])
    counts = {}
    for split in ("train", "validation", "test"):
        part = [r for r in rows if r["split"] == split]
        pq.write_table(pa.Table.from_pylist(part, schema=schema), directory / f"{split}.parquet")
        counts[split] = len(part)
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    args = ap.parse_args()

    regen, superseded = [], []
    for batch in (1, 2, 3):
        for path in sorted(args.corpus.glob(f"batch{batch}/*.parquet")):
            for position, row in enumerate(pq.read_table(path).to_pylist()):
                fixed = fix_template_text(row["text"])
                if fixed != row["text"] or fold(spoken_form(row["text"])) != fold(row["text"]):
                    superseded.append(f"b{batch}_{path.stem}_{position:05d}")
                    if _DROP.search(fixed):
                        continue
                    regen.append({"id": row["id"] + "r", "batch": 5, "split": path.stem, "text": fixed,
                                  "source": row["source"] + "+v1"})
    # Append-only: clip ids are row positions, so rows already generated keep
    # their place and sentences that newly qualify go at the end of their split.
    existing = []
    for split in ("train", "validation", "test"):
        path = args.corpus / "batch5" / f"{split}.parquet"
        if path.exists():
            existing += pq.read_table(path).to_pylist()
    known = {(r["split"], r["text"]) for r in existing}
    regen = existing + [r for r in regen if (r["split"], r["text"]) not in known]
    counts = write_split_files(regen, args.corpus / "batch5", 5)
    (args.corpus / "v1_superseded.txt").write_text("\n".join(superseded) + "\n")
    print(json.dumps({"batch5": counts, "superseded": len(superseded)}))

    sentences = args.corpus / "batch4" / "sentences.jsonl"
    if sentences.exists():
        rows, seen = [], set()
        per_category = defaultdict(int)
        for line in sentences.read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            text = item["text"].strip()
            if fold(text) in seen:
                continue
            seen.add(fold(text))
            bucket = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) % 20
            split = "validation" if bucket == 0 else "test" if bucket == 1 else "train"
            per_category[item.get("category", "other")] += 1
            rows.append({"id": f"broca04_{len(rows):05d}", "batch": 4, "split": split, "text": text,
                         "source": "v1-" + item.get("category", "other")})
        print(json.dumps({"batch4": write_split_files(rows, args.corpus / "batch4", 4),
                          "categories": dict(per_category)}))


if __name__ == "__main__":
    main()
