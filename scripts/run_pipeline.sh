#!/usr/bin/env bash
# The end-to-end chain from finished audio to a trained checkpoint:
#
#   data/audio  ->  data/dataset  ->  data/features  ->  data/annotations  ->  runs/tiny-broca
#
# Every stage resumes where it left off, so the script can be re-run after an
# interruption. Run it (and caffeinate) from the repository root once the
# generation finishes.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

echo "== 1/4 assemble dataset (corpus x accepted clips -> Parquet)"
"$PY" scripts/assemble_dataset.py --audio data/audio --corpus data/corpus \
  --refs data/refs --output data/dataset

echo "== 2/4 prepare features (mel, pitch, energy per clip)"
for split in train validation test; do
  "$PY" -m tiny_broca.prepare \
    --inputs "data/dataset/${split}.parquet" --split "$split" \
    --output data/features --resume
done

echo "== 3/4 annotate (forced alignment, letter durations)"
REV="$("$PY" -c "
import hashlib, pathlib
h = hashlib.sha256()
for p in sorted(pathlib.Path('data/dataset').glob('*.parquet')):
    h.update(p.read_bytes())
print(h.hexdigest()[:16])")"
echo "   dataset revision: $REV"
"$PY" -m tiny_broca.annotate --features data/features --output data/annotations \
  --dataset-revision "$REV" --device mps --resume

echo "== 4/4 train (104k steps)"
"$PY" -m tiny_broca.train \
  --train data/annotations/train.jsonl \
  --validation data/annotations/validation.jsonl \
  --output runs/tiny-broca --device mps --steps 104000

echo "pipeline complete: runs/tiny-broca"
