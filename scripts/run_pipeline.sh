#!/usr/bin/env bash
# The end-to-end chain from finished audio to release artifacts:
#
#   data/audio  ->  data/dataset  ->  data/features  ->  data/annotations
#              ->  runs/tiny-broca (training)  ->  runs/eval  ->  releases/model.onnx
#
# Every stage resumes where it left off, so the script can be re-run after an
# interruption. Run it (and caffeinate) from the repository root once the
# generation finishes.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

echo "== 1/6 assemble dataset (corpus x accepted clips -> Parquet)"
"$PY" scripts/assemble_dataset.py --audio data/audio --corpus data/corpus \
  --refs data/refs --output data/dataset

echo "== 2/6 prepare features (mel, pitch, energy per clip)"
for split in train validation test; do
  "$PY" -m tiny_broca.prepare \
    --inputs "data/dataset/${split}.parquet" --split "$split" \
    --output data/features --resume
done

echo "== 3/6 annotate (forced alignment, letter durations)"
REV="$("$PY" -c "
import hashlib, pathlib
h = hashlib.sha256()
for p in sorted(pathlib.Path('data/dataset').glob('*.parquet')):
    h.update(p.read_bytes())
print(h.hexdigest()[:16])")"
echo "   dataset revision: $REV"
"$PY" -m tiny_broca.annotate --features data/features --output data/annotations \
  --dataset-revision "$REV" --device mps --resume

echo "== 4/6 train (104k steps, 30-minute chunks)"
# The MPS allocator keeps one GPU program per batch shape and never frees
# them; with variable-length batches a single invocation grows without
# bound (observed: 86 GB). Chunking bounds memory (each chunk saves,
# exits, and resumes from last.pt; rollover is lossless) and the
# high-watermark ratio keeps the MPS pool from eating system RAM.
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.3
STEPS=104000
while true; do
  RESUME=()
  if [ -f runs/tiny-broca/last.pt ]; then
    RESUME=(--resume runs/tiny-broca/last.pt)
  elif [ -n "$(ls -A runs/tiny-broca 2>/dev/null)" ]; then
    # Debris from a run that died before saving a checkpoint: start fresh.
    rm -rf runs/tiny-broca
  fi
  "$PY" -m tiny_broca.train \
    --train data/annotations/train.jsonl \
    --validation data/annotations/validation.jsonl \
    --output runs/tiny-broca --device mps --steps "$STEPS" \
    --max-seconds 1800 \
    ${RESUME[@]+"${RESUME[@]}"}
  STEP=$("$PY" -c 'import torch; print(torch.load("runs/tiny-broca/last.pt", map_location="cpu", weights_only=True)["step"])')
  echo "chunk done at step $STEP"
  [ "$STEP" -ge "$STEPS" ] && break
done

echo "== 5/6 evaluate (Whisper pt WER on validation, test and novel sentences)"
"$PY" -m tiny_broca.evaluate \
  --checkpoint runs/tiny-broca/inference.pt \
  --annotations data/annotations \
  --output runs/eval --tag final --device mps

echo "== 6/6 ONNX export"
"$PY" -m tiny_broca.export \
  --checkpoint runs/tiny-broca/inference.pt \
  --output releases/model.onnx

echo "pipeline complete: runs/tiny-broca, runs/eval, releases/model.onnx"
