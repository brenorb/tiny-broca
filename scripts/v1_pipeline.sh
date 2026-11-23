#!/usr/bin/env bash
# tiny-broca v1: from the generated audio to the release artifacts.
#
#   data/audio (v0 clips, minus the superseded sentences)
#   + data/audio-v1 (batch4 new coverage, batch5 regenerated v0 sentences)
#     -> data/dataset-v1 -> data/features-v1 -> data/annotations-v1 (front end v2)
#     -> runs/tiny-broca-v1 (fine-tune of v0's 118k checkpoint, switched excitation)
#     -> runs/eval-v1 -> releases/v1/model.onnx
#
# Run after scripts/generate_audio.py (--teacher-text spoken) and
# scripts/rescue_clips.py have finished for data/audio-v1. Every stage resumes
# where it left off; training runs in 30-minute chunks to bound MPS memory.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
STEPS="${STEPS:-148000}"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1

echo "== 1/6 assemble dataset"
if [ ! -f data/dataset-v1/dataset.json ]; then
  "$PY" scripts/assemble_dataset.py --audio data/audio data/audio-v1 --exclude data/corpus/v1_superseded.txt \
    --corpus data/corpus --refs data/refs --output data/dataset-v1
fi

echo "== 2/6 prepare features"
for split in train validation test; do
  "$PY" -m tiny_broca.prepare --inputs "data/dataset-v1/${split}.parquet" --split "$split" \
    --output data/features-v1 --resume
done

echo "== 3/6 annotate (front end v2)"
REV="$("$PY" -c "
import hashlib, pathlib
h = hashlib.sha256()
for p in sorted(pathlib.Path('data/dataset-v1').glob('*.parquet')):
    h.update(p.read_bytes())
print(h.hexdigest()[:16])")"
echo "   dataset revision: $REV"
"$PY" -m tiny_broca.annotate --features data/features-v1 --output data/annotations-v1 \
  --dataset-revision "$REV" --device mps --resume

echo "== 4/6 train (fine-tune 118k -> ${STEPS}, 30-minute chunks)"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.4
while true; do
  ARGS=()
  if [ -f runs/tiny-broca-v1/last.pt ]; then
    ARGS=(--resume runs/tiny-broca-v1/last.pt)
  else
    if [ -n "$(ls -A runs/tiny-broca-v1 2>/dev/null)" ]; then
      rm -rf runs/tiny-broca-v1
    fi
    # New front end and annotation run, same symbol inventory: branch from v0.
    ARGS=(--init-from runs/tiny-broca-ft2/last.pt --new-data)
  fi
  "$PY" -m tiny_broca.train \
    --train data/annotations-v1/train.jsonl \
    --validation data/annotations-v1/validation.jsonl \
    --output runs/tiny-broca-v1 --device mps --steps "$STEPS" \
    --max-seconds 1800 --switched-excitation \
    ${ARGS[@]+"${ARGS[@]}"}
  STEP=$("$PY" -c 'import torch; print(torch.load("runs/tiny-broca-v1/last.pt", map_location="cpu", weights_only=True)["step"])')
  echo "chunk done at step $STEP"
  [ "$STEP" -ge "$STEPS" ] && break
done

echo "== 5/6 evaluate"
"$PY" -m tiny_broca.evaluate \
  --checkpoint runs/tiny-broca-v1/inference.pt \
  --annotations data/annotations-v1 \
  --output runs/eval-v1 --tag v1 --device mps --per-voice 25

echo "== 6/6 ONNX export"
mkdir -p releases/v1
"$PY" -m tiny_broca.export \
  --checkpoint runs/tiny-broca-v1/inference.pt \
  --output releases/v1/model.onnx

echo "pipeline complete: runs/tiny-broca-v1, runs/eval-v1, releases/v1/model.onnx"
