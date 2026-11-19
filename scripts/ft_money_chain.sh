#!/usr/bin/env bash
# Fine-tune chain for the money-amount batch (batch3): fresh features and
# annotation trees, then continue the 104k-step run for 6,000 more steps.
#
# The preparation and annotation stages keep strict identity guards (any
# change of inputs, front end, aligner or dataset revision requires a new
# directory — that is what makes the manifests trustworthy), so the appended
# corpus gets new trees: data/features-ft, data/annotations-ft.
#
# Re-running resumes each stage; the training loop chunks on --max-seconds
# (the MPS allocator never frees per-shape programs, so a single invocation
# grows without bound) and resumes from runs/tiny-broca-ft/last.pt.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

echo "== prepare features (money run, fresh tree)"
for split in train validation test; do
  "$PY" -m tiny_broca.prepare \
    --inputs "data/dataset/${split}.parquet" --split "$split" \
    --output data/features-ft --resume
done

echo "== annotate (money run, fresh tree)"
REV=$("$PY" -c "
import hashlib, pathlib
h = hashlib.sha256()
for p in sorted(pathlib.Path('data/dataset').glob('*.parquet')):
    h.update(p.read_bytes())
print(h.hexdigest()[:16])")
echo "   dataset revision: $REV"
"$PY" -m tiny_broca.annotate --features data/features-ft --output data/annotations-ft \
  --dataset-revision "$REV" --device mps --resume

echo "== fine-tune from the 104k checkpoint (+6,000 steps)"
# Training only: the MPS allocator keeps one GPU program per batch shape and
# never frees them; the watermark keeps the pool from eating system RAM.
# Not for the DSP stages above. torch 2.15 rejects any high watermark below
# 1.4 ("invalid low watermark ratio"); 1.4 is the tightest valid pool.
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.4
STEPS=110000
while true; do
  ARGS=()
  if [ -f runs/tiny-broca-ft/last.pt ]; then
    ARGS=(--resume runs/tiny-broca-ft/last.pt)
  else
    if [ -n "$(ls -A runs/tiny-broca-ft 2>/dev/null)" ]; then
      rm -rf runs/tiny-broca-ft
    fi
    # Fresh directory: branch from the 104k run; the annotation tree is new,
    # the front end (and its symbol inventory) is unchanged.
    ARGS=(--init-from runs/tiny-broca/last.pt --new-data)
  fi
  "$PY" -m tiny_broca.train \
    --train data/annotations-ft/train.jsonl \
    --validation data/annotations-ft/validation.jsonl \
    --output runs/tiny-broca-ft --device mps --steps "$STEPS" \
    --max-seconds 1800 \
    ${ARGS[@]+"${ARGS[@]}"}
  STEP=$("$PY" -c 'import torch; print(torch.load("runs/tiny-broca-ft/last.pt", map_location="cpu", weights_only=True)["step"])')
  echo "chunk done at step $STEP"
  [ "$STEP" -ge "$STEPS" ] && break
done

echo "ft chain complete"
