#!/usr/bin/env bash
# Round 3: train on the stretched-money dataset (data/dataset-slow).
#
# The Qwen3 teacher rushes long money values; scripts/slow_number_regions.py
# restores the human number pace in the batch3 audio (data/audio/audio-slow),
# and data/dataset-slow carries that corrected reference. Fresh feature and
# annotation trees (the strict identity guards require a new directory), the
# standard alignment gate, then:
#   base  = the 110k-ft run manifest minus the 114 original-pace money rows
#   extra = the freshly annotated (stretched) money rows
# merged into a new run, fine-tuned from the 118k checkpoint for 8,000 steps.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.4

echo "== prepare features (slow run, fresh tree)"
for split in train validation test; do
  "$PY" -m tiny_broca.prepare \
    --inputs "data/dataset-slow/${split}.parquet" --split "$split" \
    --output data/features-slow --resume
done

echo "== annotate (slow run, fresh tree, standard gate)"
REV=$("$PY" -c "
import hashlib, pathlib
h = hashlib.sha256()
for p in sorted(pathlib.Path('data/dataset-slow').glob('*.parquet')):
    h.update(p.read_bytes())
print(h.hexdigest()[:16])")
echo "   dataset revision: $REV"
"$PY" -m tiny_broca.annotate --features data/features-slow --output data/annotations-slow \
  --dataset-revision "$REV" --device mps --resume

echo "== merge: base (110k-ft, minus original-pace money) + stretched money"
"$PY" - <<'EOF'
import json
ids = {json.loads(l)["id"] for l in open("data/annotations-ft/train.jsonl")
       if json.loads(l).get("id", "").startswith("b3_")}
open("data/b3_drop_ids.txt", "w").write("\n".join(sorted(ids)))
print(f"dropping {len(ids)} original-pace money rows from the base")
EOF
"$PY" scripts/merge_annotations.py \
  --base data/annotations-ft --extra data/annotations-slow \
  --output data/annotations-slow-merged --drop-ids data/b3_drop_ids.txt

echo "== fine-tune (118k -> 126k) on the merged manifest"
STEPS=126000
while true; do
  ARGS=()
  if [ -f runs/tiny-broca-ft3/last.pt ]; then
    ARGS=(--resume runs/tiny-broca-ft3/last.pt)
  else
    if [ -n "$(ls -A runs/tiny-broca-ft3 2>/dev/null)" ]; then
      rm -rf runs/tiny-broca-ft3
    fi
    ARGS=(--init-from runs/tiny-broca-ft2/last.pt --new-data)
  fi
  "$PY" -m tiny_broca.train \
    --train data/annotations-slow-merged/train.jsonl \
    --validation data/annotations-slow-merged/validation.jsonl \
    --output runs/tiny-broca-ft3 --device mps --steps "$STEPS" \
    --max-seconds 1800 \
    ${ARGS[@]+"${ARGS[@]}"}
  STEP=$("$PY" -c 'import torch; print(torch.load("runs/tiny-broca-ft3/last.pt", map_location="cpu", weights_only=True)["step"])')
  echo "ft3 chunk done at step $STEP"
  [ "$STEP" -ge "$STEPS" ] && break
done

echo "ft3 chain complete"
