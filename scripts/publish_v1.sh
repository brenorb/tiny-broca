#!/usr/bin/env bash
# Publish tiny-broca v1 (run only after listening to the comparison samples).
#
#   bash scripts/publish_v1.sh [--dry-run]
#
# GitHub: pushes main with its commits as they are, then tags the release v1.
# Hugging Face: scripts/publish_hf.py uploads one commit per logical change to
# the model and dataset repos (brenorb/tiny-broca). The current revisions are
# tagged v0 first, and the v0 dataset files stay as they are.
# Web demo: fetches the new assets and deploys to Cloudflare Pages.
#
# Safe to re-run after an interruption: finished steps are skipped.
# Prerequisites: scripts/v1_pipeline.sh finished, everything committed,
# `hf auth login` done (the stored token must be valid), `wrangler` logged in.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
DRY="${1:-}"
DATE=$("$PY" -c 'import json; print(json.load(open("data/dataset-v1/dataset.json"))["generated_at"][:10])')  # same file names on a re-run
if [ "$DRY" != "--dry-run" ]; then
  [ "$(git branch --show-current)" = main ] || { echo "publish from main" >&2; exit 1; }
  [ -z "$(git status --porcelain)" ] || { echo "commit everything first: tag v1 must match what is published" >&2; exit 1; }
fi
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
DS="$STAGE/dataset"
MK="$STAGE/model"
RUN=runs/tiny-broca-v1
EVAL=runs/eval-v1

echo "== stage dataset additions"
mkdir -p "$DS/batch4/data" "$DS/batch5/data" "$DS/annotations-v1" "$DS/generation" "$DS/references"
"$PY" - "$DATE" "$DS" <<'PYEOF'
import sys
from pathlib import Path
import pyarrow.compute as pc
import pyarrow.parquet as pq
date, ds = sys.argv[1], Path(sys.argv[2])
for split in ("train", "validation", "test"):
    table = pq.read_table(f"data/dataset-v1/{split}.parquet")
    for batch in (4, 5):
        part = table.filter(pc.equal(table["batch"], batch))
        if len(part):
            pq.write_table(part, ds / f"batch{batch}/data/{split}-{date}.parquet")
            print(f"batch{batch} {split}: {len(part)} clips")
PYEOF
cp data/annotations-v1/*.jsonl "$DS/annotations-v1/"
cp data/corpus/v1_superseded.txt "$DS/v1_superseded.txt"
cp data/refs/voices.json "$DS/references/voices.json"  # v0 credited F1 to M1's source clip
cp scripts/generate_audio.py scripts/rescue_clips.py scripts/build_v1_corpus.py "$DS/generation/"
cp docs/dataset-card/README.md "$DS/README.md"

echo "== stage model"
mkdir -p "$MK/onnx" "$MK/samples"
cp "$RUN/inference.pt" "$MK/tiny-broca.pt"
cp releases/v1/model.onnx "$MK/onnx/model.onnx"
"$PY" scripts/build_release_config.py --config "$RUN/config.json" --checkpoint "$RUN/inference.pt" --output releases/v1/config.json
cp releases/v1/config.json "$MK/config.json"
cp src/tiny_broca/text.py "$MK/text.py"
cp runs/samples-v1/*.wav "$MK/samples/"
cp docs/model-card/README.md "$MK/README.md"
cp LICENSE "$MK/LICENSE.md"
find "$STAGE" -type f | sed "s|$STAGE/||" | sort

echo "== check the commit plan"
"$PY" scripts/publish_hf.py --dataset "$DS" --model "$MK" --dry-run
if [ "$DRY" = "--dry-run" ]; then
  echo "dry run: nothing uploaded"
  exit 0
fi

echo "== GitHub: the commits as they are"
git push origin main

echo "== Hugging Face: one commit per change"
"$PY" scripts/publish_hf.py --dataset "$DS" --model "$MK"

echo "== tag v1 on GitHub"
git rev-parse -q --verify refs/tags/v1 >/dev/null || git tag -a v1 -m "tiny-broca v1"
[ "$(git rev-parse 'v1^{commit}')" = "$(git rev-parse HEAD)" ] || { echo "tag v1 is not the published commit" >&2; exit 1; }
git push origin v1

echo "== web demo"
bash web/fetch_assets.sh "$PWD"
wrangler pages deploy web/public --project-name tiny-broca-demo

echo "published v1"
