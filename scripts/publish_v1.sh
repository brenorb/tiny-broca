#!/usr/bin/env bash
# Publish tiny-broca v1 (run only after listening to the comparison samples).
#
#   bash scripts/publish_v1.sh [--dry-run]
#
# Model repo (brenorb/tiny-broca): tags the current revision "v0", then
# replaces the weights, ONNX, config, front end, samples and card with v1's.
# Dataset repo (brenorb/tiny-broca, type dataset): adds batch4 (new coverage),
# batch5 (regenerated v0 sentences), the v1 annotations and the list of v0
# sentences v1 supersedes; the v0 files stay as they are.
# Web demo: fetches the new assets and deploys to Cloudflare Pages.
#
# Prerequisites: scripts/v1_pipeline.sh finished, `hf auth login` done (the
# stored token must be valid), `wrangler` logged in.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
DRY="${1:-}"
DATE=$(date +%F)
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

if [ "$DRY" = "--dry-run" ]; then
  echo "dry run: nothing uploaded"
  exit 0
fi

echo "== tag the v0 model revision, then upload"
"$PY" - <<'PYEOF'
from huggingface_hub import HfApi
api = HfApi()
refs = api.list_repo_refs("brenorb/tiny-broca")
if not any(t.name == "v0" for t in refs.tags):
    api.create_tag("brenorb/tiny-broca", tag="v0", tag_message="v0 release (118k steps)")
refs = api.list_repo_refs("brenorb/tiny-broca", repo_type="dataset")
if not any(t.name == "v0" for t in refs.tags):
    api.create_tag("brenorb/tiny-broca", tag="v0", repo_type="dataset", tag_message="v0 dataset")
PYEOF
hf upload brenorb/tiny-broca "$DS" . --type dataset \
  --commit-message "v1 data: $DATE" \
  --commit-description "batch4 (new coverage), batch5 (v0 sentences regenerated from spoken forms), v1 annotations"
hf upload brenorb/tiny-broca "$MK" . --type model \
  --commit-message "v1: $DATE" \
  --commit-description "Front end v2, regenerated number data, switched excitation (see docs/v1.md on GitHub)"
"$PY" - <<'PYEOF'
from huggingface_hub import HfApi
api = HfApi()
api.create_tag("brenorb/tiny-broca", tag="v1", tag_message="v1 release")
api.create_tag("brenorb/tiny-broca", tag="v1", repo_type="dataset", tag_message="v1 dataset")
PYEOF

echo "== web demo"
bash web/fetch_assets.sh "$PWD"
wrangler pages deploy web/public --project-name tiny-broca-demo

echo "published v1"
