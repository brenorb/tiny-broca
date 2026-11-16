#!/usr/bin/env bash
# Release orchestration for tiny-broca.
#
#   bash scripts/publish.sh
#
# Stages the release artifacts in the exact layout the two Hugging Face cards
# promise, fills in the cards' pending numbers from the run's outputs, uploads
# both repositories, and deploys the web demo to Cloudflare Pages.
#
# Prerequisites: scripts/run_pipeline.sh has run to completion, `hf` is
# authenticated, and `wrangler` is installed. The repos stay private until the
# final public flip, which is done by hand.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
DATE=$(date +%F)
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
DS="$STAGE/dataset"
MK="$STAGE/model"

echo "== fill in the card numbers"
"$PY" - "$DATE" <<'PYEOF'
import json, re, sys
from pathlib import Path
sys.path.insert(0, "src")

date = sys.argv[1]
ds_card = Path("docs/dataset-card/README.md")
mk_card = Path("docs/model-card/README.md")

import pyarrow.parquet as pq
total = hours = 0
for split in ("train", "validation", "test"):
    t = pq.read_table(f"data/dataset/{split}.parquet")
    total += len(t)
    hours += sum(t.column("duration_seconds").to_pylist()) / 3600
rejected = sum(1 for _ in Path("data/audio").glob("*/review.jsonl")
               for line in open(_) if line.strip())
annotations = rejected_align = 0
for split in ("train", "validation", "test"):
    m = Path(f"data/annotations/{split}.jsonl")
    if m.exists():
        annotations += sum(1 for line in m.open() if line.strip())
    r = Path(f"data/annotations/{split}.rejected.jsonl")
    if r.exists():
        rejected_align += sum(1 for line in r.open() if line.strip())

text = ds_card.read_text(encoding="utf-8")
text = re.sub(r"\| Clips \| .*", f"| Clips | {total:,} accepted of 17,968 generated |", text)
text = re.sub(r"\| Duration \| about \d+ hours \|", f"| Duration | {hours:.1f} hours |", text)
text = re.sub(r" The\s+rejections are listed in `generation/\*/review\.jsonl`\.\s*\(Counts pending the\s*first full run\.\)",
              f" {rejected:,} clips failed twice and are listed in `generation/*/review.jsonl`.", text)
text = text.replace("(Pending the first full run.)",
                    f"{annotations:,} clips are annotated; {rejected_align:,} failed the alignment checks.")
ds_card.write_text(text, encoding="utf-8")

report = json.loads(Path("runs/eval/report.json").read_text(encoding="utf-8"))
sentences = {}
for s in report["samples"]:
    if s["group"] != "novel":
        continue
    number = int(s["id"].rsplit("_", 1)[0].removeprefix("novel"))
    sentences.setdefault(number, {"text": s["text"]})[s["voice"]] = (
        f'<audio controls src="https://huggingface.co/brenorb/tiny-broca/'
        f'resolve/main/samples/{s["voice"]}-{number}.wav"></audio>')
sample_rows = "\n".join(
    f"| {sentences[n]['text']} | " + " | ".join(
        sentences[n].get(v, "–") for v in ("F1", "M1")) + " |"
    for n in sorted(sentences))
mk = mk_card.read_text(encoding="utf-8")
mk = mk.replace("SAMPLE_ROWS", sample_rows)
def pct(value): return "–" if value is None else f"{value:.1%}"
summary = report["summary"]
eval_lines = [
    "Word error rate of synthesized speech, transcribed with Whisper "
    f"(multilingual, pt, `{report['asr_model']}`).",
    "",
    "| | generated | reference |",
    "|---|---:|---:|",
]
for group in ("validation", "test", "novel"):
    if group not in summary:
        continue
    all_v = summary[group].get("all", {})
    eval_lines.append(f"| {group} | {pct(all_v.get('generated_wer'))} | {pct(all_v.get('reference_wer'))} |")
mk = re.sub(r"Word error rate of synthesized speech.*?\(Pending the training run\)\.",
            "\n".join(eval_lines), mk, flags=re.S)
mk = mk.replace("| Hardware | |", "| Hardware | Apple M4 Pro, 18-core, MPS |")
mk = re.sub(r"\| Steps \| [\d,]+ \|", f"| Steps | {report['step']:,} |", mk)
mk = re.sub(r"trained from scratch on about \d+ hours", f"trained from scratch on {hours:.0f} hours", mk)
mk_card.write_text(mk, encoding="utf-8")
print(f"dataset card: {total:,} clips, {hours:.1f} h, {rejected:,} generation-rejected, "
      f"{annotations:,} annotated, {rejected_align:,} alignment-rejected")
print(f"model card: step {report['step']}, {len(sentences)} novel sentences")
PYEOF

echo "== stage dataset"
mkdir -p "$DS/data" "$DS/batch2/data" "$DS/references" "$DS/generation" "$DS/annotations"
"$PY" - "$DATE" "$DS" <<'PYEOF'
import sys
from pathlib import Path
import pyarrow.parquet as pq
date, ds = sys.argv[1], Path(sys.argv[2])
for split in ("train", "validation", "test"):
    table = pq.read_table(f"data/dataset/{split}.parquet")
    pq.write_table(table.filter(table["batch"] == 1), ds / f"data/{split}-{date}.parquet")
    if split == "train":
        pq.write_table(table.filter(table["batch"] == 2), ds / f"batch2/data/train-{date}.parquet")
    print(split, len(table))
PYEOF
cp data/refs/F1.wav data/refs/M1.wav data/refs/voices.json "$DS/references/"
cp scripts/generate_audio.py "$DS/generation/generate_audio.py"
for voice in data/audio/*/; do
  v=$(basename "$voice")
  [ -f "$voice/review.jsonl" ] || continue
  mkdir -p "$DS/generation/$v"
  cp "$voice/review.jsonl" "$DS/generation/$v/review.jsonl"
done
cp data/annotations/*.jsonl "$DS/annotations/"
cp docs/dataset-card/README.md "$DS/README.md"
cp docs/dataset-card/LICENSE.md "$DS/LICENSE.md"

echo "== stage model"
mkdir -p "$MK/onnx" "$MK/samples"
cp runs/tiny-broca/inference.pt "$MK/tiny-broca.pt"
cp releases/model.onnx "$MK/onnx/model.onnx"
cp runs/tiny-broca/config.json "$MK/config.json"
cp src/tiny_broca/text.py "$MK/text.py"
"$PY" - "$MK" <<'PYEOF'
import json, shutil, sys
from pathlib import Path
mk = Path(sys.argv[1])
report = json.loads(Path("runs/eval/report.json").read_text(encoding="utf-8"))
count = 0
for s in report["samples"]:
    if s["group"] != "novel":
        continue
    number = int(s["id"].rsplit("_", 1)[0].removeprefix("novel"))
    shutil.copy(Path("runs/eval/audio") / s["generated"],
                mk / "samples" / f"{s['voice']}-{number}.wav")
    count += 1
print(f"staged {count} samples")
PYEOF
cp docs/model-card/README.md "$MK/README.md"
cp LICENSE "$MK/LICENSE.md"

echo "== upload"
hf upload brenorb/tiny-broca "$DS" . --type dataset \
  --commit-message "Dataset: $DATE" \
  --commit-description "Synthetic two-voice Brazilian Portuguese speech, assembled by scripts/assemble_dataset.py"
hf upload brenorb/tiny-broca "$MK" . --type model \
  --commit-message "Model: $DATE" \
  --commit-description "Trained 104k steps on the tiny-broca dataset (see runs/eval/report.json)"

echo "== web demo"
bash web/fetch_assets.sh .
if wrangler pages project get tiny-broca-demo >/dev/null 2>&1; then
  wrangler pages deploy web/public --project-name tiny-broca-demo
else
  wrangler pages project create tiny-broca-demo
  wrangler pages deploy web/public --project-name tiny-broca-demo
fi

echo "published. The repos stay private until the public flip."
