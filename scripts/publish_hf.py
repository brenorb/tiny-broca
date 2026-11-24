#!/usr/bin/env python
"""Upload the staged v1 files to Hugging Face, one commit per logical change.

    python scripts/publish_hf.py --dataset STAGE/dataset --model STAGE/model [--dry-run]

Files that only work together travel together (weights, ONNX, config and front
end), and each card goes last because it describes what the commits before it
added. The current revisions are tagged v0 first and the result is tagged v1.
Re-running after an interruption skips what is already on the Hub.
"""
import argparse
from fnmatch import fnmatch
from pathlib import Path

REPO = "brenorb/tiny-broca"
COMMITS = (
    ("dataset", ("batch4/*",), "Data: batch4, new coverage"),
    ("dataset", ("batch5/*", "v1_superseded.txt"), "Data: batch5, v0 sentences regenerated from spoken forms"),
    ("dataset", ("annotations-v1/*",), "Data: v1 annotations"),
    ("dataset", ("references/voices.json",), "Data: credit F1 to its own source clip"),
    ("dataset", ("generation/*",), "Scripts: v1 corpus and generation"),
    ("dataset", ("README.md",), "Card: v1 dataset"),
    ("model", ("tiny-broca.pt", "onnx/*", "config.json", "text.py"), "Model: v1 weights, ONNX, config and front end"),
    ("model", ("samples/*",), "Samples: v1 audio"),
    ("model", ("README.md", "LICENSE.md"), "Card: v1 results"),
)


def plan(stages):
    """The commits with their files. Stops if a staged file is in none or in two."""
    staged = {kind: {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
              for kind, root in stages.items()}
    commits, claimed = [], {kind: set() for kind in stages}
    for kind, patterns, message in COMMITS:
        files = sorted(f for f in staged[kind] if any(fnmatch(f, pattern) for pattern in patterns))
        if not files:
            raise SystemExit(f"nothing staged for {kind} commit: {message}")
        if claimed[kind] & set(files):
            raise SystemExit(f"{kind} files in two commits: {sorted(claimed[kind] & set(files))}")
        claimed[kind].update(files)
        commits.append((kind, patterns, message, files))
    left = {kind: sorted(staged[kind] - claimed[kind]) for kind in stages if staged[kind] - claimed[kind]}
    if left:
        raise SystemExit(f"staged files in no commit: {left}")
    return commits


def tag(api, name, message):
    for kind in ("dataset", "model"):
        api.create_tag(REPO, tag=name, repo_type=kind, tag_message=message, exist_ok=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true", help="check and print the commits, upload nothing")
    args = ap.parse_args()
    stages = {"dataset": args.dataset, "model": args.model}
    commits = plan(stages)
    for kind, _, message, files in commits:
        megabytes = sum((stages[kind] / f).stat().st_size for f in files) / 1e6
        print(f"{kind:7s} {message} ({len(files)} files, {megabytes:.1f} MB)")
    if args.dry_run:
        return

    from huggingface_hub import HfApi
    api = HfApi()
    tag(api, "v0", "v0 release (118k steps)")  # before the first commit moves main
    for kind, patterns, message, _ in commits:
        api.upload_folder(repo_id=REPO, repo_type=kind, folder_path=stages[kind],
                          allow_patterns=list(patterns), commit_message=message)
    tag(api, "v1", "v1 release")


if __name__ == "__main__":
    main()
