#!/usr/bin/env python3
"""Merge a targeted annotation run into a base annotation run.

A targeted run (``annotate --only-ids ...`` with different gate limits, into
its own output directory) produces rows stamped with its own fingerprint;
the training data loader requires every row of a manifest to carry that
manifest's run fingerprint, so the two trees cannot be combined in place.

This tool builds a new run directory that is the base run plus the targeted
rows: every row is re-stamped with the merged run's fingerprint, and each
imported row records the gate limits it was admitted under (``qc_gate``
provenance), so the relaxed admission of the money batch stays auditable in
the dataset manifest.

    python scripts/merge_annotations.py \
        --base data/annotations-ft --extra data/annotations-money \
        --output data/annotations-merged

The base run's identity is carried over; the merge is recorded in it.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def stamp(line: str, fingerprint: str) -> str:
    row = json.loads(line)
    row["annotation"] = fingerprint
    return json.dumps(row, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--extra", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    base_meta = json.loads((args.base / "annotations.json").read_text())
    extra_meta = json.loads((args.extra / "annotations.json").read_text())
    if not base_meta.get("complete") or not extra_meta.get("complete"):
        raise SystemExit("both runs must be complete")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"{args.output} is not empty; remove it first")
    args.output.mkdir(parents=True)

    identity = dict(base_meta["identity"])
    identity["merged_from"] = {
        "base": str(args.base.resolve()),
        "extra": str(args.extra.resolve()),
        "extra_gate": extra_meta["identity"].get("gate"),
    }
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

    extra_gate = extra_meta["identity"].get("gate")
    for split in base_meta["counts"]:
        base_rows = []
        for line in (args.base / (split + ".jsonl")).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row["annotation"] = fingerprint
            base_rows.append(row)
        extra_rows = []
        extra_path = args.extra / (split + ".jsonl")
        if extra_path.exists():
            seen = {r["id"] for r in base_rows}
            for line in extra_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["id"] in seen:
                    continue  # the stricter base admission wins for shared clips
                seen.add(row["id"])
                row["annotation"] = fingerprint
                row.setdefault("provenance", {})["qc_gate"] = extra_gate
                extra_rows.append(row)
        rows = base_rows + extra_rows
        with (args.output / (split + ".jsonl")).open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        for name in (split + ".rejected.jsonl",):
            src = args.base / name
            if src.exists():
                shutil.copyfile(src, args.output / name)
        print(f"{split}: {len(base_rows)} base + {len(extra_rows)} extra -> {len(rows)}")

    (args.output / "phonemes.jsonl").write_text((args.base / "phonemes.jsonl").read_text(), encoding="utf-8")
    (args.output / "summary.json").write_text((args.base / "summary.json").read_text(), encoding="utf-8")
    metadata = {"identity": identity, "fingerprint": fingerprint, "complete": True,
                "stage": base_meta.get("stage", "ctc"), "acoustic_root": base_meta.get("acoustic_root"),
                "limits": base_meta.get("limits"),
                "counts": {split: sum(1 for _ in open(args.output / (split + ".jsonl"))) for split in base_meta["counts"]}}
    (args.output / "annotations.json").write_text(json.dumps(metadata, indent=1))
    print(f"merged run: {args.output} (fingerprint {fingerprint[:12]})")


if __name__ == "__main__":
    main()
