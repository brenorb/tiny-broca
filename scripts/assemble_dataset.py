#!/usr/bin/env python3
"""Assemble the training dataset from the corpus and the generated audio.

Joins the accepted clips in the generation state database with the corpus
sentences (for source provenance) and the voice references (for teacher
provenance), and writes one Parquet file per split with the audio embedded,
in the layout the HF dataset and the preparation step consume:

    data/dataset/train.parquet
    data/dataset/validation.parquet
    data/dataset/test.parquet

The generation step is idempotent across resumes, so this script can be run
any time after it completes (or re-run to rebuild the same files).
"""
import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audio", type=Path, nargs="+", default=[Path("data/audio")],
                    help="Generation directories (state.db + audio/); a clip id may appear in only one")
    ap.add_argument("--exclude", type=Path, help="File of b<batch>_<split>_<position> sentence keys whose clips are left out "
                    "(sentences superseded by a regenerated batch)")
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    ap.add_argument("--refs", type=Path, default=Path("data/refs"))
    ap.add_argument("--output", type=Path, default=Path("data/dataset"))
    args = ap.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq

    refs = json.loads((args.refs / "voices.json").read_text())
    excluded = set(args.exclude.read_text().split()) if args.exclude else set()
    clips, total, audio_dir, dropped = [], 0, {}, 0
    for directory in args.audio:
        conn = sqlite3.connect(directory / "state.db")
        for row in conn.execute("SELECT clip_id, voice, batch, split, text, seconds, sr, wer, transcript,"
                                " gen_seconds, rtf FROM clips WHERE status='ok'").fetchall():
            cid, batch, split = row[0], row[2], row[3]
            if f"b{batch}_{split}_{int(cid.rsplit('_', 1)[1]):05d}" in excluded:
                dropped += 1
                continue
            if cid in audio_dir:
                raise ValueError(f"clip {cid} is in both {audio_dir[cid]} and {directory}")
            audio_dir[cid] = directory
            clips.append(row)
        total += conn.execute("SELECT COUNT(*) FROM clips").fetchone()[0]
        conn.close()

    # (batch, split, file row position) -> source; the generator indexed
    # sentences the same way.
    source_map = {}
    for batch in range(1, 10):
        for path in sorted(args.corpus.glob(f"batch{batch}/*.parquet")):
            table = pq.read_table(path)
            for i, source in enumerate(table.column("source").to_pylist()):
                source_map[(batch, path.stem, i)] = source

    args.output.mkdir(parents=True, exist_ok=True)
    splits = {"train": [], "validation": [], "test": []}
    missing_audio = missing_source = 0
    for cid, voice, batch, split, text, seconds, sr, wer, transcript, gen_seconds, rtf in clips:
        index = int(cid.rsplit("_", 1)[1])
        path = audio_dir[cid] / "audio" / f"{cid}.flac"
        if not path.is_file():
            missing_audio += 1
            print(f"skip {cid}: audio file missing", file=sys.stderr)
            continue
        source = source_map.get((batch, split, index))
        if source is None:
            missing_source += 1
            print(f"skip {cid}: no corpus row for ({batch}, {split}, {index})", file=sys.stderr)
            continue
        ref = refs.get(voice, {})
        splits[split].append({
            "id": cid,
            "text_id": f"broca{batch:02d}_{index:05d}",
            "batch": batch,
            "split": split,
            "text": text,
            "source": source,
            "source_license": "cc0-1.0" if source.startswith("cv") else "project-own",
            "source_url": "https://common-voice.org/datasets" if source.startswith("cv") else None,
            "voice": voice,
            "audio": {"bytes": path.read_bytes(), "path": f"audio/{cid}.flac",
                      "sampling_rate": sr},
            "duration_seconds": round(float(seconds), 3),
            "sample_rate": sr,
            "asr_transcript": transcript,
            "normalized_wer": float(wer),
            "teacher_model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            "reference_audio": ref.get("render"),
            "gen_seconds": round(float(gen_seconds), 2),
            "rtf": round(float(rtf), 3),
        })

    audio_field = pa.struct({"bytes": pa.binary(), "path": pa.string(),
                             "sampling_rate": pa.int64()})
    schema = pa.schema([
        pa.field("id", pa.string()), pa.field("text_id", pa.string()),
        pa.field("batch", pa.int64()), pa.field("split", pa.string()),
        pa.field("text", pa.string()), pa.field("source", pa.string()),
        pa.field("source_license", pa.string()), pa.field("source_url", pa.string()),
        pa.field("voice", pa.string()), pa.field("audio", audio_field),
        pa.field("duration_seconds", pa.float64()), pa.field("sample_rate", pa.int64()),
        pa.field("asr_transcript", pa.string()), pa.field("normalized_wer", pa.float64()),
        pa.field("teacher_model", pa.string()), pa.field("reference_audio", pa.string()),
        pa.field("gen_seconds", pa.float64()), pa.field("rtf", pa.float64()),
    ])
    summary = {}
    for split, rows in splits.items():
        table = pa.Table.from_pylist(rows, schema=schema)
        out = args.output / f"{split}.parquet"
        pq.write_table(table, out)
        hours = sum(r["duration_seconds"] for r in rows) / 3600
        summary[split] = {"clips": len(rows), "hours": round(hours, 3),
                          "file": str(out), "mb": round(out.stat().st_size / 1e6, 1)}
    summary.update({"generated_clips": len(clips), "state_db_rows": total, "excluded_superseded": dropped,
                    "skipped_missing_audio": missing_audio,
                    "skipped_missing_source": missing_source,
                    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    (args.output / "dataset.json").write_text(json.dumps(summary, indent=2))
    for split, s in summary.items():
        if isinstance(s, dict) and "clips" in s:
            print(f"{split}: {s['clips']} clips, {s['hours']} h, {s['mb']} MB")
    print(f"ok {len(clips)} / {total} state rows, skipped "
          f"{missing_audio} missing audio, {missing_source} missing source")


if __name__ == "__main__":
    main()
