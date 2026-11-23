#!/usr/bin/env python
"""Second opinion for clips the generation gate rejected.

The gate's ASR (Whisper small) "corrects" colloquial Portuguese ("deixa eu"
-> "deixe-me", "pra" -> "para") and writes numbers its own way ("seis e meia"
-> "6,5"), so it rejects many clips in which the teacher said exactly the
text. A rejected clip is accepted when a second, independent ASR (parakeet-v3
through ``fscript``) hears the text with a spoken-form WER below the gate.
The forced alignment that follows is a third, letter-level check.

    python scripts/rescue_clips.py --audio data/audio-v1
"""
import argparse
import sqlite3
import subprocess
from pathlib import Path

from jiwer import wer

from tiny_broca.text import spoken_normalize


def transcribe(path):
    result = subprocess.run(["fscript", str(path), "--model", "parakeet-v3-int8", "--text=plain", "-D", "--raw", "--stdout"],
                            capture_output=True, text=True)
    if result.returncode:  # an empty transcript would reject the clip for good
        raise RuntimeError(f"fscript failed on {path}: {(result.stderr or result.stdout).strip()[:300]}")
    return result.stdout.strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", type=Path, required=True)
    ap.add_argument("--max-wer", type=float, default=0.15)
    args = ap.parse_args()
    conn = sqlite3.connect(args.audio / "state.db")
    columns = [r[1] for r in conn.execute("PRAGMA table_info(clips)")]
    if "rescue" not in columns:
        conn.execute("ALTER TABLE clips ADD COLUMN rescue TEXT")
    rows = conn.execute("SELECT clip_id, text FROM clips WHERE status='bad' AND rescue IS NULL").fetchall()
    rescued = 0
    for i, (cid, text) in enumerate(rows):
        path = args.audio / "audio" / f"{cid}.flac"
        if not path.exists():
            continue
        heard = transcribe(path)
        score = wer(spoken_normalize(text), spoken_normalize(heard)) if heard else 1.0
        ok = score < args.max_wer
        rescued += ok
        conn.execute("UPDATE clips SET rescue=?, status=? WHERE clip_id=?",
                     (f"parakeet {score:.3f}: {heard}", "ok" if ok else "bad", cid))
        conn.commit()
        if i % 25 == 0:
            print(f"{i}/{len(rows)} rescued {rescued}", flush=True)
    print(f"rescued {rescued} of {len(rows)} rejected clips")


if __name__ == "__main__":
    main()
