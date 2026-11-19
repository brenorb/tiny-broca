#!/usr/bin/env python3
"""Slow down the teacher's number regions in the money batch (batch3).

Diagnosis: the Qwen3 teacher rushes long money values. In its batch3 audio
the number regions run at 0.050 s/letter — 38% faster than human Common
Voice speakers (0.081 s/letter) and faster than the same sentences' frame
words (0.070). The model distils the teacher faithfully, so it inherits the
rush; fine-tuning on more of the same content cannot fix it.

This script edits the batch3 teacher audio in the time domain: each number
region (a run of number words, with a small margin) is time-stretched to the
human pace; the rest of the utterance is untouched. The stretched clips are
written next to the originals with a suffix, ready for a fresh
prepare/annotate round whose targets carry a natural number pace.

    python scripts/slow_number_regions.py [--audio data/audio/audio]
        [--annotations data/annotations-money] [--suffix -slow] [--factor 1.62]

Number regions are located with the existing forced-alignment annotation:
consecutive letters that belong to number words, expanded to their aligned
frames, merged where runs are close, and padded.
"""
import argparse
import re
from pathlib import Path

import librosa
import soundfile as sf
import sys

sys.path.insert(0, "src")
from tiny_broca.text import normalize_text

FPS = 24000 / 256  # frames per second at 24 kHz / hop 256
NUM = ({"mil", "milhão", "milhões", "cento", "cem", "meio", "meia", "milhar"}
       | {str(x) for x in range(10)}
       | {"dez", "vinte", "trinta", "quarenta", "cinquenta", "sessenta",
          "setenta", "oitenta", "noventa", "reais", "centavos", "dólares",
          "pessoas", "quilos", "metros", "minutos", "horas", "dias",
          "semanas", "meses", "anos"})


def number_regions(text: str, tokens, durations, margin_frames=6, merge_gap_frames=28):
    """Return (start_frame, end_frame) regions covering the number words."""
    if len(durations) != len(tokens) or len(tokens) - 2 != len(text):
        return []
    words = {}
    for m in re.finditer(r"[a-zà-ü]+", text):
        w = m.group(0)
        for i in range(m.start(), m.end()):
            words[i] = w
    frames = []
    for j in range(1, len(durations) - 1):
        c = text[j - 1]
        if c in "abcdefghijklmnopqrstuvwxyzà-ü" and words.get(j - 1) in NUM and durations[j] > 0:
            # token j spans frames cumsum(j-1) .. cumsum(j-1)+durations[j]
            frames.append((j, durations[j]))
    if not frames:
        return []
    total = sum(durations)
    regions, start, prev_end = [], None, -10 ** 9
    offset = 0
    for j, d in frames:
        s = sum(durations[1:j]) + offset
        e = s + d
        if start is None:
            start, prev_end = s, e
        elif s - prev_end <= merge_gap_frames:
            prev_end = e
        else:
            regions.append((start, prev_end))
            start, prev_end = s, e
    if start is not None:
        regions.append((start, prev_end))
    # merge into seconds and pad
    out = []
    for s, e in regions:
        out.append((max(s - margin_frames, 0) / FPS, min((e + margin_frames) / FPS, total / FPS)))
    return out


def slow_clip(wave, sr, regions, factor, crossfade=0.01):
    """Time-stretch each region; blend the edges with a short crossfade."""
    import numpy as np
    cf = int(crossfade * sr)
    pieces, pos = [], 0
    for start, end in regions:
        s, e = int(start * sr), min(int(end * sr), len(wave))
        if e - s < cf:
            continue  # region too short to stretch safely; leave at teacher pace
        stretched = librosa.effects.time_stretch(wave[s:e], rate=1.0 / factor)
        pieces.append(wave[pos:s])
        if s - pos >= cf:
            pieces[-1][-cf:] *= np.linspace(1.0, 0.0, cf, dtype=wave.dtype)
            stretched[:cf] *= np.linspace(0.0, 1.0, cf, dtype=wave.dtype)
        pieces.append(stretched)
        pos = e
    pieces.append(wave[pos:])
    return np.concatenate(pieces)


def main():
    import json
    import numpy as np
    import sqlite3
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", type=Path, default=Path("data/audio/audio"))
    ap.add_argument("--output", type=Path, default=Path("data/audio/audio-slow"))
    ap.add_argument("--annotations", type=Path, default=Path("data/annotations-money/train.jsonl"))
    ap.add_argument("--factor", type=float, default=1.62, help="stretch factor for number regions "
                                                               "(0.081 human / 0.050 teacher = 1.62)")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    annotations = {json.loads(l)["id"]: json.loads(l)
                   for l in open(args.annotations) if json.loads(l).get("id", "").startswith("b3_")}
    print(f"money clips in annotations: {len(annotations)}")
    conn = sqlite3.connect("data/audio/state.db")
    done = stretched = 0
    for clip_id, in conn.execute("SELECT clip_id FROM clips WHERE batch=3 AND status='ok'"):
        if clip_id not in annotations:
            continue
        src = args.audio / (clip_id + ".flac")
        if not src.exists():
            src = args.audio / (clip_id + ".wav")
        if not src.exists():
            continue
        ann = annotations[clip_id]
        wave, sr = sf.read(src, dtype="float32")
        if sr != 24000:
            raise SystemExit(f"{clip_id}: unexpected rate {sr}")
        text = normalize_text(ann["text"])
        regions = number_regions(text, ann["tokens"], ann["durations"])
        out = slow_clip(wave, sr, regions, args.factor) if regions else wave
        sf.write(str(args.output / (clip_id + "-slow.flac")), out, sr)
        done += 1
        stretched += bool(regions)
        if done % 50 == 0:
            print(f"{done} processed ({stretched} stretched)", flush=True)
    print(f"done: {done} clips, {stretched} stretched -> {args.output}")


if __name__ == "__main__":
    main()
