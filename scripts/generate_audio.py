#!/usr/bin/env python
"""Generate the tiny-broca training audio with Qwen3-TTS (Brazilian Portuguese).

Every corpus sentence is read by both voices (F1, M1), like the original
tiny-voca dataset. Generation is resumable (sqlite state in the output dir),
peak-normalised to 0.95, and quality-checked per clip with faster-whisper
(int8 on CPU): WER with a number-masking normaliser plus a duration range.

Write a file named STOP inside the output directory to stop gracefully after
the current batch; rerun to continue.

Usage:
    python scripts/generate_audio.py \
        --corpus data/corpus --refs data/pilot/references \
        --out data/audio --device mps
"""

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from faster_whisper import WhisperModel
from jiwer import wer
from qwen_tts import Qwen3TTSModel

BASE_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
VOICE_REFS = {"F1": "F1", "M1": "M1"}  # selected Common Voice clones (data/refs)

from tiny_broca.text import wer_normalize


def load_corpus(corpus_dir: Path):
    import pyarrow.parquet as pq

    sentences = []
    for batch in (1, 2):
        for path in sorted(corpus_dir.glob(f"batch{batch}/*.parquet")):
            table = pq.read_table(path)
            for i, (text, source) in enumerate(zip(table.column("text").to_pylist(),
                                                   table.column("source").to_pylist())):
                sentences.append({"batch": batch, "split": path.stem,
                                  "index": i, "text": text, "source": source})
    return sentences


def open_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("""CREATE TABLE IF NOT EXISTS clips (
        clip_id TEXT PRIMARY KEY, voice TEXT, batch INTEGER, split TEXT,
        text TEXT, seconds REAL, sr INTEGER, gen_seconds REAL, rtf REAL,
        wer REAL, transcript TEXT, status TEXT)""")
    conn.execute("CREATE INDEX IF NOT EXISTS clips_status ON clips(status)")
    return conn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    ap.add_argument("--refs", type=Path, default=Path("data/refs"))
    ap.add_argument("--out", type=Path, default=Path("data/audio"))
    ap.add_argument("--device", default="mps")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261117)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)

    refs = json.loads((args.refs / "voices.json").read_text())
    sentences = load_corpus(args.corpus)
    print(f"corpus: {len(sentences)} sentences", flush=True)

    print("loading whisper small ...", flush=True)
    asr = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=args.threads)

    def transcribe(path):
        import librosa
        wave, rate = sf.read(str(path), dtype="float32")
        if rate != 16000:
            wave = librosa.resample(wave, orig_sr=rate, target_sr=16000)
        segments, _ = asr.transcribe(wave, language="pt", beam_size=1, vad_filter=False)
        return " ".join(s.text for s in segments)

    print("loading Qwen3-TTS Base ...", flush=True)
    base = Qwen3TTSModel.from_pretrained(BASE_MODEL, device_map=args.device,
                                         dtype=torch.bfloat16, attn_implementation="sdpa")
    prompts = {}
    for voice, ref_tag in VOICE_REFS.items():
        prompts[voice] = base.create_voice_clone_prompt(
            ref_audio=str(args.refs / f"{ref_tag}.wav"),
            ref_text=refs[ref_tag]["text"], x_vector_only_mode=False)
        print(f"voice {voice} <- {ref_tag}.wav", flush=True)

    conn = open_db(args.out / "state.db")
    done = {r[0] for r in conn.execute(
        "SELECT clip_id FROM clips WHERE status='ok'")}
    jobs = []
    for voice in VOICE_REFS:
        for s in sentences:
            cid = f"b{s['batch']}_{voice}_{s['index']:05d}"
            if cid not in done:
                jobs.append((voice, cid, s))
    print(f"to generate: {len(jobs)} clips "
          f"({sum(1 for j in jobs if j[0] == 'F1')} F1, "
          f"{sum(1 for j in jobs if j[0] == 'M1')} M1)", flush=True)

    audio_dir = args.out / "audio"
    audio_dir.mkdir(exist_ok=True)
    t_start = time.time()
    n_ok = n_bad = 0
    last_report = 0
    wers, rtfs = [], []

    # Group remaining jobs by voice into batches.
    per_voice = {v: [j for j in jobs if j[0] == v] for v in VOICE_REFS}
    for voice, vjobs in per_voice.items():
        prompt = prompts[voice]
        for b0 in range(0, len(vjobs), args.batch):
            if (args.out / "STOP").exists():
                print("STOP file present, exiting", flush=True)
                return
            chunk = vjobs[b0:b0 + args.batch]
            t0 = time.time()
            wavs, sr = base.generate_voice_clone(
                text=[s["text"] for _, _, s in chunk],
                language=["Portuguese"] * len(chunk),
                voice_clone_prompt=prompt, max_new_tokens=600,
                seed=args.seed + b0)
            gen = time.time() - t0
            for (v, cid, s), wave in zip(chunk, wavs):
                wave = np.asarray(wave, dtype=np.float32)
                seconds = len(wave) / sr
                peak = float(np.abs(wave).max())
                if peak > 1e-9:
                    wave = wave / peak * 0.95
                path = audio_dir / f"{cid}.flac"
                sf.write(path, wave, sr)
                transcript = transcribe(path)
                w = wer(wer_normalize(s["text"]), wer_normalize(transcript)) if transcript else 1.0
                ok = (0.5 <= seconds <= 35.0) and (w < 0.15) and transcript
                conn.execute("INSERT OR REPLACE INTO clips VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                             (cid, voice, s["batch"], s["split"], s["text"],
                              round(seconds, 3), sr, round(gen / len(chunk), 2),
                              round(gen / len(chunk) / max(seconds, 1e-9), 3),
                              round(w, 4), transcript, "ok" if ok else "bad"))
                conn.commit()
                if ok:
                    n_ok += 1
                    wers.append(w)
                    rtfs.append(gen / len(chunk) / max(seconds, 1e-9))
                else:
                    n_bad += 1
                    print(f"  BAD {cid}: {seconds:.1f}s WER {w:.2f}", flush=True)
            done_n = n_ok + n_bad
            if done_n - last_report >= 50 or b0 + args.batch >= len(vjobs):
                last_report = done_n
                (args.out / "progress.json").write_text(json.dumps({
                    "done": done_n, "ok": n_ok, "bad": n_bad,
                    "wer_mean": round(float(np.mean(wers)), 4) if wers else None,
                    "rtf_mean": round(float(np.mean(rtfs)), 3) if rtfs else None,
                    "elapsed_min": round((time.time() - t_start) / 60, 1),
                    "eta_h": round((time.time() - t_start) / max(done_n, 1)
                                   * max(len(jobs) - done_n, 0) / 3600, 2)}))
                print(f"voice {voice} batch {b0 // args.batch + 1}: "
                      f"total {done_n} (bad {n_bad}), "
                      f"WER {np.mean(wers) if wers else 0:.1%}, "
                      f"RTF {np.mean(rtfs) if rtfs else 0:.2f}, "
                      f"ETA {json.loads((args.out / 'progress.json').read_text())['eta_h']}h",
                      flush=True)

    # Clips that failed the gate get one regeneration with a different seed;
    # clips that fail twice are left out and listed in review.jsonl.
    jobs_by_cid = {j[1]: j for j in jobs}
    bad = [r for r in conn.execute('SELECT clip_id, voice FROM clips WHERE status="bad"')
           if r[0] in jobs_by_cid]
    for voice in VOICE_REFS:
        vbad = [(cid, jobs_by_cid[cid][2]) for cid, v in bad if v == voice]
        for b0 in range(0, len(vbad), args.batch):
            if (args.out / "STOP").exists():
                print("STOP file present, exiting", flush=True)
                return
            chunk = vbad[b0:b0 + args.batch]
            t0 = time.time()
            wavs, sr = base.generate_voice_clone(
                text=[s["text"] for _, s in chunk],
                language=["Portuguese"] * len(chunk),
                voice_clone_prompt=prompts[voice], max_new_tokens=600,
                seed=args.seed + 7919 + b0)
            gen = time.time() - t0
            for (cid, s), wave in zip(chunk, wavs):
                wave = np.asarray(wave, dtype=np.float32)
                seconds = len(wave) / sr
                peak = float(np.abs(wave).max())
                if peak > 1e-9:
                    wave = wave / peak * 0.95
                path = audio_dir / f"{cid}.flac"
                sf.write(path, wave, sr)
                transcript = transcribe(path)
                w = wer(wer_normalize(s["text"]), wer_normalize(transcript)) if transcript else 1.0
                ok = (0.5 <= seconds <= 35.0) and (w < 0.15) and transcript
                conn.execute("INSERT OR REPLACE INTO clips VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                             (cid, voice, s["batch"], s["split"], s["text"],
                              round(seconds, 3), sr, round(gen / len(chunk), 2),
                              round(gen / len(chunk) / max(seconds, 1e-9), 3),
                              round(w, 4), transcript, "ok" if ok else "bad"))
                conn.commit()
                print(f"  RETRY {cid}: {seconds:.1f}s WER {w:.2f} -> {'ok' if ok else 'bad'}", flush=True)
    with (args.out / "review.jsonl").open("w") as fh:
        for row in conn.execute(
                'SELECT clip_id, voice, text, seconds, wer, transcript FROM clips WHERE status="bad"'):
            fh.write(json.dumps(dict(zip(
                ("id", "voice", "text", "seconds", "wer", "transcript"), row)), ensure_ascii=False) + "\n")
    hours = sum(r[0] for r in conn.execute(
        'SELECT seconds FROM clips WHERE status="ok"')) / 3600
    conn.close()
    print(json.dumps({"ok": n_ok, "bad": n_bad,
                      "wer_mean": round(float(np.mean(wers)), 4) if wers else None,
                      "rtf_mean": round(float(np.mean(rtfs)), 3) if rtfs else None,
                      "hours_audio": round(hours, 2)}))


if __name__ == "__main__":
    main()
