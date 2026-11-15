#!/usr/bin/env python
"""Design the tiny-broca reference voices with Qwen3-TTS VoiceDesign.

VoiceDesign creates a new synthetic voice from a written description (no human
recording is used anywhere in this project). Each candidate is a short
utterance of a fixed reference text; candidates with good automated QC are
kept for a human to audition, and the chosen ones become the cloning prompts
for the corpus generator.

Usage:
    python scripts/design_voices.py --out data/voices [--device mps]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from faster_whisper import WhisperModel
from jiwer import wer
from qwen_tts import Qwen3TTSModel

DESIGN_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

VOICES = {
    "F1": [
        # English description
        "A Brazilian woman in her mid twenties with a warm, friendly voice. "
        "She speaks standard Brazilian Portuguese from São Paulo in a relaxed, "
        "natural conversational tone, like a podcast host.",
        # Portuguese description
        "Uma mulher brasileira na casa dos vinte e cinco anos, com uma voz "
        "agradável e amigável. Fala português brasileiro padrão, de São Paulo, "
        "com tom conversacional natural e relaxado, como uma apresentadora de podcast.",
    ],
    "M1": [
        "A Brazilian man in his late twenties with a calm, clear baritone voice. "
        "He speaks standard Brazilian Portuguese from São Paulo in a measured, "
        "natural conversational tone, like a radio narrator.",
        "Um homem brasileiro na casa dos vinte e poucos anos, com uma voz "
        "grave, calma e clara. Fala português brasileiro padrão, de São Paulo, "
        "com tom conversacional natural e pausado, como um locutor de rádio.",
    ],
}

REFERENCE_TEXTS = (
    "Oi, que bom te ouvir. Estava pensando em tomar um café naquela praça "
    "perto de casa depois do trabalho, e queria te ver. Se der, a gente se "
    "encontra por lá, combina?",
    "Então, acabei de terminar aquele projeto que tava me tirando o sono, e "
    "sabe como é, a gente sempre adia as coisas boas. Achei melhor te avisar "
    "antes, porque o final de semana vai correr, mas quero aproveitar um "
    "pouco com a família também.",
)

NUMBER_WORDS = {"zero", "um", "dois", "três", "quatro", "cinco", "seis", "sete",
                "oito", "nove", "dez", "onze", "doze"}


def normalize(text, unicodedata):
    import re
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.casefold()
    text = re.sub(r"\d+(?:[.,]\d+)*", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(w for w in text.split() if w not in NUMBER_WORDS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/voices"))
    ap.add_argument("--device", default="mps")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--seeds", type=int, nargs="+", default=[101, 102, 103, 111, 112, 113])
    ap.add_argument("--max-tokens", type=int, default=1400)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)

    print("loading VoiceDesign ...", flush=True)
    design = Qwen3TTSModel.from_pretrained(DESIGN_MODEL, device_map=args.device,
                                           dtype=torch.bfloat16, attn_implementation="sdpa")
    print("loading whisper small ...", flush=True)
    asr = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=args.threads)

    def transcribe(path):
        import librosa
        wave, rate = sf.read(str(path), dtype="float32")
        if rate != 16000:
            wave = librosa.resample(wave, orig_sr=rate, target_sr=16000)
        segments, _ = asr.transcribe(wave, language="pt", beam_size=1, vad_filter=False)
        return " ".join(s.text for s in segments)

    import unicodedata
    records = {}
    for voice, prompts in VOICES.items():
        for p, prompt in enumerate(prompts):
            for seed in args.seeds:
                tag = f"{voice}_p{p}_s{seed}"
                path = args.out / f"{tag}.wav"
                if path.exists():
                    continue
                torch.manual_seed(seed)
                t0 = time.time()
                wavs, sr = design.generate_voice_design(
                    text=list(REFERENCE_TEXTS), language=["Portuguese"] * len(REFERENCE_TEXTS),
                    instruct=prompt, max_new_tokens=args.max_tokens, seed=seed)
                for i, (wave, ref_text) in enumerate(zip(wavs, REFERENCE_TEXTS)):
                    full = f"{tag}_r{i}"
                    wpath = args.out / f"{full}.wav"
                    wave = np.asarray(wave, dtype=np.float32)
                    peak = float(np.abs(wave).max())
                    if peak > 1e-9:
                        wave = wave / peak * 0.95
                    sf.write(wpath, wave, sr)
                    transcript = transcribe(wpath)
                    w = wer(normalize(ref_text, unicodedata), normalize(transcript, unicodedata))
                    records[full] = {"id": full, "voice": voice, "prompt_index": p,
                                     "prompt": prompt, "seed": seed, "model": DESIGN_MODEL,
                                     "text": ref_text, "sr": sr, "seconds": round(len(wave) / sr, 2),
                                     "generation_seconds": round((time.time() - t0) / len(wavs), 2),
                                     "wer": round(w, 4), "transcript": transcript}
                    print(f"{full}: {len(wave)/sr:.1f}s WER {w:.1%}", flush=True)
    (args.out / "voices.json").write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print(json.dumps({k: round(sum(v["wer"] for v in r.values() if v["voice"] == k)
                                 / max(len([v for v in r.values() if v["voice"] == k]), 1), 3)
                      for k in VOICES}))


if __name__ == "__main__":
    main()
