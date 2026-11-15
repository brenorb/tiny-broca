"""Pilot: verify Qwen3-TTS Portuguese (Brazil) quality and speed on this machine.

1. Design candidate pt-BR voices with Qwen3-TTS-12Hz-1.7B-VoiceDesign.
2. Clone the best candidates with Qwen3-TTS-12Hz-1.7B-Base over pilot sentences.
3. Screen with faster-whisper (small, int8) and report normalised WER + RTF.

Outputs under <out>: references/, audio/, pilot.json (metrics).
"""

import argparse
import json
import math
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from faster_whisper import WhisperModel
from jiwer import wer
from qwen_tts import Qwen3TTSModel

DESIGN_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
BASE_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

# Voice design descriptions. Written in English (the instruction language),
# describing Brazilian Portuguese speakers.
VOICES = {
    "F1": {
        "label": "Clara and warm",
        "prompt": "An adult Brazilian woman in her mid twenties with a clear, "
                  "warm mid-range voice. Brazilian Portuguese (Sao Paulo), "
                  "relaxed and friendly, natural conversational intonation, "
                  "crisp articulation.",
    },
    "M1": {
        "label": "Calm and articulate",
        "prompt": "An adult Brazilian man in his late twenties with a clear, "
                  "calm baritone voice. Brazilian Portuguese (Sao Paulo), "
                  "measured and articulate, natural conversational intonation.",
    },
}

# A few candidate designs per voice so a human can audition; the one with the
# best automated QC is the reference.
CANDIDATE_SEEDS = {
    "F1": [11, 22, 33],
    "M1": [44, 55, 66],
}

REFERENCE_TEXT = ("Oi, que bom ouvir sua voz. Estava pensando em tomar um cafe "
                  "naquele lugar perto da praia, que tal? Ah, e acabei de "
                  "terminar aquele projeto, fiquei bem feliz com o resultado!")

PILOT_SENTENCES = [
    "Olá! Este é o tiny broca, um modelo pequeno de fala em português brasileiro.",
    "A reunião foi adiada para as quinze e trinta de quatorze de março.",
    "O café custa doze reais e cinquenta centavos no balcão.",
    "Meu número de telefone é zero vinte e um, nove nove nove, quatro quatro um um.",
    "Ela disse que o relógio marcava quinze e quarenta e cinco.",
    "Você acredita mesmo que eu vou esquecer o seu aniversário?",
    "O avião pousou no aeroporto de Guarulhos às oito e meia da manhã.",
    "Clique no link para abrir a página de suporte, é só apertar a tecla de enter.",
    "A conta veio quinhentos e noventa e dois reais, com doze por cento de desconto.",
    "Em dois mil e vinte e seis, a cidade completou duzentos anos de fundação.",
    "Trinta e três por cento dos eleitores não votaram neste domingo.",
    "A temperatura hoje fica entre quinze e vinte e oito graus, com pancadas de chuva.",
    "O código do endereço postal é zero zero um zero zero zero, zona sul.",
    "Ela tinha nunca visto o mar antes, e ficou um bom tempo na beira d'água sem dizer uma palavra.",
    "Me passa o saleiro, por favor? O arroz tá sem sal.",
    "O jogo termina daqui a vinte minutos, e a gente ainda tá empatado.",
    "Trabalho daqui a dois mil e vinte e sete.",
    "A fila andou e já estamos a uns três minutos do início do show.",
    "Quero um suco de laranja e dois sanduíches de presunto e queijo, por favor.",
    "O sistema de ar condicionado tá com defeito e o quarto tá um forno.",
]


_NUMBER_WORDS = {"zero", "um", "uma", "dois", "duas", "três", "tres", "quatro", "cinco",
                 "seis", "sete", "oito", "nove", "dez", "onze", "doze", "treze", "quatorze",
                 "quince", "dezenove", "vinte", "trinta", "quarenta", "cinquenta", "sessenta",
                 "setenta", "oitenta", "noventa", "cem", "cento", "mil", "milhão", "milhao",
                 "milhão"}
NUM = re.compile(r"\d+(?:[.,]\d+)*")


def normalize_pt(text):
    """Whisper-style normalisation for Portuguese: casefold, strip marks and
    punctuation, and mask every number. Qwen3-TTS speaks numbers correctly but
    whisper transcribes them as digits ("R$12,50" for "doze reais e
    cinquenta"), so a word-level WER would be meaningless; the duration checks
    cover skipped numbers instead."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.casefold()
    text = re.sub(r"r\s?\$\s?", " <num> ", text)
    text = re.sub(r"%", " <num> ", text)
    text = NUM.sub(" <num> ", text)
    text = re.sub(r"[^\w\s<>=]", " ", text)
    words = [w for w in text.split() if not w.startswith("<") and w not in _NUMBER_WORDS
             and w.rstrip("s") not in _NUMBER_WORDS and "num" not in w]
    return " ".join(words).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--limit", type=int, default=0, help="Only the first N pilot sentences")
    parser.add_argument("--skip-design", action="store_true", help="Reuse existing references")
    parser.add_argument("--threads", type=int, default=12)
    args = parser.parse_args()
    out = args.out
    refs_dir = out / "references"
    audio_dir = out / "audio"
    refs_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    report = {"voices": {}, "clips": []}

    torch.set_num_threads(args.threads)

    references = {}
    if args.skip_design:
        refs_file = refs_dir / "voices.json"
        assert refs_file.exists(), "no references yet; run without --skip-design"
        references = json.loads(refs_file.read_text())
    else:
        print("loading VoiceDesign model ...", flush=True)
        design = Qwen3TTSModel.from_pretrained(DESIGN_MODEL, device_map=args.device,
                                               dtype=torch.bfloat16, attn_implementation="sdpa")
        for voice, spec in VOICES.items():
            for seed in CANDIDATE_SEEDS[voice]:
                tag = f"{voice}_s{seed}"
                path = refs_dir / f"{tag}.wav"
                if path.exists():
                    continue
                torch.manual_seed(seed)
                t0 = time.time()
                wavs, sr = design.generate_voice_design(
                    text=REFERENCE_TEXT, language="Portuguese", instruct=spec["prompt"],
                    max_new_tokens=600, seed=seed)
                wave = np.asarray(wavs[0], dtype=np.float32)
                wave = wave / max(float(np.abs(wave).max()), 1e-9) * 0.95
                sf.write(path, wave, sr)
                references[tag] = {"id": tag, "voice": voice, "label": spec["label"],
                                   "prompt": spec["prompt"], "text": REFERENCE_TEXT,
                                   "seed": seed, "model": DESIGN_MODEL, "sr": sr,
                                   "seconds": len(wave) / sr,
                                   "generation_seconds": round(time.time() - t0, 2)}
                print(f"{tag}: {len(wave)/sr:.1f}s in {time.time()-t0:.1f}s", flush=True)
                del wavs
        design = None
        torch.mps.empty_cache() if args.device == "mps" else None
        (refs_dir / "voices.json").write_text(json.dumps(references, indent=2, ensure_ascii=False))

    # Whisper QC of the references (also used as the ref_text transcription check).
    print("loading whisper small ...", flush=True)
    asr = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=args.threads)

    def transcribe(path):
        import librosa
        wave, rate = sf.read(str(path), dtype="float32")
        if rate != 16000:
            wave = librosa.resample(wave, orig_sr=rate, target_sr=16000)
        segments, info = asr.transcribe(wave, language="pt", beam_size=1, vad_filter=False)
        return " ".join(s.text for s in segments), info.duration

    ref_qc = {}
    for tag, spec in references.items():
        transcript, duration = transcribe(refs_dir / f"{tag}.wav")
        n_ref, n_hyp = normalize_pt(spec["text"]), normalize_pt(transcript)
        w = wer(n_ref, n_hyp)
        ref_qc[tag] = {"wer": round(w, 4), "transcript": transcript, "seconds": duration}
        print(f"ref {tag}: WER {w:.1%} ({duration:.1f}s)", flush=True)

    # Pick the best candidate per base voice.
    best = {}
    for voice in VOICES:
        candidates = [t for t in references if references[t]["voice"] == voice and t in ref_qc]
        if not candidates:
            continue
        best[voice] = min(candidates, key=lambda t: ref_qc[t]["wer"])
    print("best:", best, flush=True)

    print("loading Base model ...", flush=True)
    base = Qwen3TTSModel.from_pretrained(BASE_MODEL, device_map=args.device,
                                         dtype=torch.bfloat16, attn_implementation="sdpa")
    prompts = {v: base.create_voice_clone_prompt(ref_audio=str(refs_dir / f"{best[v]}.wav"),
                                                 ref_text=references[best[v]]["text"],
                                                 x_vector_only_mode=False)
               for v in best}

    sentences = PILOT_SENTENCES[: args.limit] if args.limit else PILOT_SENTENCES
    batch_size = 8
    existing_report = {}
    previous = out / "pilot.json"
    if previous.exists():
        existing_report = {c["id"]: c for c in json.loads(previous.read_text()).get("clips", [])}
    for voice, prompt in prompts.items():
        for b0 in range(0, len(sentences), batch_size):
            batch = list(enumerate(sentences[b0:b0 + batch_size], start=b0))
            if all((audio_dir / f"pilot_{voice}_{i:02d}.flac").exists() for i, _ in batch):
                for i, text in batch:
                    report["clips"].append(existing_report.get(f"pilot_{voice}_{i:02d}"))
                continue
            torch.manual_seed(20261115 + b0)
            t0 = time.time()
            wavs, sr = base.generate_voice_clone(
                text=[text for _, text in batch], language=["Portuguese"] * len(batch),
                voice_clone_prompt=prompt, max_new_tokens=600,
                seed=20261115 + b0)
            gen_seconds = time.time() - t0
            for (i, text), wave in zip(batch, wavs):
                clip_id = f"pilot_{voice}_{i:02d}"
                wave = np.asarray(wave, dtype=np.float32)
                seconds = len(wave) / sr
                peak = float(np.abs(wave).max())
                if peak > 1e-9:
                    wave = wave / peak * 0.95
                path = audio_dir / f"{clip_id}.flac"
                sf.write(path, wave, sr)
                transcript, duration = transcribe(path)
                w = wer(normalize_pt(text), normalize_pt(transcript))
                record = {"id": clip_id, "voice": voice, "text": text, "seconds": round(seconds, 3),
                          "sr": sr, "generation_seconds": round(gen_seconds / len(batch), 2),
                          "rtf": round(gen_seconds / len(batch) / max(seconds, 1e-9), 3),
                          "duration_ok": 0.5 <= seconds <= 35.0,
                          "wer": round(w, 4), "transcript": transcript}
                report["clips"].append(record)
                print(f"{clip_id}: {seconds:.1f}s audio, RTF {record['rtf']}, WER {w:.1%}, "
                      f"duration_ok={record['duration_ok']}", flush=True)

    report["references"] = {t: ref_qc[best[t]] for t in best}
    report["best"] = best
    report["supported_languages"] = getattr(base, "get_supported_languages", lambda: None)()
    (out / "pilot.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    wers = [c["wer"] for c in report["clips"]]
    rtfs = [c["rtf"] for c in report["clips"]]
    print(json.dumps({"n": len(wers), "wer_mean": round(float(np.mean(wers)), 4),
                      "wer_max": max(wers), "rtf_mean": round(float(np.mean(rtfs)), 3),
                      "rtf_max": max(rtfs)}, indent=2))


if __name__ == "__main__":
    main()
