# Qwen3-TTS in Portuguese (research notes, 15 Nov 2026)

Teacher TTS for tiny-broca: `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` (voice design)
and `Qwen/Qwen3-TTS-12Hz-1.7B-Base` (cloning). Both Apache-2.0, Portuguese is one
of the ten supported languages (`codec_language_id.portuguese = 2071`; there is no
Brazil-specific tag, the dialect is steered through the design instruction).

## What was measured (scripts/pilot_qwen_pt.py, data/pilot/)

* Six VoiceDesign candidates (three per voice) were designed from English
  instructions that say "Brazilian Portuguese (Sao Paulo)".
  Whisper small (int8) WER on the reference text: 0%, 6.2%, 18.8% (F1) and
  0%, 0%, 9.4% (M1). Best kept: F1_s11 and M1_s44.
* 20 pilot sentences cloned in both voices, batch of 8.
  - Plain prose: 0-9% WER.
  - Numbers: the model speaks them correctly in Brazilian Portuguese; whisper
    writes them as digits ("R$12,50" for "doze reais e cinquenta"), which the
    QC normalizer masks.
* RTF on the M4 Pro (18 cores, MPS, bf16, sdpa): ~1.5 single, **~0.30 in
  batches of 8** -> ~19 hours of corpus audio in roughly 6 hours of wall time.

## Accent check (whisper medium vs CV19 human clips)

| bucket | n | mean WER | median | max |
|---|---:|---:|---:|---:|
| Qwen references | 6 | 3.7% | 3.1% | 9.4% |
| human pt-BR (CV19 variant "Portuguese (Brasil)") | 15 | 8.1% | 0% | 50% |
| human pt-PT (CV19 variant "Portuguese (Portugal)") | 15 | 11.6% | 0% | 83% |

No intelligibility problem and consistent with Brazilian Portuguese; a native
listener audition of the references is the final gate.

## qwen-tts 0.1.1 API notes

* `generate_voice_clone(text=[...], language=["Portuguese"],
  voice_clone_prompt=<list of VoiceClonePromptItem>)`; a one-item list is
  broadcast over the batch. Passing the bare item or a list-of-lists fails.
* Requires SoX on PATH (audio resampling inside the wrapper).
* faster-whisper in this venv hits a pyav incompatibility for file paths;
  pass 16 kHz float32 numpy arrays instead (soundfile + librosa resample).
