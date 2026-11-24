---
license: apache-2.0
language:
- pt
pipeline_tag: text-to-speech
tags:
- text-to-speech
- onnx
- portuguese
datasets:
- brenorb/tiny-broca
---
# tiny-broca

tiny-broca is a small **Brazilian Portuguese** text-to-speech model with two
voices, F1 (female) and M1 (male). The complete model, including the waveform
generator, has 4.5M parameters and was trained from scratch on synthetic
speech.

<audio controls src="https://huggingface.co/brenorb/tiny-broca/resolve/main/samples/F1-1.wav"></audio>

- Code: https://github.com/brenorb/tiny-broca
- Dataset: https://huggingface.co/datasets/brenorb/tiny-broca

| | |
|---|---|
| Parameters | 4,533,835 |
| Voices | F1, M1 (Brazilian Portuguese) |
| Output | 24 kHz mono |
| Input | Text, tokenised as letters (58 symbols), no dictionary |
| Pretrained weights in the model | None |
| License | Apache-2.0 |

## v1.1

v1.1 changes only the front end. The weights and the ONNX model are v1's.

- A y reads as i, as in Portuguese spelling: `Paraty` → "parati", `Yuri` →
  "iuri", `Stefany` → "stefani". English words that say it otherwise are in
  the respelling table (`byte` → "báite", `Tyson` → "táisson"). A lone y is the
  letter, "ípsilon" (`eixo y`, `f(x, y)`), except the Spanish conjunction
  between names (`Ortega y Gasset`).
- 131 new respellings for the crypto, AI and tech words Brazilians don't
  translate, and for English names: `exchange` → "equistchêindje",
  `Lightning Network` → "láitnin nétuórque", `smart contract` → "ismarte
  cóntrecte", `DeFi` → "difái", `HODL` → "rôdou", `PyTorch` → "páitórtche",
  `Hugging Face` → "râguin fêice", `Daisy` → "dêizi". The table now has 1,078
  loanwords and 127 acronyms.

## v1

v1 fixes what v0 read wrongly. The front end now writes out what a Brazilian
reader says: `online` → "onlaine", `4 > 3` → "quatro maior que três",
`R$3,00` → "três reais", `14/03` → "quatorze de março", along with acronyms,
units, ordinals, roman numerals, dates and 948 English loanwords. The training
audio was re-read from these spoken forms, and unvoiced sounds are now made
from noise. On the probe suite, missed keywords fell from 242 to 18 of 574.
The v0 release is tagged `v0` in this repository. Details:
[docs/v1.md](https://github.com/brenorb/tiny-broca/blob/main/docs/v1.md).

## Files

| File | Contents |
|---|---|
| `tiny-broca.pt` | PyTorch weights, float32 |
| `onnx/model.onnx` | ONNX export, float32, opset 17 |
| `config.json` | Released settings: the model block (sample rate, windows, voices, vocabulary size) plus the phonemiser identity, letter vocabulary, and ONNX inputs/outputs that the web demo reads |
| `text.py` | The front end module (version 2), verbatim (the web demo runs it in the browser) |
| `samples/` | The audio on this page |

## Samples

| Text | F1 | M1 |
|---|---|---|
| A reunião de amanhã foi remarcada para as quinze horas e quinze. | <audio controls src="https://huggingface.co/brenorb/tiny-broca/resolve/main/samples/F1-0.wav"></audio> | <audio controls src="https://huggingface.co/brenorb/tiny-broca/resolve/main/samples/M1-0.wav"></audio> |
| Ligue para a sua avó antes que comece o fim de semana, por favor. | <audio controls src="https://huggingface.co/brenorb/tiny-broca/resolve/main/samples/F1-1.wav"></audio> | <audio controls src="https://huggingface.co/brenorb/tiny-broca/resolve/main/samples/M1-1.wav"></audio> |
| Quantas pessoas vão à conferência em setembro? | <audio controls src="https://huggingface.co/brenorb/tiny-broca/resolve/main/samples/F1-2.wav"></audio> | <audio controls src="https://huggingface.co/brenorb/tiny-broca/resolve/main/samples/M1-2.wav"></audio> |
| Não acredito que já é meia-noite e meia; devemos ir embora agora! | <audio controls src="https://huggingface.co/brenorb/tiny-broca/resolve/main/samples/F1-3.wav"></audio> | <audio controls src="https://huggingface.co/brenorb/tiny-broca/resolve/main/samples/M1-3.wav"></audio> |
| A tiny broca é um pequeno modelo de texto para fala treinado do zero. | <audio controls src="https://huggingface.co/brenorb/tiny-broca/resolve/main/samples/F1-4.wav"></audio> | <audio controls src="https://huggingface.co/brenorb/tiny-broca/resolve/main/samples/M1-4.wav"></audio> |
| Depois da tempestade, o ar da montanha parecia fresco, limpo e surpreendentemente quente. | <audio controls src="https://huggingface.co/brenorb/tiny-broca/resolve/main/samples/F1-5.wav"></audio> | <audio controls src="https://huggingface.co/brenorb/tiny-broca/resolve/main/samples/M1-5.wav"></audio> |
| Você pode me enviar o relatório até sexta-feira à tarde, por favor? | <audio controls src="https://huggingface.co/brenorb/tiny-broca/resolve/main/samples/F1-6.wav"></audio> | <audio controls src="https://huggingface.co/brenorb/tiny-broca/resolve/main/samples/M1-6.wav"></audio> |
| A conta do aluguel fechou em mil duzentos e quarenta e cinco reais e dez centavos. | <audio controls src="https://huggingface.co/brenorb/tiny-broca/resolve/main/samples/F1-7.wav"></audio> | <audio controls src="https://huggingface.co/brenorb/tiny-broca/resolve/main/samples/M1-7.wav"></audio> |

## Usage

### PyTorch

```bash
pip install git+https://github.com/brenorb/tiny-broca
hf download brenorb/tiny-broca --local-dir tiny-broca

python -m tiny_broca.infer \
  --checkpoint tiny-broca/tiny-broca.pt \
  --voice F1 \
  --text "Olá! Eu sou a tiny broca." \
  --output ola.wav
```

| Option | Default | Meaning |
|---|---|---|
| `--voice` | `F1` | `F1` or `M1` |
| `--speed` | 1.0 | Speaking rate |
| `--pitch-shift` | 0 | Added to log F0. 0.058 is about one semitone. |
| `--pitch-range` | 1.0 | Scales pitch movement around the utterance mean. |

### ONNX

The graph takes letter ids and stress values and returns the waveform. Text is
converted to ids outside the graph by the front end (`text.py`); the ids are
the positions in its letter inventory.

```python
import numpy as np
import onnxruntime as ort
import soundfile as sf
import sys
sys.path.insert(0, "tiny-broca")
from text import phonemize, SYMBOL_TO_ID

letters = phonemize("Olá! Eu sou a tiny broca.")
one = lambda value: np.array([value], dtype=np.float32)

session = ort.InferenceSession("tiny-broca/onnx/model.onnx")
audio, durations = session.run(None, {
    "tokens": np.array([letters.ids], dtype=np.int64),
    "stress": np.array([letters.stress], dtype=np.int64),
    "voice": np.array([0], dtype=np.int64),   # 0 = F1, 1 = M1
    "speed": one(1.0),
    "pitch_shift": one(0.0),
    "pitch_range": one(1.0),
})
sf.write("ola.wav", audio[0], 24000)
```

| Input | Type | Shape | Meaning |
|---|---|---|---|
| `tokens` | int64 | [1, N] | Letter ids, including the start and end tokens |
| `stress` | int64 | [1, N] | 0 unstressed, 1 stressed syllable |
| `voice` | int64 | [1] | 0 = F1, 1 = M1 |
| `speed` | float32 | [1] | 1.0 as predicted; larger is faster |
| `pitch_shift` | float32 | [1] | Added to log F0 |
| `pitch_range` | float32 | [1] | Scales pitch movement |

| Output | Type | Shape | Meaning |
|---|---|---|---|
| `audio` | float32 | [1, samples] | Waveform at 24 kHz |
| `durations` | int64 | [1, N] | Length of each token in frames of 256 samples |

The model reads one utterance per call, up to 512 tokens and 32 seconds. Split
longer text into sentences. The same file runs in onnxruntime-web
(WebAssembly); the page at the demo URL speaks in the browser.

## Model

Text is tokenised as letters (the 26 letters plus the accented graphemes that
are distinct letters in Portuguese), with a stress value per letter. A
transformer encoder (4 blocks) reads the letters, a duration predictor gives
each one a length in frames, and two frame-level transformers (2 and 5 blocks)
predict pitch, voicing and energy and then the parameters of the waveform
generator. All blocks use 192 channels, 4 heads and rotary position
embeddings.

The waveform is produced by a source-filter synthesizer instead of a neural
vocoder. A band-limited pulse train at the predicted pitch and a noise signal
are each shaped by a time-varying filter, whose cepstrum the network predicts
once per frame (256 samples, 10.7 ms). This part has 25,862 parameters.

| Component | Parameters |
|---|---:|
| Embeddings (letter, stress, voice) | 13,248 |
| Text encoder | 1,588,864 |
| Duration and pause predictor | 123,906 |
| Frame transformer A, prosody head | 795,971 |
| Frame transformer B | 1,985,984 |
| Source-filter waveform head | 25,862 |
| Total | 4,533,835 |

## Training

| | |
|---|---|
| Data | [brenorb/tiny-broca](https://huggingface.co/datasets/brenorb/tiny-broca), v1 training set: 12,026 sentences, 26,130 clips, 28.2 hours, two voices |
| Steps | 148,000: 118,000 for v0, then 30,000 on the v1 data |
| Losses | Multi-resolution spectral losses; adversarial loss with multi-period and multi-resolution discriminators after step 8,000 |
| Hardware | Apple M4 Pro, 18-core, MPS |

The training audio is synthetic: every clip was generated with Qwen3-TTS 1.7B
Base (Apache-2.0) in one of the two voices. The voices are clones of two
Common Voice speakers (CC0): short clean recordings of real people, used only
as voice prompts for the generator. No recording of a real speaker is part of
the model. Letter durations come from forced alignment with
`jonatasgrosman/wav2vec2-xls-r-1b-portuguese`, which is used only to label
the training data. No pretrained weights are part of the model.

The first 104,000 steps use `batch1` and `batch2`. The released model then
continues for 14,000 more steps (to step 118,000) on the same data plus
`batch3`, money phrases: the generator tends to rush long number strings, so
the later rounds give money values extra weight.

v1 continues from v0's full training state (step 118,000) for 30,000 steps on
the v1 training set: `batch1` and `batch2` without the 2,821 v0 sentences that
had been read from digits and symbols, plus `batch4` (new coverage) and
`batch5` (those v0 sentences re-read from their spoken form). From v1 on, the
harmonic branch of the waveform head is excited with noise in unvoiced frames.

## Evaluation

The synthesized speech is transcribed automatically and compared with the
words the text should produce. This measures whether the words come out, not
how natural they sound.

**Probe suite.** 287 sentences in 29 categories (numbers, money, dates,
symbols, acronyms, loanwords, stress, prosody...), read by both voices: 574
clips, transcribed by Whisper medium and parakeet. A keyword counts as missed
when neither of them hears it.

| | Keywords missed | Word error rate |
|---|---:|---:|
| v0 | 242 of 574 | 20.0% |
| v0 weights with the v1 front end | 43 of 574 | 6.3% |
| v1 | 18 of 574 | 2.5% |

Of v1's 18 misses, 8 are roman numerals the transcribers write as numerals
("Dom Pedro II" for "segundo"), and 2 are "CEO", whose respelling was fixed
after the run. The rest: "Copa XX" read as letters, "Paraty" (fixed in v1.1) and "Itu",
"story" and "crush".

**Held-out sentences** (Whisper small, numbers included). On 16 sentences
outside every split, v0 had 8.1% word error and v1 has 1.5%. On 50 test-split
clips v1 has 2.1%; the teacher's own recordings of the same sentences have
2.8%.

## Limitations

- Brazilian Portuguese only, two fixed voices, no voice cloning.
- The voice is clear but can sound slightly synthetic.
- The same text is always read the same way. There is no sampling of prosody.
- Very long number strings (a dozen or more words) are read a little faster
  than the training data; the duration predictor compresses them slightly.
- Words outside the loanword table and foreign names are often
  mispronounced. English-derived names ending in -y are stressed on the last
  syllable ("Lucy" is read "lu-CÍ").
- Ambiguous text gets one reading. A number from 1900 to 2099 is read as a
  year only where something marks a date, and an English word that is also
  Portuguese ("set", "date") is respelled only inside a phrase that settles it.
- No streaming.
- The training data is itself synthetic, so the model cannot be more natural
  than the model that generated it.

## License

The weights and code are released under Apache-2.0. No pronouncing dictionary
is used anywhere: the front end is self-contained. The Common Voice reference
recordings are CC0.

## Citation

```bibtex
@software{brenorb2026tinybroca,
  author = {brenorb},
  title  = {tiny-broca: a 4.5M-parameter two-voice Brazilian Portuguese text-to-speech model},
  year   = {2026},
  url    = {https://huggingface.co/brenorb/tiny-broca}
}
```

## Acknowledgements

The project replicates the open English two-voice model by Muhammed Nazeem
(tiny-voca, Apache-2.0): same architecture, training recipe and release
methodology, adapted to Brazilian Portuguese. The design builds on FastSpeech
2 (explicit duration and prosody prediction), DDSP and the neural homomorphic
vocoder (source-filter synthesis with predicted filters), and HiFi-GAN and
UnivNet (discriminators). No code or weights from these projects are used.
