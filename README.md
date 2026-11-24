# tiny-broca

A 4.5M-parameter two-voice **Brazilian Portuguese** text-to-speech model,
trained from scratch. One female (F1) and one male (M1) voice. 24 kHz output.
Runs on a CPU.

Everything is open: the code (this repository), the weights and the dataset
(Hugging Face), and the training recipe.

| | |
|---|---|
| Parameters | 4,533,835, waveform generator included |
| Voices | F1 (female), M1 (male), Brazilian Portuguese |
| Output | 24 kHz mono |
| Input | Text; tokenised as letters, no dictionary |
| Pretrained weights in the model | None |
| License | Apache-2.0 (code and weights) |

## Design

The architecture follows the English two-voice model this project replicates:
a FastSpeech-2-style sequence model (text encoder, explicit duration predictor,
two frame-level transformers for prosody) with a source-filter waveform
generator instead of a neural vocoder.

The front end is where Portuguese diverges. No permissive Portuguese
pronouncing dictionary exists (eSpeak's Portuguese voice falls back to English
phonemes for most words), so the model reads **letters**, the way a human
reader does: 58 letter/punctuation symbols plus a stress feature per letter
from a Portuguese syllabifier (explicit accents, oxytone endings, hiatus).
Numbers, dates, times, money, percentages, symbols and document identifiers
are written out as a Brazilian reader says them (`R$3,00` → "três reais",
`4 > 3` → "quatro maior que três", `14/03` → "quatorze de março"), acronyms
are spelled with Portuguese letter names, and English loanwords are respelled
the Brazilian way (`online` → "onlaine").

The training audio is synthetic: Qwen3-TTS 1.7B generated 19,504 clips
(about 21 hours) across three batches for v0; 16,434 clips (about 17 hours)
survived quality control, every sentence read by both voices. For v1 it read
14,600 more clips from their spoken forms (14,047 kept), and v1 trains on
26,130 clips (28 hours). The two voices are clones
of two Common Voice speakers; the short reference recordings are the only real
human audio in the pipeline and they are not part of the model.

See [docs/model.md](docs/model.md) for the full architecture and data
description, and [docs/annotations.md](docs/annotations.md) for the letter
durations (forced alignment with a Portuguese wav2vec 2.0 model, used only to
label the training data).

## Versions

v1.1 is the current release: the v1 model with an updated front end (y read
as i, and 131 new respellings for crypto, AI, tech and English names; see
[docs/v1.md](docs/v1.md#v11)).

v1 fixes what v0 read wrongly (English loanwords, symbols, money, dates,
acronyms) and makes unvoiced sounds from noise; on the probe suite, missed
keywords fell from 242 to 18 of 574. See [docs/v1.md](docs/v1.md). The
releases are tagged `v1.1.1`, `v1.1`, `v1` and `v0`.

## Usage

```bash
pip install git+https://github.com/brenorb/tiny-broca
hf download brenorb/tiny-broca --local-dir tiny-broca

python -m tiny_broca.infer \
  --checkpoint tiny-broca/tiny-broca.pt \
  --voice F1 \
  --text "Olá! Eu sou a tiny broca." \
  --output hello.wav
```

The ONNX export runs in browsers (onnxruntime-web); the page in `web/` runs
the model and the front end entirely client-side.

## The model and the dataset

- Weights: https://huggingface.co/brenorb/tiny-broca
- Dataset: https://huggingface.co/datasets/brenorb/tiny-broca
- Demo: <https://tiny-broca-demo.pages.dev> (runs entirely in the browser)

## Repository layout

| Path | Contents |
|---|---|
| `src/tiny_broca/` | Front end, model, data, alignment, training, evaluation, export |
| `data/corpus/` | The text corpus (Parquet, five batches) |
| `data/voices/`, `data/pilot/` | Voice references and pilot recordings |
| `scripts/` | Corpus builder, voice design, audio generation |
| `docs/` | Model and data description, annotation format, HF cards |
| `web/` | The in-browser demo (Cloudflare Pages) |

## License

Code and weights: Apache-2.0. Dataset: CC BY 4.0. The Common Voice reference
recordings are CC0; the conversational slot pools and templates were written
for this project. Neither eSpeak NG nor any pronouncing dictionary is used
anywhere.
