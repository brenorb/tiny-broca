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
(explicit accents win; else the penultimate syllable, or the last after a
word-final L, N, R or X). Numbers, dates, times, money, percentages and
document identifiers are normalised to words with Brazilian conventions
(`1.234,56` → "mil duzentos e trinta e quatro vírgula cinquenta e seis").

The training audio is synthetic: Qwen3-TTS 1.7B generated 19,504 clips
(about 21 hours) across three batches; 16,434 clips (about 17 hours) survived
quality control, every sentence read by both voices. The two voices are clones
of two Common Voice speakers; the short reference recordings are the only real
human audio in the pipeline and they are not part of the model.

See [docs/model.md](docs/model.md) for the full architecture and data
description, and [docs/annotations.md](docs/annotations.md) for the letter
durations (forced alignment with a Portuguese wav2vec 2.0 model, used only to
label the training data).

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
| `data/corpus/` | The 9,384-sentence text corpus (Parquet, three batches) |
| `data/voices/`, `data/pilot/` | Voice references and pilot recordings |
| `scripts/` | Corpus builder, voice design, audio generation |
| `docs/` | Model and data description, annotation format, HF cards |
| `web/` | The in-browser demo (Cloudflare Pages) |

## License

Code and weights: Apache-2.0. Dataset: CC BY 4.0. The Common Voice reference
recordings are CC0; the conversational slot pools and templates were written
for this project. Neither eSpeak NG nor any pronouncing dictionary is used
anywhere.
