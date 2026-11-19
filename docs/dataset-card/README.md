---
language: pt
task_categories:
- text-to-speech
license: cc-by-4.0
pretty_name: tiny-broca
configs:
- config_name: batch1
  default: true
  data_files:
  - split: train
    path: data/train-*.parquet
  - split: validation
    path: data/validation-*.parquet
  - split: test
    path: data/test-*.parquet
- config_name: batch2
  data_files:
  - split: train
    path: batch2/data/train-*.parquet
- config_name: batch3
  data_files:
  - split: train
    path: batch3/data/train-*.parquet
---
# tiny-broca dataset

A synthetic **Brazilian Portuguese** speech corpus with two voices, F1
(female) and M1 (male), made to train the tiny-broca text-to-speech model.
Every sentence is read by both voices.

| | |
|---|---|
| Sentences | 9,384 |
| Clips | 16,434 (after QC rejection; 19,504 generated) |
| Duration | about 19 hours |
| Voices | F1, M1 |
| Audio | 24 kHz mono, 16-bit FLAC, stored inside the Parquet files |
| Language | Portuguese (Brazil) |
| License | CC BY 4.0 |

## Usage

```python
from datasets import load_dataset

batch1 = load_dataset("brenorb/tiny-broca", "batch1")
batch2 = load_dataset("brenorb/tiny-broca", "batch2")
batch3 = load_dataset("brenorb/tiny-broca", "batch3")

row = batch1["train"][0]
print(row["voice"], row["text"])
audio = row["audio"]["array"]
```

## Contents

Counts are the accepted clips' unique sentences; each sentence is read by
both voices, so the clip count is roughly twice the sentence count.

| Config | Split | Sentences | Clips | Text |
|---|---|---:|---:|---|
| `batch1` | train | 3,560 | 7,119 | conversational sentences and Common Voice |
| `batch1` | validation | 647 | 1,294 | same |
| `batch1` | test | 404 | 807 | same |
| `batch2` | train | 3,327 | 6,653 | Common Voice and templates |
| `batch3` | train | 343 | 561 | money phrases: amounts, cards and bills |

`batch1` mirrors the first batch of the corpus this project follows
(conversational speech plus natural sentences); `batch2` is training data
only, mirroring its second batch (news-style sentences plus sentences that
exercise every rule of the text normalisation). `batch3` is training data
only: phrases dominated by money amounts, generated after the first training
run so the model hears numbers read at their full value. Splits are made by
sentence: both readings of a sentence are always in the same split, so the
validation and test sentences are the same for every model trained on this
corpus.

## How it was made

1. The two voices were created by voice cloning: each is Qwen3-TTS 1.7B
   prompted with a short clean recording of a real speaker from Mozilla Common
   Voice (CC0). Voices designed from text descriptions (the English model's
   method) were auditioned and rejected as less natural. The reference clips
   are in `references/`, with their sources and seeds in `voices.json`.
2. Each sentence was synthesised in both voices with Qwen3-TTS 1.7B Base,
   using the reference clips as the voice prompt.
3. Each clip was transcribed with Whisper (multilingual, pt). Clips above 15%
   word error rate, or with implausible duration, speaking rate or silence,
   were generated a second time; clips that failed twice were left out. The
   rejections are listed in `generation/*/review.jsonl`. (Counts pending the
   first full run.)
4. Accepted clips were peak-normalised to 0.95.

The generation scripts and their settings are in `generation/`.

The automatic screening catches wrong or missing words. It does not measure
naturalness, and the corpus has not been checked by listeners clip by clip.

## Text sources

The `source` column records where every sentence came from.

| Source | Sentences | License of the text |
|---|---:|---|
| Mozilla Common Voice (Brazilian Portuguese) | 4,405 | CC0 |
| Conversational sentences, generated deterministically from written slot pools | 3,500 | this project |
| Templates: Brazilian numbers, dates, times, money, identifiers, questions | 1,079 | this project |
| Money phrases (`batch3`) | 400 | this project |

Accepted clips by source: Common Voice 7,706, conversational 6,589,
templates 1,578, money 561.

The conversational slots keep the corpus reproducible: the same seed produces
the same sentences.

## Columns

| Column | Description |
|---|---|
| `id` | Clip ID: batch, voice, split and sentence index, e.g. `b1_F1_test_00390` |
| `text_id` | Sentence ID in the corpus, e.g. `broca01_00000` |
| `batch` | 1 (conversational and Common Voice) or 2 (Common Voice and templates) |
| `split` | train, validation or test |
| `text` | The sentence that was read |
| `source` | `cv`, `converse` or `templates` |
| `source_license`, `source_url` | `cc0-1.0` for Common Voice sentences, otherwise `project-own` |
| `voice` | F1 or M1 |
| `audio` | 24 kHz audio, as bytes, path and sample rate |
| `duration_seconds`, `sample_rate` | Audio length and rate |
| `asr_transcript`, `normalized_wer` | Whisper transcript and its word error rate against `text` |
| `gen_seconds`, `rtf` | Generation time and real-time factor of the clip |
| `teacher_model`, `reference_audio` | Model and voice prompt used |

## Annotations

`annotations/` holds the letter-level annotations used for training: the
letter tokens, the stress feature, and the duration of each letter in 10.7 ms
frames, from forced alignment with
`jonatasgrosman/wav2vec2-xls-r-1b-portuguese`. Clips that fail the alignment
checks are listed with a reason in the `*.rejected.jsonl` files. See
[docs/annotations.md](https://github.com/brenorb/tiny-broca/blob/main/docs/annotations.md)
in the code repository. (Pending the first full run.)

## Limitations

- The speech is synthetic. It carries the pronunciation habits and errors of
  the model that generated it (for example, a reduced vowel in the standalone
  word *se* in some clips).
- Two voices, one accent, read speech only.
- The two voice references are real recordings of Common Voice contributors;
  they are included as `references/` for provenance and are not training
  audio.

## License

CC BY 4.0. Please credit "tiny-broca dataset, brenorb" and link to this page.

```bibtex
@misc{brenorb2026tinybrocadata,
  author = {brenorb},
  title  = {tiny-broca dataset: synthetic two-voice Brazilian Portuguese speech},
  year   = {2026},
  url    = {https://huggingface.co/datasets/brenorb/tiny-broca}
}
```

The Common Voice sentences are CC0, from
https://github.com/common-voice/common-voice. The audio was generated with
Qwen3-TTS (Apache-2.0). See `LICENSE.md`.
