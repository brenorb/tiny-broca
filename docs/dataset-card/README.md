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
- config_name: batch4
  data_files:
  - split: train
    path: batch4/data/train-*.parquet
  - split: validation
    path: batch4/data/validation-*.parquet
  - split: test
    path: batch4/data/test-*.parquet
- config_name: batch5
  data_files:
  - split: train
    path: batch5/data/train-*.parquet
  - split: validation
    path: batch5/data/validation-*.parquet
  - split: test
    path: batch5/data/test-*.parquet
---
# tiny-broca dataset

A synthetic **Brazilian Portuguese** speech corpus with two voices, F1
(female) and M1 (male), made to train the tiny-broca text-to-speech model.
Every sentence is read by both voices.

| | |
|---|---|
| Sentences | 9,384 in v0, plus 4,500 new and 2,800 re-read in v1 |
| Clips | v0: 16,434 accepted of 19,504; v1: 14,047 accepted of 14,600 |
| Duration | 17.3 hours (v0) plus 16.4 hours (v1) |
| v1 training set | 26,130 clips, 28.2 hours (see [Versions](#versions)) |
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
batch4 = load_dataset("brenorb/tiny-broca", "batch4")   # v1
batch5 = load_dataset("brenorb/tiny-broca", "batch5")   # v1

row = batch1["train"][0]
print(row["voice"], row["text"])
audio = row["audio"]["array"]
```

## Contents

Counts are the accepted clips' unique sentences; each sentence is read by
both voices, so the clip count is roughly twice the sentence count.

| Config | Split | Sentences | Clips | Contents |
|---|---|---:|---:|---|
| `batch1` | train | 3,560 | 7,119 | conversational sentences and Common Voice |
| `batch1` | validation | 647 | 1,294 | same |
| `batch1` | test | 404 | 807 | same |
| `batch2` | train | 3,327 | 6,653 | Common Voice and templates |
| `batch3` | train | 343 | 561 | money phrases: amounts, cards and bills |
| `batch4` | train | 3,936 | 7,729 | v1: what v0 read badly (loanwords, money, numbers, dates, symbols...) |
| `batch4` | validation | 221 | 436 | same |
| `batch4` | test | 228 | 452 | same |
| `batch5` | train | 2,536 | 4,994 | v1: v0 sentences re-read from their spoken form |
| `batch5` | validation | 91 | 180 | same |
| `batch5` | test | 131 | 256 | same |

`batch1` mirrors the first batch of the corpus this project follows
(conversational speech plus natural sentences); `batch2` is training data
only, mirroring its second batch (news-style sentences plus sentences that
exercise every rule of the text normalisation). `batch3` is training data
only: phrases dominated by money amounts, generated after the first training
run so the model hears numbers read at their full value. Splits are made by
sentence: both readings of a sentence are always in the same split, so the
validation and test sentences are the same for every model trained on this
corpus.

`batch4` and `batch5` were made for v1. `batch4` adds 4,500 sentences for what
v0 read badly: English loanwords and brands, money with "reais" in every
position, long numbers and identifiers, dates, times, symbols and math, units,
ordinals, acronyms, gender agreement, URLs, stress, questions and
exclamations. `batch5` re-reads the 2,800 v0 sentences whose spoken form
differs from what is written (every sentence with a number, symbol, acronym or
loanword), plus the template sentences that had grammar errors, corrected.

## Versions

The v0 files are unchanged. v1 was trained on `batch1` and `batch2` without
the 2,821 v0 sentences listed in `v1_superseded.txt` (all of `batch3` among
them), plus `batch4` and `batch5`: 26,130 clips, 28.2 hours. The superseded v0
clips were read from digits and symbols, so some say the wrong words.

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
   were generated a second time; clips that failed twice were left out (3,070
   in v0).
4. Accepted clips were peak-normalised to 0.95.

For `batch4` and `batch5` the teacher read the spoken form, the words a
Brazilian reader says (`R$ 3,00` → "três reais", `online` → "onlaine"), and
the screening compared spoken forms with numbers included. v0's screening
ignored numbers, which let misread amounts through. Clips Whisper rejected got
a second opinion from parakeet, and forced alignment was the third check.

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
| v1 coverage sentences (`batch4`), from templates and word lists | 4,500 | this project |

In v0, accepted clips by source: Common Voice 7,706, conversational 6,589,
templates 1,578, money 561. `batch5` re-reads sentences of these sources.

The conversational slots keep the corpus reproducible: the same seed produces
the same sentences.

## Columns

| Column | Description |
|---|---|
| `id` | Clip ID: batch, voice, split and sentence index, e.g. `b1_F1_test_00390` |
| `text_id` | Sentence ID in the corpus, e.g. `broca01_00000` |
| `batch` | 1 to 5 (see [Contents](#contents)) |
| `split` | train, validation or test |
| `text` | What the teacher read. In `batch4` and `batch5` this is the spoken form, with loanwords respelled the Brazilian way |
| `written` | `batch4` and `batch5` only: the sentence as written |
| `source` | `cv`, `converse`, `templates` or `money`; `v1-<category>` in `batch4`; the v0 source plus `+v1` in `batch5` |
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
in the code repository. 14,290 clips are annotated; 933 failed the alignment checks.

`annotations-v1/` holds the same annotations for the v1 training set: 25,108
clips (train 22,274, validation 1,577, test 1,257); 1,022 failed the
alignment checks.

## Limitations

- The speech is synthetic. It carries the pronunciation habits and errors of
  the model that generated it (for example, a reduced vowel in the standalone
  word *se* in some clips).
- Two voices, one accent, read speech only.
- The two voice references are real recordings of Common Voice contributors;
  they are included as `references/` for provenance and are not training
  audio.

## Errata

- v0's `voices.json` credited F1 to M1's source clip. F1 comes from Common
  Voice clip `cv_36536891`; `references/voices.json` is corrected in v1, but
  the `reference_audio` column of the v0 files keeps the old value.
- The first version of this card said 0 clips failed twice. 3,070 of the
  19,504 v0 clips did, and they are not in the dataset.

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
