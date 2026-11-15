# tiny-broca annotations: letter tokens and validated durations

Annotation layer for the recordings in this dataset. It adds, for every
accepted recording, the letter token sequence and the number of acoustic frames
each token lasts. The audio shards are untouched; nothing here replaces them.

The English original annotated with eSpeak phonemes (59 phones, 76 symbols)
because a permissive English pronouncing dictionary exists. No permissive
Portuguese equivalent exists, so this version's token alphabet is the
Portuguese written form (58 symbols, front end version 1): one token per
letter, including the accented graphemes, with stress as a parallel 0/1
feature. The same downstream guarantees apply: tokens are produced by exactly
one program, one aligned label is one token, and the aligner never enters the
model.

## How these annotations are made

1. **Text to letters.** The front end (`tiny_broca.text`) normalises the text
   (NFKC, casefold, Brazilian number/date/money conventions) and tokenises it
   into 58 symbols: letters, punctuation, and explicit tokens for utterance
   edges, word boundaries and the unknown character. Stress is 1 on the
   letters of the stressed syllable of each word and 0 elsewhere.
2. **Forced alignment.** A pretrained Portuguese recogniser,
   `jonatasgrosman/wav2vec2-xls-r-1b-portuguese` (revision
   `8926743abe7e95bb05b81b64305cb3c5fa85173f6b0`, Apache-2.0, wav2vec 2.0
   XLS-R 1B fine-tuned with CTC on written Portuguese) aligns each letter
   sequence to its recording with CTC Viterbi. **It is used only to annotate
   this data. None of its weights or outputs are part of the TTS model.** Its
   47-token vocabulary covers every inventory letter, so one aligned label is
   one token; the check runs at load time.
3. **Boundaries.** The recogniser emits each label once, at the start of the
   letter's sound. The English original calibrated an onset offset of about 19
   ms on its own recordings; the same offset is carried over here, and the
   quality metrics below are the check that it still fits. A letter lasts
   from its own onset to the next letter's onset.
4. **Pauses.** Silence (frame level below the recording's loud level minus
   45 dB) between two letters is given to the pause-capable token there: BOS,
   EOS, punctuation, or the word boundary. At least 75 ms of silence is
   needed, otherwise the gap stays with the letters (stop closures).
5. **Frame grid.** Durations are counted in 256-sample frames at 24 kHz, frame
   `i` centred on sample `i * 256`, the same grid as the stored mel and pitch
   features. They sum to the frame count exactly and every letter has at
   least one frame.

## Quality checks

Per row, the label-free checks of `align.quality` gate acceptance: mean token
probability, weak-letter fraction, maximum letter length, letters per second,
vowel voicing share, voiceless voicing share and pause statistics. The limits
mirror the English original (mean token probability ≥ 0.5, at most a quarter
of letters weak, a letter at most 70 frames, 4–30 letters per second, at least
half of the vowel frames voiced).

The label-free timing probe (voicing flips at vowel↔voiceless boundaries) and
the per-row rejection reasons are reported in the same form as the English
original: `<split>.rejected.jsonl` with reasons, `summary.json` with the
quantiles and the probe. (Pending the first full annotation run: the corpus
audio does not exist yet.)

## Files

| File | Content |
|---|---|
| `train.jsonl`, `val.jsonl`, `test.jsonl` | accepted rows: `id`, `speaker`, `text`, `symbols`, `tokens`, `stress`, `durations`, `frames`, `qc`, `provenance`, fingerprints |
| `*.rejected.jsonl` | rejected rows with reasons |
| `annotations.json` | identity of this run (front end, aligner revision, settings, source hashes), completion flag |
| `summary.json` | QC quantiles, timing probe, duration tables |
| `letters.jsonl` | front-end output per distinct text |
| `raw/` | per-recording alignment journal (resumable) |

`path` in a row is relative to the acoustic feature directory on the training
machine. Provenance fields (`text_id`, `origin`, `source_id`, `source_license`,
`source_url`, `teacher_model`, `reference_audio`) are copied from the source
rows; the dataset's `LICENSE.md` continues to apply.

## Limits

* No hand-labelled boundaries exist for this corpus; accuracy is estimated from
  the voicing probe and from whether a model trained on these durations speaks
  intelligibly.
* The teacher (Qwen3-TTS) reduces the vowel of the standalone word *se* in
  some clips ("a gente se encontra" can surface with a near-silent *e*); about
  5% of the corpus sentences contain a standalone *se*. Such clips are
  identified in QC and left to the gate, not silently kept.
* A pause inside a word has no token and stays with a letter, as in the
  English original.

Reproduce on the training machine:

```bash
python -m tiny_broca.annotate --features <acoustic feature dir> \
  --output <new directory> --dataset-revision <corpus revision>
```
