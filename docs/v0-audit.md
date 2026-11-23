# How v0 fails (audit for v1)

Audit of the released v0 model (`runs/tiny-broca-ft2/inference.pt`, step
118,000, git tag `v0`) with the v0 front end. Sources: a probe suite of 287
sentences in 29 categories read by both voices (636 clips, transcribed by
parakeet-v3 through `fscript`, faster-whisper medium, and greedy CTC letters
from the project's wav2vec 2.0 aligner, scored on spoken forms), spectrograms,
an audit of the teacher's training clips, and the v0 build log.
Scripts and data: `runs/probe-v0/` (not in git).

## 1. Ranked failure modes

| # | Severity | Failure | Evidence |
|---|---|---|---|
| 1 | content lost | The front end emits `<unk>` for symbols and formats it does not know (`>`, `/`, `º`, `14h`, digits of numbers over 5 digits, `#`, `@`, emoji). `<unk>` never occurs in training, so each one becomes ~107 ms of vowel-like noise. | 54/287 probes. "4 > 3" heard "quatro e três"; "14/03" heard "quatorze é três"; "às 14h" heard "a Z"; "1º" heard "hein, O". |
| 2 | wrong words | The normaliser reads many formats wrongly, and the model says exactly what it is given. | 115/287 probes reach the model with wrong or missing words: "12,5%" → "cento e vinte e cinco por cento"; "1.000.000" read digit by digit; "R$ 1 mil" → "um real mil"; "R$ 0,50" → "zero reais e cinquenta centavos"; "US$ 10" → "usar dez"; numbers never agree in gender ("dois horas", "duzentos pessoas"); no "menos", no ranges, no fractions, no ordinals, no units, no abbreviations, no roman numerals. |
| 3 | content lost | "reais" and "centavos" are swallowed. | Neither ASR hears "reais" in 19/38 clips; "custa três reais" is heard "custa três vezes" in both voices. Mid-utterance "reais" lasts a median 181 ms (36 ms per letter, below the corpus 5th percentile). |
| 4 | unintelligible | English loanwords are read with Portuguese letter rules. | hacker → "aquele", crush → "cruz", like → "lique", home office → "homem o fiz", site → "sate", iPhone → "ifione"; "online" has a steady /i/. Loanwords that occur in the corpus (notebook, shopping, show) are fine. |
| 5 | unintelligible | Spelled-out acronyms. | 27/34 missed by both ASRs: CNPJ → "campeão", "Desliga a TV" → "Desliga-te", EUA → "Zé". |
| 6 | unintelligible | Long numbers are compressed. | Number words last 0.86× the teacher's duration in utterances with 10+ number words (0.70× at p10); the conjunction "e" is swallowed 26 times. User report: `novel07_F1` "acelera falando o número e nem dá para entender o final" (50% WER at 104k, 110k and 118k steps). |
| 7 | prosody → intelligibility | The stress feature is wrong for 37/105 audit words (every unaccented oxytone ending in -i, -u, -im, -um, -om, -z, and hiatus words). In 14/26 such cases the model lengthens the wrong syllable. | caju → "cajo" (4/4), atum → "Atom", Itu → "dito", abacaxi → "abacate". |
| 8 | prosody | Yes/no questions are not intonated. | 1/6 get a "?" from either ASR (wh-questions 4/4). M1 questions end 1.8–2.2 semitones down. "!" occurs only 306 times in training. |
| 9 | naturalness | Utterance-final money words fade into a whisper (the teacher does it too). | |

Clean: nasals, x/ch/qu/gu/lh/nh, hiatus vowels, "se/de/que" runs, plain long
sentences (WER ≤ 6%), short utterances.

## 2. Root cause of the number and money failures: the training data

v0's teacher (Qwen3-TTS) was given the **written** sentence ("R$ 7.586,29",
"20/03/2022"), and the quality gate masked every number before comparing the
transcript, so nobody checked what the teacher actually said.

* An audit of 60 random accepted money clips (Whisper small, spoken-form WER
  against the front end's reading) gave a mean WER of 31%, and 50/60 were
  above 15%. The teacher read amounts digit by digit or garbled them:
  "R$ 7.586,29" → "resatras 58862919", "R$ 42.030,02" → "42-030-02",
  "20/03/2022" → "2023-2022", "02/07/2026" → "0207-2026". Other numbers: mean
  13.6%, 24/60 above 15%. Plain sentences: 2.0%, 0/60.
* The aligner's own CTC hears "reais" in **0 of 585** `R$` training clips and
  "centavos" in 0 of 427 clips with cents; the teacher said "erre" in about
  19% of them, "dólares" in 32, or "vírgula". The aligner squeezed the letters
  of "reais" onto whatever was there (median 203 ms, sometimes 107 ms,
  sometimes 896 ms). When "reais" was written out, the teacher said it 7/7
  times at 320–400 ms.
* This is also why the aligner rejected 447 of 561 batch3 money clips, and why
  the money fine-tunes did not help.
* Fed the spoken form instead ("três reais", "quatorze de março", "quatro
  maior que três"), the same teacher reads every one of these correctly
  (`runs/teacher-ab/`).

## 3. Corpus defects

* 501 sentences (5.3%) carry conversational template errors: "vou à a escola"
  (181), "na o aeroporto" (230), "O aluguel sobiu" (23), "Moro no Rua" (73),
  tails without punctuation ("amanhã de manhã faz sentido").
* 83.6% of training utterances contain no number.
* A clip-id collision lost 1,668 sentence-voice pairs until a top-up run.

## 4. Front-end bugs (v0 `text.py`)

| Bug | Example |
|---|---|
| NFKC runs before the º/ª replacement, so it is dead code; `ordinal_words` is never called | `1º` → "1o" (`<unk>` digit) |
| `%` and `°` drop the decimal comma | `12,5%` → 125 |
| `°C` leaves a stray "c"; negatives keep the hyphen | "grausc" |
| dates need a year; no ISO or month/year dates; day 1 → "um" | `14/03` → "quatorze/três" |
| `14h`, `7h15min` unhandled | `<unk>` |
| numbers are always masculine | "dois horas" |
| the document-ID rule catches numbers with two thousands dots | `1.000.000` read as an ID |
| integers of 6+ digits unhandled; leading zeros dropped | `123456` → `<unk>`; `007` → "sete" |
| money: one-digit cents read as units, no scale words, "zero reais", no "de" before reais, no US$/€ | `R$ 10,5` → "dez reais e cinco centavos" |
| "e" after every milhão/bilhão | "um milhão e duzentos e trinta e quatro mil" |
| no symbols, ranges, fractions, units, abbreviations, roman numerals, acronyms, URLs, loanwords | |
| stress rule covers only final l/n/r/x; no hiatus | aqui → [a]qui, reais → whole word |
| the QC normaliser masks numbers (and turns its own mask into a literal word) | money errors invisible to QC and evaluation |

## 5. Other findings

* The released model card's evaluation section still says "Pending the
  training run" (the publish script's regex never matched); the dataset card
  says "0 clips failed twice" (3,070 did); `references/voices.json` lists F1's
  source as M1's clip.
* The evaluation sets were tiny (8 clips per split, 16 novel sentences).
* The source-filter head's pulse train never stops, so unvoiced sounds are
  partly built out of pulses (the "buzz" Alpa found in the same head).
  Switching the harmonic excitation to noise in unvoiced frames fixed one of
  four "reais" probes at inference; the main cause is the data.
* The voiced fraction of generated speech is about 10 points above the
  recordings' (partly by design: vowel frames are marked voiced).
