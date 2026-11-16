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
generator, has 4.5M parameters and was trained from scratch on about 19 hours
of synthetic speech.

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

## Files

| File | Contents |
|---|---|
| `tiny-broca.pt` | PyTorch weights, float32 |
| `onnx/model.onnx` | ONNX export, float32, opset 17 |
| `config.json` | Model settings: sample rate, windows, voices, vocabulary size |
| `text.py` | The front end module, verbatim (the web demo runs it in the browser) |
| `samples/` | The audio on this page |

## Samples

| Text | F1 | M1 |
|---|---|---|
SAMPLE_ROWS

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
| Data | [brenorb/tiny-broca](https://huggingface.co/datasets/brenorb/tiny-broca): 8,984 sentences, two voices |
| Steps | 104,000 |
| Losses | Multi-resolution spectral losses; adversarial loss with multi-period and multi-resolution discriminators after step 8,000 |
| Hardware | |

The training audio is synthetic: every clip was generated with Qwen3-TTS 1.7B
Base (Apache-2.0) in one of the two voices. The voices are clones of two
Common Voice speakers (CC0): short clean recordings of real people, used only
as voice prompts for the generator. No recording of a real speaker is part of
the model. Letter durations come from forced alignment with
`jonatasgrosman/wav2vec2-xls-r-1b-portuguese`, which is used only to label
the training data. No pretrained weights are part of the model.

## Evaluation

Word error rate of synthesized speech, transcribed with Whisper
(multilingual, pt). The recordings of the same sentences are scored the same
way for reference. (Pending the training run.)

## Limitations

- Brazilian Portuguese only, two fixed voices, no voice cloning.
- The voice is clear but can sound slightly synthetic.
- The same text is always read the same way. There is no sampling of prosody.
- Foreign names and unusual spellings are often mispronounced.
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
