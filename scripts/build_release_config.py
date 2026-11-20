"""Build the release config.json (Hugging Face + web demo).

The training run writes a flat config (``Config.to_dict``), which is what the
pipeline loads. The released config nests it under ``"model"`` — the same
layout tiny-voca ships — and adds the web-facing metadata the browser worker
reads (``config.model.max_tokens``, ``config.voices``, ``config.sample_rate``)
plus the phonemiser identity and the letter vocabulary, so a consumer can
verify the front end without trusting the code.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from tiny_broca.export import INPUTS, OUTPUTS  # noqa: E402
from tiny_broca.text import SYMBOL_TO_ID, frontend_fingerprint  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="flat training config (runs/<run>/config.json)")
    parser.add_argument("--checkpoint", type=Path, required=True, help="inference checkpoint (for the step count)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = json.loads(args.config.read_text(encoding="utf-8"))
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)

    release = {
        "model": model,
        "step": state.get("step"),
        "sample_rate": model["sample_rate"],
        "hop": model["hop"],
        "voices": list(model["speakers"]),
        "phonemizer": {"name": "pt", "version": 1, "fingerprint": frontend_fingerprint()},
        "stress": {"none": 0, "stressed": 1},
        "onnx": {"file": "onnx/model.onnx", "inputs": list(INPUTS), "outputs": list(OUTPUTS)},
        "vocab": SYMBOL_TO_ID,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(release, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output} ({args.output.stat().st_size / 1024:.1f} KB, step {release['step']:,})")


if __name__ == "__main__":
    main()
