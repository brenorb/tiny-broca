#!/usr/bin/env bash
# Fetch the files the demo serves into public/assets (they are not stored in git).
#
#   bash fetch_assets.sh /path/to/tiny-broca
#
# Downloads the ONNX model and its config from the Hugging Face repository and
# copies the front end module (the browser runs the exact Python code the
# training and annotation pipelines use). While the model repository is
# private, the token of the logged-in Hugging Face user is used.
set -euo pipefail
cd "$(dirname "$0")"
REPO="${1:?path to the tiny-broca repository}"
BASE="https://huggingface.co/brenorb/tiny-broca/resolve/main"
TOKEN_FILE="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"
mkdir -p public/assets
get() {
  if [ -f "$TOKEN_FILE" ]; then
    curl -fsSL -H "Authorization: Bearer $(cat "$TOKEN_FILE")" -o "$2" "$BASE/$1"
  else
    curl -fsSL -o "$2" "$BASE/$1"
  fi
}
get onnx/model.onnx public/assets/model.onnx
get config.json public/assets/config.json
cp "$REPO/src/tiny_broca/text.py" public/assets/text.py
# The browser caches the assets under this key (see worker.js): it changes
# whenever a served file does, so returning visitors fetch the new release.
cat public/assets/model.onnx public/assets/config.json public/assets/text.py | shasum -a 256 | cut -c1-16 > public/assets/version.txt
ls -l public/assets
