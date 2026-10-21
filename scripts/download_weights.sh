#!/usr/bin/env bash
# Download the weights needed to run UniAVC into ./models.
#
#   bash scripts/download_weights.sh            # model.safetensors + Gemma-3
#
# The released checkpoint is a single file, model.safetensors, holding the DiT
# together with the video/audio VAEs, the vocoder and the text post-modules, so
# one download covers every task.
#
# Two repositories are involved:
#
#   TK3105/UniAVC            the released model: one file, model.safetensors.
#   google/gemma-3-12b-...   the text encoder. A general model, not ours to republish.
#
# So there is no third repository and no base-model overlay: model.safetensors is a
# complete model and replaces the base model's weights outright.
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-models}"
for arg in "$@"; do
  case "$arg" in
    -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

command -v hf >/dev/null 2>&1 || {
  echo "the 'hf' command was not found. Install it with:" >&2
  echo "  pip install -U huggingface_hub" >&2
  exit 1
}

mkdir -p "${MODELS_DIR}"

echo "==> Gemma-3 text encoder (~23 GB)"
hf download google/gemma-3-12b-it-qat-q4_0-unquantized \
  --local-dir "${MODELS_DIR}/gemma-3-12b-it-qat-q4_0-unquantized"

echo "==> UniAVC checkpoint (~46 GB)"
hf download TK3105/UniAVC model.safetensors \
  --local-dir "${MODELS_DIR}/UniAVC"

echo
echo "Done. Weights are under ${MODELS_DIR}/"
echo
echo "Run with:"
echo "  python inference/infer_t2av.py --prompt '...' -o out.mp4"
echo
echo "Both model paths are resolved from ${MODELS_DIR}/ automatically."
