#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# download_weights.sh – Fetch vision-model checkpoints & label files.
#
#  • ResNet-18-Places365 weights          (≈ 46 MB)
#  • IO_places365.txt  – indoor/outdoor map
#  • categories_places365.txt – 365 labels
#
# The MIT Places2 server is occasionally offline.  This script:
#   1. Tries the original MIT URL.
#   2. Falls back to a Hugging Face mirror if the first download fails.
# ----------------------------------------------------------------------------
set -euo pipefail

MODEL_DIR="data/models"
mkdir -p "${MODEL_DIR}"

WEIGHTS_FILE="${MODEL_DIR}/resnet18_places365.pth.tar"
MIT_URL="http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar"
HF_URL="https://huggingface.co/azurite/resnet18_places365/resolve/main/resnet18_places365.pth.tar"

# ---------------------------------------------------------------------------
# helper: download  <url>  <target_file>
dl() {
  local url="$1"
  local out="$2"
  echo "⬇️   Downloading from ${url}"
  curl -L --retry 3 --connect-timeout 15 --output "${out}" "${url}"
}

# ── 1 ▸ ResNet-18 weights ───────────────────────────────────────────────────
if [[ -f "${WEIGHTS_FILE}" ]]; then
  echo "✅  ResNet-18 Places365 weights already present."
else
  set +e
  dl "${MIT_URL}" "${WEIGHTS_FILE}"
  STATUS=$?
  set -e
  if [[ $STATUS -ne 0 ]]; then
    echo "⚠️   MIT server unavailable.  Falling back to Hugging Face mirror …"
    dl "${HF_URL}" "${WEIGHTS_FILE}"
  fi
fi

# ── 2 ▸ Label & helper files (GitHub raw links are stable) ─────────────────-
IO_LIST="${MODEL_DIR}/IO_places365.txt"
CAT_LIST="${MODEL_DIR}/categories_places365.txt"

if [[ ! -f "${IO_LIST}" ]]; then
  curl -L https://raw.githubusercontent.com/csailvision/places365/master/IO_places365.txt \
       -o "${IO_LIST}"
  echo "✅  IO_places365.txt downloaded."
else
  echo "✅  IO_places365.txt already present."
fi

if [[ ! -f "${CAT_LIST}" ]]; then
  curl -L https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt \
       -o "${CAT_LIST}"
  echo "✅  categories_places365.txt downloaded."
else
  echo "✅  categories_places365.txt already present."
fi

echo -e "\n🎉  All vision weights & label files are ready."
