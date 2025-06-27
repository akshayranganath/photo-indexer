#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# prepare_env.sh  –  bootstrap a Python virtual-env and install all runtime
#                    dependencies for the Photo-Indexer project.
#
# Prerequisites (already installed on your machine):
#   • Homebrew            • Xcode command-line tools
#   • Python 3.13         • exiftool
#   • Ollama              • git clone of this repo
# ---------------------------------------------------------------------------

set -euo pipefail

VENV_DIR="${HOME}/.venvs/photoindex"
PY_BIN="$(command -v python3.13 || true)"

# ── 1. sanity checks ────────────────────────────────────────────────────────
if [[ -z "${PY_BIN}" ]]; then
  echo "❌  Python 3.13 not found in \$PATH.   Aborting." >&2
  exit 1
fi

if [[ ! -f "requirements.txt" ]]; then
  echo "❌  Run this script from the repo root where requirements.txt lives." >&2
  exit 1
fi

# ── 2. create / reuse virtual-env ───────────────────────────────────────────
echo "📦  Creating virtual-env at: ${VENV_DIR}"
"${PY_BIN}" -m venv "${VENV_DIR}"

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

# ── 3. upgrade packaging tooling ───────────────────────────────────────────
python -m pip install --upgrade pip wheel setuptools

# ── 4. install Python dependencies ─────────────────────────────────────────
echo "📚  Installing project requirements …"
pip install -r requirements.txt

# ── 5. success banner ──────────────────────────────────────────────────────
cat <<EOF

🎉  Environment ready!

To start using it:
  source "${VENV_DIR}/bin/activate"

Then fetch the vision weights:
  bash scripts/download_weights.sh     # ResNet-18 Places, sample NEFs
  ollama pull llava:7b-q4_k_m          # caption model

Finally run the indexer:
  pi index /path/to/photo/folder --workers 8

EOF
