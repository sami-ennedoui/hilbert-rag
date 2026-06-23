#!/usr/bin/env bash
# Create the CPU-only Python 3.12 venv and install dependencies.
# Installs torch from the PyTorch CPU index first so pip does not pull the large CUDA wheel.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=python3.12
command -v "$PY" >/dev/null 2>&1 || { echo "python3.12 not found"; exit 1; }

"$PY" -m venv .venv
# shellcheck disable=SC1091
. .venv/bin/activate

python -m pip install -U pip wheel
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"

python - <<'PY'
import torch, faiss, sentence_transformers, fastapi, hilbertcurve, sklearn
print("torch", torch.__version__, "cuda?", torch.cuda.is_available())
print("sentence_transformers", sentence_transformers.__version__)
print("faiss / fastapi / hilbertcurve / sklearn import OK")
PY
echo "ENV OK"
