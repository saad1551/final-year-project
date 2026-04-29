#!/usr/bin/env bash
# Run ON the VM to set up the Python environment and Playwright browser.
# Assumes a Deep Learning VM image (CUDA + Python pre-installed).

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/final-year-project}"
ENV_NAME="${ENV_NAME:-insta}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

echo "===================================================================="
echo "VM environment setup"
echo "===================================================================="
echo "Repo dir : $REPO_DIR"
echo "Conda env: $ENV_NAME (python $PYTHON_VERSION)"
echo "===================================================================="

cd "$REPO_DIR"

if ! command -v conda >/dev/null; then
  echo "Installing miniconda..."
  curl -fsSL -o /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
  eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
  conda init bash
fi

eval "$(conda shell.bash hook)"

if ! conda env list | grep -q "^$ENV_NAME "; then
  if [[ -f "environment.yml" ]]; then
    echo "Creating conda env from environment.yml..."
    conda env create -n "$ENV_NAME" -f environment.yml
  else
    echo "Creating conda env (fallback path)..."
    conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
  fi
fi

conda activate "$ENV_NAME"

if [[ -f "requirements.txt" ]]; then
  echo "Installing requirements.txt..."
  pip install --upgrade pip
  pip install -r requirements.txt
fi

echo "Installing Playwright browsers..."
python -m playwright install --with-deps chromium

echo ""
echo "Smoke checks:"
python - <<'PY'
import torch
print(f"  torch: {torch.__version__}, cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device: {torch.cuda.get_device_name(0)}")
PY

echo ""
echo "Done. Next: bash gcp/launch_training.sh"
