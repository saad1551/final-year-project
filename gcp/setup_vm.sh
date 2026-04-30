#!/usr/bin/env bash
# Run ON the VM to set up everything training needs.
#
# Idempotent: safe to re-run after preemption + restart. Each install step
# guards on "already installed".
#
# DL VM (pytorch-2-9-cu129-ubuntu-2204) ships with:
#   - Python 3.10.12 + PyTorch 2.9.1+cu129 (system-wide)
#   - screen, tmux, git, build-essential
# We add: Node.js LTS, npm packages for the Playwright server, and the
# Python deps needed for training/eval.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/final-year-project}"

echo "===================================================================="
echo "VM environment setup  ($(date))"
echo "===================================================================="
cd "$REPO_DIR"

# 1. Node.js LTS (for the Playwright JS server in javascript/server/)
if ! command -v node >/dev/null 2>&1; then
  echo "[setup] Installing Node.js LTS..."
  curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
  sudo apt-get install -y nodejs
else
  echo "[setup] Node.js already installed: $(node --version)"
fi

# 2. npm install for the Playwright server
if [[ -d javascript/server ]]; then
  if [[ ! -d javascript/server/node_modules ]]; then
    echo "[setup] Installing JS server dependencies..."
    pushd javascript/server >/dev/null
    npm install --silent
    popd >/dev/null
  else
    echo "[setup] javascript/server/node_modules already present"
  fi
fi

# 3. Playwright browsers for the JS server.
# IMPORTANT: install browsers as the *user* (not sudo) so they land in
# ~/.cache/ms-playwright where the JS server (running as the user) can find
# them. Using sudo would put them in /root/.cache and break runtime.
# We do still need sudo for system deps (libs, fonts) once.
echo "[setup] Installing Playwright system deps (sudo)..."
sudo npx playwright install-deps chromium >/dev/null 2>&1 || true
echo "[setup] Installing Chromium browser into user cache..."
npx playwright install chromium

# 4. Python deps for training / eval
# Skip vllm, sk-video, gradio-client, anthropic, lxml-html-clean — not used at train time
PY_DEPS=(
  "transformers>=4.35.0"
  "accelerate>=0.24.0"
  "peft>=0.7.0"
  "bitsandbytes>=0.45.0"
  "sentencepiece>=0.1.99"
  "protobuf>=3.20.0"
  "stable-baselines3>=2.0.0"
  "gymnasium>=0.28.0"
  "pandas>=1.5.0"
  "huggingface-hub>=0.16.4"
  "numpy>=1.24.0"
  "datasets>=2.14.0"
  "tabulate>=0.9.0"
  "playwright>=1.40.0"
  "requests>=2.31.0"
  "beautifulsoup4>=4.12.0"
  "lxml>=4.9.0"
  "lxml-html-clean>=0.1.0"
  "pillow>=10.0.0"
  "openai>=1.0.0"
  "google-genai>=1.0.0"
  "tqdm>=4.65.0"
  "colorama>=0.4.6"
  "python-dotenv>=1.0.0"
)

echo "[setup] Installing Python training deps..."
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet "${PY_DEPS[@]}"

# 5. Install the InSTA package (local, editable) if a setup.py is present
if [[ -f setup.py ]]; then
  echo "[setup] Installing local insta package (editable)..."
  python3 -m pip install --quiet -e .
fi

# 6. Smoke checks
echo ""
echo "===================================================================="
echo "Smoke checks"
echo "===================================================================="
python3 - <<'PY'
import sys
print(f"  python    : {sys.version.split()[0]}")
import torch
print(f"  torch     : {torch.__version__}  cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device    : {torch.cuda.get_device_name(0)}  ({torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GiB)")
import transformers, peft, bitsandbytes
print(f"  transformers: {transformers.__version__}")
print(f"  peft        : {peft.__version__}")
print(f"  bitsandbytes: {bitsandbytes.__version__}")
PY

echo ""
echo "Done. Next steps:"
echo "  export JUDGE_API_KEY=<your Gemini API key>"
echo "  bash gcp/launch_training.sh"
