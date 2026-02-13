# Installation Guide

This guide covers installing all dependencies for the RL Browser Navigation Training Pipeline.

## Quick Start

### Option 1: Using Conda (Recommended)

```bash
# Update your existing conda environment
conda env update -f environment.yml --prune

# Activate the environment
conda activate fyp-venv

# Install playwright browsers (one-time)
playwright install

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import transformers, peft, stable_baselines3; print('✓ All packages installed!')"
```

### Option 2: Using pip

```bash
# If you have CUDA 12.1 GPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install all other dependencies
pip install -r requirements.txt

# Install playwright browsers (one-time)
playwright install
```

## Checking Your CUDA Version

```bash
nvidia-smi
```

Look for the CUDA Version in the output. Common versions:
- CUDA 11.8: Use `pytorch-cuda=11.8`
- CUDA 12.1: Use `pytorch-cuda=12.1`

If you see "CUDA Version: 12.1", your GPU supports CUDA 12.1.

## For CPU-Only (No GPU)

Edit `environment.yml` and **remove** this line:
```yaml
  - pytorch-cuda=12.1
```

Then run:
```bash
conda env update -f environment.yml --prune
```

## Verifying Your Installation

Run this comprehensive check:

```bash
python - << 'EOF'
import sys
print("=" * 60)
print("DEPENDENCY CHECK")
print("=" * 60)

checks = [
    ("Python", lambda: sys.version.split()[0]),
    ("PyTorch", lambda: __import__('torch').__version__),
    ("CUDA Available", lambda: str(__import__('torch').cuda.is_available())),
    ("Transformers", lambda: __import__('transformers').__version__),
    ("PEFT (LoRA)", lambda: __import__('peft').__version__),
    ("BitsAndBytes", lambda: __import__('bitsandbytes').__version__),
    ("Stable Baselines3", lambda: __import__('stable_baselines3').__version__),
    ("Gymnasium", lambda: __import__('gymnasium').__version__),
    ("Pandas", lambda: __import__('pandas').__version__),
    ("OpenAI", lambda: __import__('openai').__version__),
    ("Playwright", lambda: __import__('playwright').__version__),
]

all_ok = True
for name, check_fn in checks:
    try:
        version = check_fn()
        print(f"✓ {name:20s} {version}")
    except Exception as e:
        print(f"✗ {name:20s} MISSING")
        all_ok = False

print("=" * 60)
if all_ok:
    print("✓ All dependencies installed successfully!")
else:
    print("✗ Some dependencies are missing. Install them first.")
print("=" * 60)
EOF
```

## Troubleshooting

### `bitsandbytes` fails on Windows
- Use WSL2 (Windows Subsystem for Linux) instead
- Or use the Windows-compatible version: `pip install bitsandbytes-windows`

### CUDA out of memory errors
Already optimized in the pipeline:
- 4-bit quantization
- Gradient checkpointing
- Micro-batching
- LoRA (trains only ~2% of parameters)

If still having issues:
1. Reduce `MAX_TRAJECTORY_STEPS` to 8 in `pipeline_in_steps.py`
2. Use the `low_memory` SB3 preset: `--sb3_preset low_memory`

### ImportError for `insta` modules
The project uses local modules. Make sure you're running from the project root:
```bash
cd /Users/saadashraf/fyp/final-year-project
python pipeline_in_steps.py --help
```

### Playwright browsers not found
```bash
# Install browsers
playwright install

# If that fails, try with sudo (Linux/Mac)
sudo playwright install
```

## Training Pipeline Usage

After installation, run your training pipeline:

```bash
# Default: SB3 PPO with aggressive preset, 5e-5 learning rate
python pipeline_in_steps.py \
    --num_trajectories 30 \
    --debug

# Custom configuration
python pipeline_in_steps.py \
    --algorithm sb3_ppo \
    --sb3_preset conservative \
    --learning_rate 3e-5 \
    --num_trajectories 50 \
    --save_every 10
```

## GPU Memory Recommendations

- **8GB VRAM**: Should work with default settings (10 trajectory steps, 4-bit quantization)
- **12GB+ VRAM**: Increase to `MAX_TRAJECTORY_STEPS = 15`
- **24GB+ VRAM**: Can use 8-bit quantization for better quality

Monitor GPU usage during training:
```bash
watch -n 1 nvidia-smi
```
