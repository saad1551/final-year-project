# SB3 PPO Integration - Usage Guide

## ✅ Integration Complete!

SB3 PPO has been successfully integrated into your pipeline as the `sb3_ppo` algorithm option.

## Quick Start

### 1. Install Dependencies (if not already installed)

```bash
pip install stable-baselines3 gymnasium
```

### 2. Run with SB3 PPO

```bash
# Basic usage with default preset
python pipeline_in_steps.py --algorithm sb3_ppo --num_trajectories 1

# With low memory preset (for limited GPU)
python pipeline_in_steps.py --algorithm sb3_ppo --sb3_preset low_memory --num_trajectories 1

# With aggressive preset (fast learning)
python pipeline_in_steps.py --algorithm sb3_ppo --sb3_preset aggressive --num_trajectories 1 --learning_rate 1e-3

# With conservative preset (stable training)
python pipeline_in_steps.py --algorithm sb3_ppo --sb3_preset conservative --num_trajectories 5

# With exploration preset (high entropy)
python pipeline_in_steps.py --algorithm sb3_ppo --sb3_preset exploration --num_trajectories 1
```

## Available Algorithms

You can now choose from **4 algorithms**:

```bash
--algorithm reinforce      # Custom REINFORCE implementation
--algorithm ppo            # Custom PPO implementation  
--algorithm grpo           # Custom GRPO implementation
--algorithm sb3_ppo        # Stable Baselines3 PPO ⭐ NEW!
```

## SB3 PPO Configuration Presets

When using `--algorithm sb3_ppo`, you can specify a preset with `--sb3_preset`:

| Preset | Description | Use Case |
|--------|-------------|----------|
| `default` | Standard PPO settings | General purpose training |
| `low_memory` | Reduced batch sizes (n_steps=512, batch=32) | Limited GPU memory (8GB) |
| `aggressive` | Fast learning (lr=1e-3, batch=128) | Quick experimentation |
| `conservative` | Stable learning (lr=1e-4, tight clip) | Production/stable training |
| `exploration` | High entropy (ent_coef=0.05, SDE) | Exploration-focused |

## Examples

### Example 1: Training with Low Memory

```bash
python pipeline_in_steps.py \
    --algorithm sb3_ppo \
    --sb3_preset low_memory \
    --num_trajectories 10 \
    --save_every 5 \
    --checkpoint_dir checkpoints_sb3
```

### Example 2: Fast Experimentation

```bash
python pipeline_in_steps.py \
    --algorithm sb3_ppo \
    --sb3_preset aggressive \
    --num_trajectories 5 \
    --learning_rate 1e-3
```

### Example 3: Stable Production Training

```bash
python pipeline_in_steps.py \
    --algorithm sb3_ppo \
    --sb3_preset conservative \
    --num_trajectories 20 \
    --save_every 5 \
    --checkpoint_dir checkpoints_production
```

### Example 4: Compare Algorithms

```bash
# Custom PPO
python pipeline_in_steps.py --algorithm ppo --num_trajectories 5

# SB3 PPO with same learning rate
python pipeline_in_steps.py --algorithm sb3_ppo --num_trajectories 5

# Compare results!
```

## Parameters

### SB3 PPO Specific

- `--sb3_preset`: Configuration preset (default, low_memory, aggressive, conservative, exploration)

### Common Parameters (work with all algorithms)

- `--algorithm`: Algorithm choice (reinforce, ppo, grpo, sb3_ppo)
- `--num_trajectories`: Number of trajectories to run
- `--learning_rate`: Learning rate for optimizer
- `--checkpoint_dir`: Directory for saving checkpoints
- `--save_every`: Save checkpoint every N trajectories
- `--debug`: Enable debug logging

### Custom PPO Parameters (ignored by sb3_ppo)

- `--ppo_epochs`: Number of PPO epochs
- `--ppo_clip_epsilon`: PPO clipping epsilon

### GRPO Parameters (ignored by sb3_ppo)

- `--grpo_group_size`: GRPO group size
- `--grpo_beta`: GRPO beta coefficient

## What Changed

The integration added:

1. **New imports**: `SB3PPOTrainer` and `get_sb3_config`
2. **New algorithm option**: `sb3_ppo` in `--algorithm` choices
3. **New preset argument**: `--sb3_preset` for configuration
4. **Smart trainer creation**: Automatically uses SB3 or custom trainer based on algorithm
5. **Backward compatibility**: All existing custom algorithms work exactly as before

## Verification

To verify the integration works:

```bash
# Check help message shows sb3_ppo
python pipeline_in_steps.py --help

# Look for:
# --algorithm {reinforce,ppo,grpo,sb3_ppo}
# --sb3_preset {default,low_memory,aggressive,conservative,exploration}
```

## Notes

- **SB3 PPO uses different hyperparameters** than custom PPO (configured via presets)
- **Custom PPO parameters** (`--ppo_epochs`, `--ppo_clip_epsilon`) are ignored when using `sb3_ppo`
- **Learning rate** can be overridden for any preset via `--learning_rate`
- **Checkpoints** are saved the same way for all algorithms

## Troubleshooting

### Import Error

```
ModuleNotFoundError: No module named 'stable_baselines3'
```

**Fix:**:
```bash
pip install stable-baselines3 gymnasium
```

### CUDA Out of Memory

**Fix:** Use the low_memory preset:
```bash
python pipeline_in_steps.py --algorithm sb3_ppo --sb3_preset low_memory
```

### Want to see debug output

**Fix:** Add `--debug` flag:
```bash
python pipeline_in_steps.py --algorithm sb3_ppo --debug
```

## Next Steps

1. **Install dependencies**: `pip install stable-baselines3 gymnasium`
2. **Try it out**: `python pipeline_in_steps.py --algorithm sb3_ppo --num_trajectories 1`
3. **Compare with custom PPO**: Run both and compare results
4. **Experiment with presets**: Try different presets for your use case

Enjoy using SB3 PPO! 🚀
