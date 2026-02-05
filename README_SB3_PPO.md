# Stable Baselines3 PPO Implementation

This directory contains an implementation of PPO (Proximal Policy Optimization) using the [Stable Baselines3](https://stable-baselines3.readthedocs.io/) library, designed to work with the browser navigation RL pipeline.

## Overview

The SB3 PPO implementation provides an alternative to the custom PPO implementation in `rl_trainer.py`. It leverages the well-tested and optimized PPO algorithm from Stable Baselines3 while maintaining compatibility with the existing pipeline.

## Files

- **`rl_sb3_ppo.py`**: Main implementation file containing:
  - `SB3PPOConfig`: Configuration dataclass for PPO hyperparameters
  - `BrowserNavigationEnv`: Gymnasium environment wrapper
  - `SB3PPOTrainer`: Main trainer class with interface compatible with custom RL algorithms
  - `CustomLMFeatureExtractor`: Feature extractor for language models
  - `TrainingCallback`: Custom callback for monitoring training

- **`rl_sb3_config_examples.py`**: Configuration presets including:
  - `default`: Standard PPO configuration
  - `low_memory`: Memory-efficient settings for limited GPU memory
  - `aggressive`: Fast learning with higher learning rate
  - `conservative`: Stable, slower learning with smaller updates
  - `exploration`: Exploration-focused with high entropy
  - `success_focused`: Emphasizes task success
  - `efficiency_focused`: Emphasizes efficient completion

- **`test_sb3_ppo.py`**: Comprehensive test suite for the implementation

- **`README_SB3_PPO.md`**: This file

## Installation

Install the required dependencies:

```bash
pip install stable-baselines3
pip install gymnasium
```

Optional dependencies for advanced features:

```bash
# For tensorboard logging
pip install tensorboard

# For video recording and rendering
pip install opencv-python
```

## Quick Start

### Basic Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from rl_sb3_ppo import SB3PPOTrainer, SB3PPOConfig
from rl_sb3_config_examples import get_config
from insta.configs.judge_config import BrowserJudgment

# Load your model and tokenizer
model = AutoModelForCausalLM.from_pretrained("your-model")
tokenizer = AutoTokenizer.from_pretrained("your-model")

# Use a preset configuration
config = get_config("default")  # or "low_memory", "aggressive", etc.

# Create trainer
trainer = SB3PPOTrainer(
    model=model,
    tokenizer=tokenizer,
    config=config
)

# Update policy with trajectory data
trajectory_prompts = ["Navigate to example.com", "Click login"]
trajectory_responses = ["I will navigate...", "I will click..."]
judgment = BrowserJudgment(success=0.8, efficiency=0.7, self_correction=0.6)

result = trainer.update_policy(
    trajectory_prompts=trajectory_prompts,
    trajectory_responses=trajectory_responses,
    judgment=judgment
)

print(f"Reward: {result['trajectory_reward']:.4f}")
```

### Custom Configuration

```python
from rl_sb3_ppo import SB3PPOConfig
from rl_sb3_config_examples import get_custom_config

# Option 1: Create from scratch
config = SB3PPOConfig(
    learning_rate=1e-3,
    n_steps=2048,
    batch_size=128,
    clip_range=0.2,
    ent_coef=0.01,
    success_weight=0.7,
    efficiency_weight=0.2,
    self_correction_weight=0.1
)

# Option 2: Use custom config helper
config = get_custom_config(
    learning_rate=1e-3,
    batch_size=128,
    success_weight=0.7
)
```

### Running Tests

```bash
# Run all tests
python test_sb3_ppo.py

# Or import and run specific tests
python -c "from test_sb3_ppo import test_sb3_ppo_basic; test_sb3_ppo_basic()"
```

## Configuration Options

### Learning Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 3e-4 | Learning rate for the optimizer |
| `n_steps` | 2048 | Steps to collect per environment per update |
| `batch_size` | 64 | Minibatch size for updates |
| `n_epochs` | 10 | Number of epochs per update |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE (Generalized Advantage Estimation) parameter |

### PPO-Specific Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `clip_range` | 0.2 | Clipping parameter for PPO objective |
| `clip_range_vf` | None | Clipping parameter for value function |
| `normalize_advantage` | True | Whether to normalize advantages |
| `ent_coef` | 0.01 | Entropy coefficient |
| `vf_coef` | 0.5 | Value function coefficient |
| `max_grad_norm` | 0.5 | Maximum gradient norm for clipping |
| `target_kl` | None | Early stopping KL divergence threshold |

### Reward Weights

| Parameter | Default | Description |
|-----------|---------|-------------|
| `success_weight` | 0.5 | Weight for task success score |
| `efficiency_weight` | 0.3 | Weight for efficiency score |
| `self_correction_weight` | 0.2 | Weight for self-correction score |

## Configuration Presets

### Default Configuration
Standard settings recommended for most tasks.

```python
config = get_config("default")
```

### Low Memory Configuration
Reduced batch sizes and steps for limited GPU memory.

```python
config = get_config("low_memory")
```

### Aggressive Configuration
Faster learning with higher learning rate and more updates.

```python
config = get_config("aggressive")
```

### Conservative Configuration
Stable learning with smaller updates and early stopping.

```python
config = get_config("conservative")
```

### Exploration Configuration
Encourages exploration with high entropy and SDE.

```python
config = get_config("exploration")
```

### Success/Efficiency Focused
Emphasizes specific reward components.

```python
config = get_config("success_focused")
config = get_config("efficiency_focused")
```

## API Reference

### SB3PPOTrainer

Main trainer class for SB3 PPO.

#### Methods

##### `__init__(model, tokenizer, config)`
Initialize the trainer.

**Parameters:**
- `model`: PreTrainedModel - The language model
- `tokenizer`: PreTrainedTokenizer - The tokenizer
- `config`: SB3PPOConfig - Configuration object

##### `update_policy(trajectory_prompts, trajectory_responses, judgment)`
Update policy using trajectory data.

**Parameters:**
- `trajectory_prompts`: List[str] - List of prompts
- `trajectory_responses`: List[str] - List of responses
- `judgment`: BrowserJudgment - Judgment with scores

**Returns:**
- Dict with training metrics

##### `train(total_timesteps, callback, log_interval)`
Train the PPO agent for specified timesteps.

**Parameters:**
- `total_timesteps`: int - Total timesteps to train
- `callback`: Optional callback for monitoring
- `log_interval`: int - Logging interval

##### `predict(observation, deterministic)`
Predict action for given observation.

**Parameters:**
- `observation`: Current observation
- `deterministic`: bool - Use deterministic policy

**Returns:**
- action, state tuple

##### `save_checkpoint(path)`
Save model checkpoint.

**Parameters:**
- `path`: str - Directory to save checkpoint

##### `load_checkpoint(path)`
Load model checkpoint.

**Parameters:**
- `path`: str - Directory to load checkpoint from

## Comparison with Custom PPO

### SB3 PPO Advantages
- ✓ Well-tested and optimized implementation
- ✓ Extensive documentation and community support
- ✓ Built-in features like tensorboard logging
- ✓ Advanced features (SDE, multiple environments, etc.)
- ✓ Regular updates and bug fixes

### Custom PPO Advantages
- ✓ Direct control over implementation details
- ✓ Easier to customize for specific use cases
- ✓ Better integration with existing codebase
- ✓ Fewer dependencies

## Integration with Existing Pipeline

The SB3 PPO implementation is designed to be a drop-in replacement for the custom algorithms. It provides the same `update_policy` interface and similar configuration structure.

### Using with rl_trainer.py Pattern

You can use SB3 PPO alongside the custom algorithms:

```python
from rl_trainer import RLConfig, OnPolicyTrainer
from rl_sb3_ppo import SB3PPOTrainer, SB3PPOConfig
from rl_sb3_config_examples import get_config

# Custom algorithm
custom_config = RLConfig(algorithm="ppo")
custom_trainer = OnPolicyTrainer(model, tokenizer, custom_config)

# SB3 algorithm
sb3_config = get_config("default")
sb3_trainer = SB3PPOTrainer(model, tokenizer, sb3_config)

# Both use the same interface
result1 = custom_trainer.update_policy(prompts, responses, judgment)
result2 = sb3_trainer.update_policy(prompts, responses, judgment)
```

## Troubleshooting

### Import Error: stable-baselines3 not found

```bash
pip install stable-baselines3
```

### CUDA Out of Memory

Use the `low_memory` configuration preset:

```python
config = get_config("low_memory")
```

Or reduce parameters manually:

```python
config = SB3PPOConfig(
    n_steps=512,
    batch_size=32,
    n_epochs=5
)
```

### Slow Training

Use the `aggressive` configuration:

```python
config = get_config("aggressive")
```

### Unstable Training

Use the `conservative` configuration:

```python
config = get_config("conservative")
```

## Examples

See `test_sb3_ppo.py` for comprehensive usage examples.

## References

- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [Gymnasium Documentation](https://gymnasium.farama.org/)

## License

Same as the parent project.
