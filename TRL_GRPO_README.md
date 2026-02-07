# TRL GRPO Implementation

This directory contains the TRL (Transformers Reinforcement Learning) based implementation of Group Relative Policy Optimization (GRPO) for the browser navigation agent.

## Files

- **`rl_trl_grpo.py`**: Main implementation with `TRLGRPOTrainer` class
- **`rl_trl_grpo_config_examples.py`**: Configuration presets and examples
- **`rl_trl_grpo_usage_examples.py`**: Complete usage examples
- **`GRPO_COMPARISON.md`**: Comparison with custom GRPO implementation

## Quick Start

### 1. Installation

```bash
pip install trl>=0.8.0 datasets>=2.14.0
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

### 2. Basic Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from rl_trl_grpo import TRLGRPOTrainer, TRLGRPOConfig
from insta.configs.judge_config import BrowserJudgment

# Load model
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

# Create trainer
config = TRLGRPOConfig()
trainer = TRLGRPOTrainer(model, tokenizer, config)

# Train on a trajectory
trajectory_prompts = ["Navigate to login", "Enter credentials", "Submit"]
trajectory_responses = ["Clicked login link", "Filled form", "Pressed button"]
judgment = BrowserJudgment(success=0.8, efficiency=0.7, self_correction=0.6)

metrics = trainer.update_policy(trajectory_prompts, trajectory_responses, judgment)
print(f"Reward: {metrics['trajectory_reward']:.3f}")
```

### 3. Using Presets

```python
from rl_trl_grpo import get_grpo_preset

# Fast training (for experimentation)
fast_config = get_grpo_preset("fast")

# High quality (for final training)
quality_config = get_grpo_preset("quality")

# Memory efficient (for limited GPU)
efficient_config = get_grpo_preset("memory_efficient")

trainer = TRLGRPOTrainer(model, tokenizer, quality_config)
```

## Configuration Presets

| Preset | Generations | Batch Size | Best For |
|--------|-------------|------------|----------|
| **default** | 4 | 1 | General use |
| **fast** | 2 | 2 | Quick experiments |
| **quality** | 8 | 1 | Best results |
| **memory_efficient** | 4 | 1 | Limited GPU memory |

## Key Features

### ✅ Same Interface as Other RL Algorithms
The TRL GRPO implementation has the **exact same interface** as other RL algorithms (REINFORCE, PPO, custom GRPO), making it easy to switch between them:

```python
# All of these work the same way!
from rl_trainer import REINFORCEAlgorithm, PPOAlgorithm, GRPOAlgorithm
from rl_sb3_ppo import SB3PPOTrainer
from rl_trl_grpo import TRLGRPOTrainer

# Choose one:
trainer = REINFORCEAlgorithm(model, tokenizer, config)
# OR
trainer = TRLGRPOTrainer(model, tokenizer, config)

# Then use the same interface:
metrics = trainer.update_policy(prompts, responses, judgment)
```

### ✅ True GRPO Implementation
Unlike the custom GRPO which approximates group comparison using reward history, TRL GRPO:
- Generates multiple completions per prompt
- Performs true group-relative comparison
- More sample efficient and stable

### ✅ Memory Efficient
- No separate value network (unlike PPO)
- Uses group-based baseline
- Gradient checkpointing support

### ✅ Battle-Tested
- Used in DeepSeekMath (51.7% on MATH benchmark)
- Maintained by Hugging Face
- Optimized for LLM fine-tuning

## Configuration Options

### Core Parameters

```python
config = TRLGRPOConfig(
    # Learning
    learning_rate=1e-5,
    num_train_epochs=1,
    
    # GRPO specific
    num_generations=4,      # Completions per prompt
    temperature=1.0,         # Sampling temperature
    max_new_tokens=512,      # Max completion length
    beta=0.1,                # KL penalty coefficient
    
    # Training
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=False,
    
    # Reward weights
    success_weight=0.5,
    efficiency_weight=0.3,
    self_correction_weight=0.2,
)
```

### Custom Configuration Example

```python
# Optimize for browser navigation
browser_config = TRLGRPOConfig(
    learning_rate=2e-5,
    num_generations=4,
    temperature=0.9,           # Higher for exploration
    max_new_tokens=512,
    beta=0.1,
    success_weight=0.4,
    efficiency_weight=0.3,
    self_correction_weight=0.3,
)
```

## Examples

### Example 1: Single Update

```python
trainer = TRLGRPOTrainer(model, tokenizer, config)

prompts = ["What is 2+2?", "What is 3*4?"]
responses = ["4", "12"]
judgment = BrowserJudgment(success=1.0, efficiency=0.9, self_correction=0.8)

metrics = trainer.update_policy(prompts, responses, judgment)
```

### Example 2: Training Loop

```python
for episode in episodes:
    metrics = trainer.update_policy(
        episode["prompts"],
        episode["responses"],
        episode["judgment"]
    )
    
    if metrics["avg_reward"] > best_reward:
        trainer.save_checkpoint(f"./checkpoints/best")
```

### Example 3: Checkpointing

```python
# Save
trainer.save_checkpoint("./checkpoints/episode_100")

# Load
new_trainer = TRLGRPOTrainer(model, tokenizer, config)
new_trainer.load_checkpoint("./checkpoints/episode_100")
```

## Comparison with Custom GRPO

| Feature | Custom GRPO | TRL GRPO |
|---------|-------------|----------|
| Implementation | Custom scratch | HuggingFace TRL |
| True GRPO | ❌ (approximation) | ✅ |
| Generations/prompt | 1 | 4+ |
| Memory usage | Medium | Low |
| Maintenance | Manual | Community |

See **`GRPO_COMPARISON.md`** for detailed comparison.

## When to Use TRL GRPO

Use TRL GRPO when you want:
- ✅ True GRPO with multiple generations
- ✅ Battle-tested, production-ready implementation
- ✅ Better sample efficiency
- ✅ Integration with HuggingFace ecosystem
- ✅ Memory-efficient training

## Integration with Pipeline

The TRL GRPO trainer can be easily integrated into your existing pipeline:

```python
# In pipeline_in_steps.py or similar

if args.algorithm == "trl_grpo":
    from rl_trl_grpo import TRLGRPOTrainer, get_grpo_preset
    
    config = get_grpo_preset(args.preset or "default")
    trainer = TRLGRPOTrainer(model, tokenizer, config)

# Use same interface as other algorithms
for episode in training_episodes:
    metrics = trainer.update_policy(
        episode.prompts,
        episode.responses,
        episode.judgment
    )
```

## Advanced Usage

### Custom Reward Function

The trainer uses `BrowserJudgment` by default, but you can extend the reward function:

```python
# Modify in rl_trl_grpo.py
def _create_reward_function(self):
    def reward_function(prompts, completions, **kwargs):
        # Your custom logic here
        return rewards
    return reward_function
```

### Hyperparameter Tuning

Key parameters to tune:
- `num_generations`: More = better comparison but slower
- `temperature`: Higher = more exploration
- `beta`: KL penalty (higher = stay closer to reference)
- `learning_rate`: Standard tuning

## Troubleshooting

### Out of Memory
```python
# Use memory_efficient preset
config = get_grpo_preset("memory_efficient")

# Or enable gradient checkpointing
config.gradient_checkpointing = True
config.num_generations = 2  # Reduce generations
```

### Slow Training
```python
# Use fast preset
config = get_grpo_preset("fast")

# Or reduce generations
config.num_generations = 2
config.max_new_tokens = 256
```

### Poor Performance
```python
# Use quality preset
config = get_grpo_preset("quality")

# Or increase generations
config.num_generations = 8
config.learning_rate = 1e-5
```

## References

- [DeepSeekMath Paper](https://arxiv.org/abs/2402.03300) - Original GRPO paper
- [TRL Documentation](https://huggingface.co/docs/trl/grpo_trainer) - Official TRL docs
- [HuggingFace TRL](https://github.com/huggingface/trl) - TRL repository

## Support

For issues or questions:
1. Check `rl_trl_grpo_usage_examples.py` for examples
2. See `GRPO_COMPARISON.md` for comparisons
3. Refer to TRL documentation for TRL-specific questions

## License

Follows the same license as the main project.
