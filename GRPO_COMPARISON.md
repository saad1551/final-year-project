# TRL GRPO vs Custom GRPO Implementation Comparison

This document compares the TRL-based GRPO implementation (`rl_trl_grpo.py`) with the custom GRPO implementation in `rl_trainer.py`.

## Quick Summary

| Feature | Custom GRPO (`rl_trainer.py`) | TRL GRPO (`rl_trl_grpo.py`) |
|---------|-------------------------------|------------------------------|
| **Implementation** | Custom from scratch | Hugging Face TRL library |
| **Maturity** | Project-specific | Battle-tested, used in DeepSeekMath |
| **Memory Efficiency** | Moderate | High (no separate value network) |
| **True GRPO** | Approximation (single trajectory) | True (multiple generations per prompt) |
| **Dependencies** | PyTorch only | TRL + datasets |
| **Learning Curve** | Custom to understand | Standard TRL API |

## Detailed Comparison

### 1. Algorithm Implementation

#### Custom GRPO
```python
# Approximates GRPO by using step-level comparisons within a trajectory
# - Uses reward history for group comparison
# - Single response per prompt
# - Reference log probs stored manually
```

**Key characteristics:**
- Adapts GRPO for single-response trajectories
- Uses reward history as pseudo-group
- Implements group advantages via softmax normalization
- Manual KL divergence computation

#### TRL GRPO
```python
# True GRPO implementation
# - Generates multiple completions per prompt
# - Group-relative advantage estimation
# - Built-in KL divergence handling
```

**Key characteristics:**
- Generates `num_generations` completions per prompt (default: 4)
- True group-based comparison within each prompt
- Optimized for LLM fine-tuning
- Integrated with HuggingFace ecosystem

### 2. Interface Comparison

Both implementations follow the **same interface pattern**:

```python
# Initialization
trainer = Trainer(model, tokenizer, config)

# Training
metrics = trainer.update_policy(
    trajectory_prompts,
    trajectory_responses,
    judgment
)

# Checkpointing
trainer.save_checkpoint(path)
trainer.load_checkpoint(path)
```

This means you can **swap between implementations** without changing your main code!

### 3. Configuration

#### Custom GRPO Config
```python
class RLConfig:
    grpo_group_size: int = 4
    grpo_temperature: float = 1.0
    grpo_beta: float = 0.1
    # ... plus general RL params
```

#### TRL GRPO Config
```python
class TRLGRPOConfig:
    num_generations: int = 4  # Similar to group_size
    temperature: float = 1.0
    beta: float = 0.1
    max_new_tokens: int = 512
    # ... plus TRL-specific params
```

### 4. Key Differences

#### A. Generation Strategy

**Custom GRPO:**
- Uses pre-collected responses (one per prompt)
- Never generates new completions
- Relies on external data collection

**TRL GRPO:**
- Generates multiple completions internally
- Samples `num_generations` responses per prompt
- Self-contained generation loop

#### B. Advantage Computation

**Custom GRPO:**
```python
def compute_group_advantages(rewards, temperature):
    # Softmax normalization within reward history
    normalized_rewards = F.softmax(rewards / temperature, dim=0)
    advantages = normalized_rewards - normalized_rewards.mean()
    return advantages
```

**TRL GRPO:**
```python
# Internal implementation (simplified):
# For each prompt, generate N completions
# Compute rewards for all completions
# Normalize within the group
# Use as advantages for policy update
```

#### C. Memory Usage

**Custom GRPO:**
- Stores reference log probs
- Maintains reward history
- Standard gradient computation

**TRL GRPO:**
- No separate value network (memory saving!)
- Uses group-based baseline (no explicit baseline storage)
- Optimized for large models

#### D. Training Loop

**Custom GRPO:**
```python
# One trajectory = one update
# Uses step-level rewards
# Manual optimization step
```

**TRL GRPO:**
```python
# Batch of prompts -> multiple generations
# Computes group advantages automatically
# Uses HuggingFace Trainer internals
```

### 5. When to Use Which?

#### Use Custom GRPO when:
- ✅ You have pre-collected trajectory data
- ✅ You want full control over the algorithm
- ✅ You're working with single responses per prompt
- ✅ You want to minimize dependencies
- ✅ You need tight integration with existing custom code

#### Use TRL GRPO when:
- ✅ You want the official implementation (used in DeepSeekMath)
- ✅ You need true multi-generation comparison
- ✅ Memory efficiency is critical
- ✅ You're fine-tuning LLMs
- ✅ You want HuggingFace ecosystem integration
- ✅ You want battle-tested, maintained code

### 6. Performance Considerations

#### Custom GRPO
**Pros:**
- Lightweight, minimal dependencies
- Direct control over training loop
- Fast for single-response scenarios

**Cons:**
- Approximates GRPO (not true group comparison)
- May have higher variance
- Less memory efficient than TRL

#### TRL GRPO
**Pros:**
- True GRPO with multiple generations
- Lower variance (better group comparison)
- Memory efficient (no value network)
- Optimized implementation

**Cons:**
- Generates multiple responses (slower per update)
- More dependencies (TRL, datasets)
- Less direct control

### 7. Code Example: Switching Between Implementations

```python
# Your main training code can work with BOTH!

# Option 1: Use Custom GRPO
from rl_trainer import GRPOAlgorithm, RLConfig

config = RLConfig(algorithm="grpo")
trainer = GRPOAlgorithm(model, tokenizer, config)

# Option 2: Use TRL GRPO
from rl_trl_grpo import TRLGRPOTrainer, TRLGRPOConfig

config = TRLGRPOConfig()
trainer = TRLGRPOTrainer(model, tokenizer, config)

# Rest of the code is IDENTICAL!
metrics = trainer.update_policy(prompts, responses, judgment)
trainer.save_checkpoint("./checkpoint")
```

### 8. Recommendation

**For your browser navigation project:**

1. **Start with Custom GRPO** if:
   - You're still experimenting
   - You have limited compute
   - You want to understand every detail

2. **Switch to TRL GRPO** when:
   - You're ready for production training
   - You want better sample efficiency
   - You need the quality gains from true GRPO

3. **Use both in parallel** to:
   - Compare performance
   - Validate your custom implementation
   - Choose the best for your specific task

### 9. Integration with Pipeline

Both can be integrated into `pipeline_in_steps.py` with the same pattern:

```python
# In pipeline_in_steps.py

if args.algorithm == "grpo":
    from rl_trainer import GRPOAlgorithm, RLConfig
    config = RLConfig(algorithm="grpo")
    trainer = GRPOAlgorithm(model, tokenizer, config)

elif args.algorithm == "trl_grpo":
    from rl_trl_grpo import TRLGRPOTrainer, TRLGRPOConfig
    config = TRLGRPOConfig()
    trainer = TRLGRPOTrainer(model, tokenizer, config)

# Then use trainer.update_policy() as usual
```

### 10. Summary Table

| Aspect | Custom GRPO | TRL GRPO |
|--------|-------------|----------|
| **Algorithm Fidelity** | Approximation | True GRPO |
| **Generations/Prompt** | 1 | 4+ (configurable) |
| **Dependencies** | Low | Medium |
| **Memory Usage** | Medium | Low |
| **Training Speed** | Fast (1 gen) | Slower (N gens) |
| **Sample Efficiency** | Lower | Higher |
| **Maintenance** | You | HuggingFace |
| **Customization** | High | Medium |
| **Production Ready** | Yes (for your use case) | Yes (industry standard) |

## Conclusion

Both implementations have the **same interface**, so you can:
1. Start with Custom GRPO for quick iterations
2. Switch to TRL GRPO for final training runs
3. Compare results between both
4. Choose based on your specific needs

The beauty of maintaining interface compatibility is that **switching is just a few lines of code**!
