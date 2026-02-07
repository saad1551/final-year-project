"""
Configuration examples for TRL GRPO implementation.

This file demonstrates how to configure and use the TRL GRPO trainer
with different presets and custom configurations.
"""

from rl_trl_grpo import TRLGRPOConfig, get_grpo_preset

# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================

def get_default_config():
    """Default configuration for general use."""
    return get_grpo_preset("default")


def get_fast_config():
    """
    Fast training configuration.
    - Fewer generations per prompt (2 instead of 4)
    - Larger batch size
    - Shorter completions
    
    Good for rapid experimentation and debugging.
    """
    return get_grpo_preset("fast")


def get_quality_config():
    """
    High quality configuration.
    - More generations per prompt (8 instead of 4)
    - Lower temperature for more focused sampling
    - Longer completions
    - More gradient accumulation
    
    Good for final training runs where quality matters.
    """
    return get_grpo_preset("quality")


def get_memory_efficient_config():
    """
    Memory efficient configuration.
    - Gradient checkpointing enabled
    - Smaller batch size
    - More gradient accumulation to compensate
    
    Good for training on GPUs with limited memory.
    """
    return get_grpo_preset("memory_efficient")


# ============================================================================
# CUSTOM CONFIGURATIONS
# ============================================================================

def get_custom_math_reasoning_config():
    """
    Custom configuration for math reasoning tasks.
    Similar to DeepSeekMath setup.
    """
    return TRLGRPOConfig(
        learning_rate=1e-5,
        num_generations=8,  # More samples for better comparison
        temperature=0.7,  # Lower temperature for focused reasoning
        max_new_tokens=1024,  # Longer for complex reasoning
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        beta=0.05,  # Lower KL penalty for more exploration
        success_weight=0.7,  # Prioritize success
        efficiency_weight=0.2,
        self_correction_weight=0.1,
    )


def get_custom_browser_navigation_config():
    """
    Custom configuration optimized for browser navigation.
    """
    return TRLGRPOConfig(
        learning_rate=2e-5,
        num_generations=4,
        temperature=0.9,  # Higher temperature for diverse actions
        max_new_tokens=512,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        beta=0.1,
        # Balance all three metrics equally
        success_weight=0.4,
        efficiency_weight=0.3,
        self_correction_weight=0.3,
    )


def get_custom_low_resource_config():
    """
    Configuration for training on limited resources (e.g., single GPU).
    """
    return TRLGRPOConfig(
        learning_rate=3e-5,
        num_generations=2,  # Minimum viable
        temperature=1.0,
        max_new_tokens=256,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        beta=0.1,
    )


def get_custom_exploration_config():
    """
    Configuration that encourages exploration.
    Higher temperature and lower KL penalty.
    """
    return TRLGRPOConfig(
        learning_rate=1e-5,
        num_generations=6,
        temperature=1.2,  # High temperature for exploration
        max_new_tokens=512,
        beta=0.01,  # Very low KL penalty to allow divergence
        per_device_train_batch_size=1,
        gradient_accumulation_steps=6,
    )


# ============================================================================
# CONFIGURATION HELPERS
# ============================================================================

def print_config(config: TRLGRPOConfig, name: str = "Config"):
    """Print configuration details in a readable format."""
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print(f"Learning Parameters:")
    print(f"  - Learning Rate: {config.learning_rate}")
    print(f"  - Epochs per Update: {config.num_train_epochs}")
    print(f"\nGRPO Parameters:")
    print(f"  - Generations per Prompt: {config.num_generations}")
    print(f"  - Temperature: {config.temperature}")
    print(f"  - Max New Tokens: {config.max_new_tokens}")
    print(f"  - Beta (KL Penalty): {config.beta}")
    print(f"\nBatch Parameters:")
    print(f"  - Batch Size: {config.per_device_train_batch_size}")
    print(f"  - Gradient Accumulation: {config.gradient_accumulation_steps}")
    print(f"  - Gradient Checkpointing: {config.gradient_checkpointing}")
    print(f"  - Effective Batch Size: {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
    print(f"\nReward Weights:")
    print(f"  - Success: {config.success_weight}")
    print(f"  - Efficiency: {config.efficiency_weight}")
    print(f"  - Self Correction: {config.self_correction_weight}")
    print(f"{'='*60}\n")


def compare_configs(config_dict: dict):
    """Compare multiple configurations side by side."""
    print("\n" + "="*80)
    print("CONFIGURATION COMPARISON")
    print("="*80)
    
    for name, config in config_dict.items():
        print(f"\n{name}:")
        print(f"  LR: {config.learning_rate}, Gens: {config.num_generations}, "
              f"Temp: {config.temperature}, Beta: {config.beta}")
        print(f"  Batch: {config.per_device_train_batch_size}, "
              f"GradAcc: {config.gradient_accumulation_steps}, "
              f"GradCkpt: {config.gradient_checkpointing}")
    
    print("="*80 + "\n")


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    # Example 1: Print all preset configurations
    print("\n" + "="*80)
    print("PRESET CONFIGURATIONS")
    print("="*80)
    
    presets = {
        "Default": get_default_config(),
        "Fast": get_fast_config(),
        "Quality": get_quality_config(),
        "Memory Efficient": get_memory_efficient_config(),
    }
    
    for name, config in presets.items():
        print_config(config, name)
    
    # Example 2: Print custom configurations
    print("\n" + "="*80)
    print("CUSTOM CONFIGURATIONS")
    print("="*80)
    
    custom_configs = {
        "Math Reasoning": get_custom_math_reasoning_config(),
        "Browser Navigation": get_custom_browser_navigation_config(),
        "Low Resource": get_custom_low_resource_config(),
        "Exploration": get_custom_exploration_config(),
    }
    
    for name, config in custom_configs.items():
        print_config(config, name)
    
    # Example 3: Compare configurations
    print("\n")
    compare_configs({
        "Fast": get_fast_config(),
        "Default": get_default_config(),
        "Quality": get_quality_config(),
    })
