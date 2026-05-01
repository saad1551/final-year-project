"""
Example configurations for Stable Baselines3 PPO.

This file demonstrates different configuration presets for the SB3 PPO
implementation, similar to how the custom RL algorithms can be configured.
"""

from experimental.rl_sb3_ppo import SB3PPOConfig

# ============================================================================
# Standard Configurations
# ============================================================================

def get_default_config() -> SB3PPOConfig:
    """
    Default PPO configuration.
    
    This uses standard hyperparameters recommended by Stable Baselines3
    for most tasks.
    """
    return SB3PPOConfig(
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        success_weight=0.5,
        efficiency_weight=0.3,
        self_correction_weight=0.2,
        device="auto",
        verbose=1
    )


def get_low_memory_config() -> SB3PPOConfig:
    """
    Memory-efficient PPO configuration.
    
    Use this when training on systems with limited GPU memory.
    Reduces batch sizes and steps per update.
    """
    return SB3PPOConfig(
        learning_rate=3e-4,
        n_steps=512,  # Reduced from 2048
        batch_size=32,  # Reduced from 64
        n_epochs=5,  # Reduced from 10
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        normalize_advantage=True,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        success_weight=0.5,
        efficiency_weight=0.3,
        self_correction_weight=0.2,
        device="auto",
        verbose=1
    )


def get_aggressive_config() -> SB3PPOConfig:
    """
    Aggressive learning configuration.
    
    Use this for faster learning with more updates and higher learning rate.
    May be less stable but can converge faster.
    """
    return SB3PPOConfig(
        learning_rate=1e-3,  # Higher learning rate
        n_steps=2048,
        batch_size=128,  # Larger batch size
        n_epochs=15,  # More epochs
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.3,  # Wider clip range
        normalize_advantage=True,
        ent_coef=0.02,  # Higher entropy for more exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        success_weight=0.5,
        efficiency_weight=0.3,
        self_correction_weight=0.2,
        device="auto",
        verbose=1
    )


def get_conservative_config() -> SB3PPOConfig:
    """
    Conservative learning configuration.
    
    Use this for stable, slower learning with smaller updates.
    Good for fine-tuning or when stability is critical.
    """
    return SB3PPOConfig(
        learning_rate=1e-4,  # Lower learning rate
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,  # Tighter clip range
        clip_range_vf=0.1,  # Value function clipping
        normalize_advantage=True,
        ent_coef=0.005,  # Lower entropy
        vf_coef=0.5,
        max_grad_norm=0.3,  # Lower gradient clipping
        target_kl=0.01,  # Early stopping on KL divergence
        success_weight=0.5,
        efficiency_weight=0.3,
        self_correction_weight=0.2,
        device="auto",
        verbose=1
    )


def get_exploration_config() -> SB3PPOConfig:
    """
    Exploration-focused configuration.
    
    Use this when you want the agent to explore more, using higher
    entropy bonuses and State Dependent Exploration (SDE).
    """
    return SB3PPOConfig(
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        normalize_advantage=True,
        ent_coef=0.05,  # Much higher entropy
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=True,  # Enable State Dependent Exploration
        sde_sample_freq=4,  # Sample new noise every 4 steps
        success_weight=0.5,
        efficiency_weight=0.3,
        self_correction_weight=0.2,
        device="auto",
        verbose=1
    )


def get_success_focused_config() -> SB3PPOConfig:
    """
    Success-focused reward configuration.
    
    Emphasizes task success over efficiency and self-correction.
    """
    return SB3PPOConfig(
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        normalize_advantage=True,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        success_weight=0.8,  # Much higher weight on success
        efficiency_weight=0.1,
        self_correction_weight=0.1,
        device="auto",
        verbose=1
    )


def get_efficiency_focused_config() -> SB3PPOConfig:
    """
    Efficiency-focused reward configuration.
    
    Emphasizes efficient task completion over raw success.
    """
    return SB3PPOConfig(
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        normalize_advantage=True,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        success_weight=0.3,
        efficiency_weight=0.5,  # Higher weight on efficiency
        self_correction_weight=0.2,
        device="auto",
        verbose=1
    )


def get_custom_config(
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    success_weight: float = 0.5,
    efficiency_weight: float = 0.3,
    self_correction_weight: float = 0.2,
    **kwargs
) -> SB3PPOConfig:
    """
    Create a custom configuration with specific parameters.
    
    Args:
        learning_rate: Learning rate for the optimizer
        n_steps: Number of steps per update
        batch_size: Minibatch size
        clip_range: PPO clip range
        ent_coef: Entropy coefficient
        success_weight: Weight for success score
        efficiency_weight: Weight for efficiency score
        self_correction_weight: Weight for self-correction score
        **kwargs: Additional keyword arguments passed to SB3PPOConfig
        
    Returns:
        Custom SB3PPOConfig
    """
    return SB3PPOConfig(
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        clip_range=clip_range,
        ent_coef=ent_coef,
        success_weight=success_weight,
        efficiency_weight=efficiency_weight,
        self_correction_weight=self_correction_weight,
        **kwargs
    )


# ============================================================================
# Configuration Examples Dictionary
# ============================================================================

CONFIG_PRESETS = {
    "default": get_default_config,
    "low_memory": get_low_memory_config,
    "aggressive": get_aggressive_config,
    "conservative": get_conservative_config,
    "exploration": get_exploration_config,
    "success_focused": get_success_focused_config,
    "efficiency_focused": get_efficiency_focused_config,
}


def get_config(preset_name: str = "default") -> SB3PPOConfig:
    """
    Get a configuration preset by name.
    
    Args:
        preset_name: Name of the preset configuration
        
    Returns:
        SB3PPOConfig instance
        
    Raises:
        ValueError: If preset_name is not found
    """
    if preset_name not in CONFIG_PRESETS:
        available = ", ".join(CONFIG_PRESETS.keys())
        raise ValueError(
            f"Unknown config preset: {preset_name}. "
            f"Available presets: {available}"
        )
    
    return CONFIG_PRESETS[preset_name]()


# ============================================================================
# Usage Examples
# ============================================================================

if __name__ == "__main__":
    # Example 1: Using a preset configuration
    print("=== Example 1: Preset Configuration ===")
    default_config = get_config("default")
    print(f"Learning rate: {default_config.learning_rate}")
    print(f"Batch size: {default_config.batch_size}")
    print(f"Clip range: {default_config.clip_range}")
    print()
    
    # Example 2: Using low memory configuration
    print("=== Example 2: Low Memory Configuration ===")
    low_mem_config = get_config("low_memory")
    print(f"N steps: {low_mem_config.n_steps}")
    print(f"Batch size: {low_mem_config.batch_size}")
    print(f"N epochs: {low_mem_config.n_epochs}")
    print()
    
    # Example 3: Custom configuration
    print("=== Example 3: Custom Configuration ===")
    custom_config = get_custom_config(
        learning_rate=1e-3,
        batch_size=128,
        success_weight=0.7,
        efficiency_weight=0.2,
        self_correction_weight=0.1
    )
    print(f"Learning rate: {custom_config.learning_rate}")
    print(f"Success weight: {custom_config.success_weight}")
    print()
    
    # Example 4: Print all available presets
    print("=== Example 4: Available Presets ===")
    for preset_name in CONFIG_PRESETS.keys():
        print(f"- {preset_name}")
