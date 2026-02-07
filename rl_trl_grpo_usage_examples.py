"""
Usage examples for TRL GRPO implementation.

This file demonstrates how to use the TRL GRPO trainer
with the same interface as other RL implementations.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
from rl_trl_grpo import TRLGRPOTrainer, TRLGRPOConfig, create_trl_grpo_trainer, get_grpo_preset
from insta.configs.judge_config import BrowserJudgment


# ============================================================================
# EXAMPLE 1: Basic Usage with Default Config
# ============================================================================

def example_basic_usage():
    """
    Basic example showing how to initialize and use TRL GRPO trainer.
    This follows the same interface as other RL implementations.
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Usage")
    print("="*80 + "\n")
    
    # Load model and tokenizer (example with small model)
    model_name = "Qwen/Qwen2-0.5B-Instruct"
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Create trainer with default config
    config = TRLGRPOConfig()
    trainer = TRLGRPOTrainer(model, tokenizer, config)
    
    # Simulate a trajectory
    trajectory_prompts = [
        "Navigate to the login page",
        "Enter username and password",
        "Click the login button",
    ]
    
    trajectory_responses = [
        "I'll click on the 'Login' link in the navigation bar.",
        "I'll enter 'user@example.com' in the username field and 'password123' in the password field.",
        "I'll click the blue 'Sign In' button.",
    ]
    
    # Simulate judgment from browser evaluation
    judgment = BrowserJudgment(
        success=0.8,
        efficiency=0.7,
        self_correction=0.6,
        explanation="Successfully logged in with minor inefficiencies"
    )
    
    # Update policy - same interface as other RL algorithms!
    print("Updating policy...")
    metrics = trainer.update_policy(
        trajectory_prompts,
        trajectory_responses,
        judgment
    )
    
    print("\nTraining Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


# ============================================================================
# EXAMPLE 2: Using Preset Configurations
# ============================================================================

def example_with_presets():
    """
    Example showing how to use preset configurations.
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Using Presets")
    print("="*80 + "\n")
    
    model_name = "Qwen/Qwen2-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Try different presets
    presets = ["fast", "default", "quality", "memory_efficient"]
    
    for preset_name in presets:
        print(f"\nTesting preset: {preset_name}")
        config = get_grpo_preset(preset_name)
        
        trainer = create_trl_grpo_trainer(model, tokenizer, config)
        
        print(f"  - Num generations: {config.num_generations}")
        print(f"  - Temperature: {config.temperature}")
        print(f"  - Learning rate: {config.learning_rate}")


# ============================================================================
# EXAMPLE 3: Multiple Updates with Statistics
# ============================================================================

def example_multiple_updates():
    """
    Example showing multiple policy updates and tracking statistics.
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Multiple Updates")
    print("="*80 + "\n")
    
    model_name = "Qwen/Qwen2-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    config = TRLGRPOConfig(
        learning_rate=2e-5,
        num_generations=4,
    )
    
    trainer = TRLGRPOTrainer(model, tokenizer, config)
    
    # Simulate 3 episodes
    episodes = [
        {
            "prompts": ["Find the search box", "Enter query", "Click search"],
            "responses": ["Located search box", "Typed 'test query'", "Clicked button"],
            "judgment": BrowserJudgment(success=0.7, efficiency=0.6, self_correction=0.5),
        },
        {
            "prompts": ["Navigate to settings", "Change theme", "Save changes"],
            "responses": ["Opened settings", "Selected dark mode", "Clicked save"],
            "judgment": BrowserJudgment(success=0.9, efficiency=0.8, self_correction=0.7),
        },
        {
            "prompts": ["Open menu", "Select profile", "Edit info"],
            "responses": ["Clicked menu icon", "Went to profile", "Updated name"],
            "judgment": BrowserJudgment(success=0.85, efficiency=0.75, self_correction=0.65),
        },
    ]
    
    for i, episode in enumerate(episodes, 1):
        print(f"\nEpisode {i}:")
        metrics = trainer.update_policy(
            episode["prompts"],
            episode["responses"],
            episode["judgment"]
        )
        
        print(f"  Reward: {metrics['trajectory_reward']:.3f}")
        print(f"  Avg Reward: {metrics['avg_reward']:.3f}")
        print(f"  Total Updates: {metrics['total_updates']}")


# ============================================================================
# EXAMPLE 4: Checkpoint Saving and Loading
# ============================================================================

def example_checkpointing():
    """
    Example showing how to save and load checkpoints.
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Checkpointing")
    print("="*80 + "\n")
    
    model_name = "Qwen/Qwen2-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    config = TRLGRPOConfig()
    trainer = TRLGRPOTrainer(model, tokenizer, config)
    
    # Do some training
    print("Training...")
    trajectory_prompts = ["Step 1", "Step 2"]
    trajectory_responses = ["Response 1", "Response 2"]
    judgment = BrowserJudgment(success=0.8, efficiency=0.7, self_correction=0.6)
    
    trainer.update_policy(trajectory_prompts, trajectory_responses, judgment)
    
    # Save checkpoint
    checkpoint_path = "./checkpoints/grpo_test"
    print(f"\nSaving checkpoint to: {checkpoint_path}")
    trainer.save_checkpoint(checkpoint_path)
    
    # Load checkpoint (in a new trainer)
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    new_trainer = TRLGRPOTrainer(model, tokenizer, config)
    new_trainer.load_checkpoint(checkpoint_path)
    
    print("\nCheckpoint loaded successfully!")
    print(f"  Total updates: {new_trainer.training_stats['total_updates']}")
    print(f"  Avg reward: {new_trainer.training_stats['avg_reward']:.3f}")


# ============================================================================
# EXAMPLE 5: Custom Configuration
# ============================================================================

def example_custom_config():
    """
    Example showing how to create and use custom configurations.
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Custom Configuration")
    print("="*80 + "\n")
    
    model_name = "Qwen/Qwen2-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Create custom config optimized for your specific task
    custom_config = TRLGRPOConfig(
        learning_rate=1e-5,
        num_generations=6,  # More generations for better comparison
        temperature=0.85,  # Balanced exploration/exploitation
        max_new_tokens=400,
        beta=0.08,  # Custom KL penalty
        # Custom reward weights
        success_weight=0.6,  # Prioritize success
        efficiency_weight=0.25,
        self_correction_weight=0.15,
        # Training parameters
        per_device_train_batch_size=1,
        gradient_accumulation_steps=6,
    )
    
    trainer = TRLGRPOTrainer(model, tokenizer, custom_config)
    
    print("Custom configuration created:")
    print(f"  Learning rate: {custom_config.learning_rate}")
    print(f"  Num generations: {custom_config.num_generations}")
    print(f"  Temperature: {custom_config.temperature}")
    print(f"  Reward weights: Success={custom_config.success_weight}, "
          f"Efficiency={custom_config.efficiency_weight}, "
          f"Self-correction={custom_config.self_correction_weight}")


# ============================================================================
# EXAMPLE 6: Interface Compatibility Demo
# ============================================================================

def example_interface_compatibility():
    """
    Demonstrate that TRL GRPO has the same interface as other RL algorithms.
    This means you can swap between algorithms without changing your code!
    """
    print("\n" + "="*80)
    print("EXAMPLE 6: Interface Compatibility")
    print("="*80 + "\n")
    
    model_name = "Qwen/Qwen2-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # All these have the same interface!
    print("Creating trainers with the same interface:")
    
    # TRL GRPO
    trl_grpo = create_trl_grpo_trainer(
        model, tokenizer, 
        TRLGRPOConfig()
    )
    print("✓ TRL GRPO trainer created")
    
    # Common interface methods:
    print("\nCommon interface methods available:")
    print("  - update_policy(prompts, responses, judgment)")
    print("  - save_checkpoint(path)")
    print("  - load_checkpoint(path)")
    
    print("\nYou can switch between algorithms by just changing the trainer initialization!")
    print("The rest of your code remains the same.")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("TRL GRPO USAGE EXAMPLES")
    print("="*80)
    
    # Note: These examples use a small model for demonstration
    # In practice, you'd use your actual trained model
    
    # Uncomment the examples you want to run:
    
    # example_basic_usage()
    # example_with_presets()
    # example_multiple_updates()
    # example_checkpointing()
    # example_custom_config()
    example_interface_compatibility()
    
    print("\n" + "="*80)
    print("Examples completed!")
    print("="*80 + "\n")
