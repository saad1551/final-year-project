"""
Integration Guide: Using SB3 PPO with the Existing Pipeline

This guide shows how to integrate the Stable Baselines3 PPO implementation
with your existing browser navigation RL pipeline.
"""

# ============================================================================
# Example 1: Using SB3 PPO as a Drop-in Replacement
# ============================================================================

def example_basic_replacement():
    """
    Replace custom PPO with SB3 PPO in the pipeline.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from experimental.rl_sb3_ppo import SB3PPOTrainer
    from experimental.rl_sb3_config_examples import get_config
    from insta.configs.judge_config import BrowserJudgment
    
    # Load your model
    model = AutoModelForCausalLM.from_pretrained("your-model-path")
    tokenizer = AutoTokenizer.from_pretrained("your-model-path")
    
    # Create SB3 PPO trainer (instead of OnPolicyTrainer)
    config = get_config("default")
    trainer = SB3PPOTrainer(model, tokenizer, config)
    
    # Use the exact same interface as custom algorithms
    trajectory_prompts = ["prompt1", "prompt2", "prompt3"]
    trajectory_responses = ["response1", "response2", "response3"]
    judgment = BrowserJudgment(
        success=0.8,
        efficiency=0.7,
        self_correction=0.6
    )
    
    # Update policy - same interface!
    result = trainer.update_policy(
        trajectory_prompts=trajectory_prompts,
        trajectory_responses=trajectory_responses,
        judgment=judgment
    )
    
    print(f"Training result: {result}")
    
    # Save checkpoint - same interface!
    trainer.save_checkpoint("./checkpoints/sb3_ppo_model")


# ============================================================================
# Example 2: Side-by-Side Comparison
# ============================================================================

def example_compare_algorithms():
    """
    Compare custom PPO with SB3 PPO on the same data.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from rl_trainer import OnPolicyTrainer, RLConfig
    from experimental.rl_sb3_ppo import SB3PPOTrainer, SB3PPOConfig
    from experimental.rl_sb3_config_examples import get_config
    from insta.configs.judge_config import BrowserJudgment
    
    # Load models
    model = AutoModelForCausalLM.from_pretrained("your-model-path")
    tokenizer = AutoTokenizer.from_pretrained("your-model-path")
    
    # Setup custom PPO
    custom_config = RLConfig(algorithm="ppo", learning_rate=3e-4)
    custom_trainer = OnPolicyTrainer(model, tokenizer, custom_config)
    
    # Setup SB3 PPO
    sb3_config = get_config("default")
    sb3_trainer = SB3PPOTrainer(model, tokenizer, sb3_config)
    
    # Prepare data
    trajectory_prompts = ["Navigate to page", "Click button"]
    trajectory_responses = ["I will navigate", "I will click"]
    judgment = BrowserJudgment(success=0.9, efficiency=0.8, self_correction=0.7)
    
    # Train with custom PPO
    custom_result = custom_trainer.update_policy(
        trajectory_prompts, trajectory_responses, judgment
    )
    print(f"Custom PPO result: {custom_result}")
    
    # Train with SB3 PPO
    sb3_result = sb3_trainer.update_policy(
        trajectory_prompts, trajectory_responses, judgment
    )
    print(f"SB3 PPO result: {sb3_result}")


# ============================================================================
# Example 3: Configurable Algorithm Selection
# ============================================================================

def example_configurable_selection(algorithm_type="sb3_ppo"):
    """
    Choose between custom and SB3 PPO based on configuration.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from rl_trainer import OnPolicyTrainer, RLConfig
    from experimental.rl_sb3_ppo import SB3PPOTrainer
    from experimental.rl_sb3_config_examples import get_config
    
    # Load models
    model = AutoModelForCausalLM.from_pretrained("your-model-path")
    tokenizer = AutoTokenizer.from_pretrained("your-model-path")
    
    # Create trainer based on algorithm type
    if algorithm_type == "sb3_ppo":
        config = get_config("default")
        trainer = SB3PPOTrainer(model, tokenizer, config)
        print("Using Stable Baselines3 PPO")
    
    elif algorithm_type in ["reinforce", "ppo", "grpo"]:
        config = RLConfig(algorithm=algorithm_type)
        trainer = OnPolicyTrainer(model, tokenizer, config)
        print(f"Using custom {algorithm_type.upper()}")
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_type}")
    
    return trainer


# ============================================================================
# Example 4: Pipeline Integration with Command-Line Args
# ============================================================================

def example_pipeline_integration():
    """
    Integration pattern for use in pipeline scripts.
    
    This shows how to add SB3 PPO as an option to your existing
    pipeline that already supports REINFORCE, PPO, and GRPO.
    """
    import argparse
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from rl_trainer import OnPolicyTrainer, RLConfig
    from experimental.rl_sb3_ppo import SB3PPOTrainer, SB3PPOConfig
    from experimental.rl_sb3_config_examples import get_config
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm",
        type=str,
        default="ppo",
        choices=["reinforce", "ppo", "grpo", "sb3_ppo"],
        help="RL algorithm to use"
    )
    parser.add_argument(
        "--sb3_preset",
        type=str,
        default="default",
        choices=["default", "low_memory", "aggressive", "conservative"],
        help="SB3 PPO configuration preset (only used with sb3_ppo)"
    )
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--model_path", type=str, required=True)
    
    args = parser.parse_args()
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Create trainer based on algorithm
    if args.algorithm == "sb3_ppo":
        # Use SB3 PPO
        config = get_config(args.sb3_preset)
        config.learning_rate = args.learning_rate  # Override if needed
        trainer = SB3PPOTrainer(model, tokenizer, config)
    else:
        # Use custom algorithms
        config = RLConfig(
            algorithm=args.algorithm,
            learning_rate=args.learning_rate
        )
        trainer = OnPolicyTrainer(model, tokenizer, config)
    
    return trainer


# ============================================================================
# Example 5: Using Different SB3 Configurations
# ============================================================================

def example_different_configs():
    """
    Examples of using different SB3 PPO configurations for different scenarios.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from experimental.rl_sb3_ppo import SB3PPOTrainer
    from experimental.rl_sb3_config_examples import get_config
    
    model = AutoModelForCausalLM.from_pretrained("your-model-path")
    tokenizer = AutoTokenizer.from_pretrained("your-model-path")
    
    # Scenario 1: Limited GPU memory
    print("Scenario 1: Limited GPU Memory")
    low_mem_config = get_config("low_memory")
    trainer_low_mem = SB3PPOTrainer(model, tokenizer, low_mem_config)
    
    # Scenario 2: Fast prototyping / experimentation
    print("Scenario 2: Fast Prototyping")
    aggressive_config = get_config("aggressive")
    trainer_aggressive = SB3PPOTrainer(model, tokenizer, aggressive_config)
    
    # Scenario 3: Production / stable training
    print("Scenario 3: Production Training")
    conservative_config = get_config("conservative")
    trainer_conservative = SB3PPOTrainer(model, tokenizer, conservative_config)
    
    # Scenario 4: Emphasize exploration
    print("Scenario 4: Exploration")
    exploration_config = get_config("exploration")
    trainer_exploration = SB3PPOTrainer(model, tokenizer, exploration_config)
    
    # Scenario 5: Focus on task success
    print("Scenario 5: Task Success")
    success_config = get_config("success_focused")
    trainer_success = SB3PPOTrainer(model, tokenizer, success_config)
    
    return {
        "low_memory": trainer_low_mem,
        "aggressive": trainer_aggressive,
        "conservative": trainer_conservative,
        "exploration": trainer_exploration,
        "success": trainer_success,
    }


# ============================================================================
# Example 6: Experiment Tracking
# ============================================================================

def example_experiment_tracking():
    """
    Track experiments comparing different algorithms and configurations.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from rl_trainer import OnPolicyTrainer, RLConfig
    from experimental.rl_sb3_ppo import SB3PPOTrainer
    from experimental.rl_sb3_config_examples import get_config
    from insta.configs.judge_config import BrowserJudgment
    import json
    
    # Setup
    model = AutoModelForCausalLM.from_pretrained("your-model-path")
    tokenizer = AutoTokenizer.from_pretrained("your-model-path")
    
    # Test data
    trajectory_prompts = ["prompt1", "prompt2"]
    trajectory_responses = ["response1", "response2"]
    judgment = BrowserJudgment(success=0.85, efficiency=0.75, self_correction=0.65)
    
    # Experiments to run
    experiments = {
        "custom_reinforce": (OnPolicyTrainer, RLConfig(algorithm="reinforce")),
        "custom_ppo": (OnPolicyTrainer, RLConfig(algorithm="ppo")),
        "custom_grpo": (OnPolicyTrainer, RLConfig(algorithm="grpo")),
        "sb3_ppo_default": (SB3PPOTrainer, get_config("default")),
        "sb3_ppo_aggressive": (SB3PPOTrainer, get_config("aggressive")),
    }
    
    # Run experiments
    results = {}
    for exp_name, (trainer_class, config) in experiments.items():
        print(f"\nRunning experiment: {exp_name}")
        
        trainer = trainer_class(model, tokenizer, config)
        result = trainer.update_policy(
            trajectory_prompts, trajectory_responses, judgment
        )
        
        results[exp_name] = result
        print(f"Result: {result}")
    
    # Save results
    with open("experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nExperiment results saved to experiment_results.json")
    return results


# ============================================================================
# Example 7: Integration with Existing rl_trainer Pattern
# ============================================================================

def example_factory_pattern():
    """
    Add SB3 PPO to a factory pattern similar to rl_trainer.py.
    """
    from transformers import PreTrainedModel, PreTrainedTokenizer
    from rl_trainer import OnPolicyTrainer, RLConfig
    from experimental.rl_sb3_ppo import SB3PPOTrainer, SB3PPOConfig
    from experimental.rl_sb3_config_examples import get_config
    
    def create_rl_trainer(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        algorithm: str,
        **kwargs
    ):
        """
        Factory function to create RL trainer.
        
        Args:
            model: Pre-trained model
            tokenizer: Tokenizer
            algorithm: Algorithm name ("reinforce", "ppo", "grpo", "sb3_ppo")
            **kwargs: Additional config parameters
        """
        if algorithm == "sb3_ppo":
            # Get preset if specified, otherwise use default
            preset = kwargs.pop("sb3_preset", "default")
            config = get_config(preset)
            
            # Override with kwargs
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            return SB3PPOTrainer(model, tokenizer, config)
        
        elif algorithm in ["reinforce", "ppo", "grpo"]:
            config = RLConfig(algorithm=algorithm, **kwargs)
            return OnPolicyTrainer(model, tokenizer, config)
        
        else:
            raise ValueError(
                f"Unknown algorithm: {algorithm}. "
                f"Choose from: reinforce, ppo, grpo, sb3_ppo"
            )
    
    return create_rl_trainer


# ============================================================================
# Main: Run Examples
# ============================================================================

if __name__ == "__main__":
    print("SB3 PPO Integration Examples")
    print("=" * 70)
    
    print("\nThese are code examples showing how to integrate SB3 PPO")
    print("with your existing pipeline. To run them, uncomment the")
    print("example you want to try and provide the necessary model paths.\n")
    
    # Uncomment to run examples:
    # example_basic_replacement()
    # example_compare_algorithms()
    # trainer = example_configurable_selection("sb3_ppo")
    # example_pipeline_integration()
    # trainers = example_different_configs()
    # results = example_experiment_tracking()
    # factory = example_factory_pattern()
    
    print("Examples loaded. Check the code for usage patterns.")
