"""
Test script for TRL GRPO implementation.

This script demonstrates how to use the TRL GRPO trainer with
mock data, similar to test_sb3_ppo.py for SB3 PPO.
"""

import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from insta.configs.judge_config import BrowserJudgment

# Import TRL GRPO components
try:
    from rl_trl_grpo import (
        TRLGRPOTrainer, 
        TRLGRPOConfig, 
        create_trl_grpo_trainer,
        get_grpo_preset,
        TRL_AVAILABLE
    )
except ImportError as e:
    print(f"Error importing TRL GRPO modules: {e}")
    print("Make sure TRL is installed: pip install trl>=0.8.0 datasets>=2.14.0")
    sys.exit(1)


def test_trl_grpo_basic():
    """Test basic TRL GRPO functionality with mock data."""
    print("=" * 70)
    print("Testing TRL GRPO - Basic Functionality")
    print("=" * 70)
    
    if not TRL_AVAILABLE:
        print("❌ TRL is not installed!")
        print("   Install with: pip install trl>=0.8.0 datasets>=2.14.0")
        return False
    
    print("\n✓ TRL is available")
    
    # Create a small mock model for testing
    print("\n1. Creating mock model and tokenizer...")
    try:
        # Use a tiny model for testing
        model_name = "gpt2"  # Small model for quick testing
        print(f"   Loading model: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32
        )
        
        print(f"   ✓ Model loaded: {model.config.model_type}")
        print(f"   ✓ Model size: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
        
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        return False
    
    # Create TRL GRPO configuration
    print("\n2. Creating TRL GRPO configuration...")
    try:
        # Use fast config for testing (fewer generations)
        config = get_grpo_preset("fast")
        config.verbose = 1
        config.num_generations = 2  # Very small for quick testing
        config.max_new_tokens = 32  # Short completions for testing
        config.num_train_epochs = 1
        config.logging_steps = 1
        config.output_dir = "./test_grpo_output"
        
        print(f"   ✓ Config: learning_rate={config.learning_rate}")
        print(f"   ✓ Config: num_generations={config.num_generations}")
        print(f"   ✓ Config: temperature={config.temperature}, beta={config.beta}")
        print(f"   ✓ Config: max_new_tokens={config.max_new_tokens}")
        
    except Exception as e:
        print(f"   ❌ Error creating config: {e}")
        return False
    
    # Create TRL GRPO trainer
    print("\n3. Initializing TRL GRPO trainer...")
    try:
        trainer = TRLGRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            config=config
        )
        print("   ✓ Trainer initialized successfully")
        
    except Exception as e:
        print(f"   ❌ Error initializing trainer: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Create mock trajectory data
    print("\n4. Creating mock trajectory data...")
    trajectory_prompts = [
        "Navigate to example.com",
        "Click on the login button",
        "Enter username: test@example.com",
    ]
    trajectory_responses = [
        "I will navigate to example.com",
        "I will click on the login button",
        "I will enter the username",
    ]
    
    # Create mock judgment
    judgment = BrowserJudgment(
        success=0.8,
        efficiency=0.7,
        self_correction=0.6,
        response="Task completed successfully with minor inefficiencies"
    )
    
    print(f"   ✓ Created trajectory with {len(trajectory_prompts)} steps")
    print(f"   ✓ Judgment scores: success={judgment.success}, "
          f"efficiency={judgment.efficiency}, "
          f"self_correction={judgment.self_correction}")
    
    # Test update_policy method
    print("\n5. Testing update_policy method...")
    print("   NOTE: This will generate completions and train, may take a moment...")
    try:
        result = trainer.update_policy(
            trajectory_prompts=trajectory_prompts,
            trajectory_responses=trajectory_responses,
            judgment=judgment
        )
        
        print("   ✓ Policy update completed")
        print(f"   Result keys: {list(result.keys())}")
        print(f"   - Algorithm: {result.get('algorithm', 'N/A')}")
        print(f"   - Trajectory reward: {result.get('trajectory_reward', 0.0):.4f}")
        print(f"   - Total updates: {result.get('total_updates', 0)}")
        print(f"   - Num steps: {result.get('num_steps', 0)}")
        print(f"   - Success: {result.get('success', 0.0):.2f}")
        print(f"   - Efficiency: {result.get('efficiency', 0.0):.2f}")
        print(f"   - Self-correction: {result.get('self_correction', 0.0):.2f}")
        
        if 'error' in result:
            print(f"   ⚠️  Warning: {result['error']}")
        
    except Exception as e:
        print(f"   ❌ Error during policy update: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test training statistics
    print("\n6. Checking training statistics...")
    try:
        stats = trainer.training_stats
        print(f"   ✓ Total updates: {stats['total_updates']}")
        print(f"   ✓ Total episodes: {stats['total_episodes']}")
        print(f"   ✓ Average reward: {stats['avg_reward']:.4f}")
        print(f"   ✓ Average loss: {stats['avg_loss']:.4f}")
        
    except Exception as e:
        print(f"   ❌ Error checking stats: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("✓ All basic tests passed!")
    print("=" * 70)
    return True


def test_trl_grpo_configs():
    """Test different configuration presets."""
    print("\n" + "=" * 70)
    print("Testing TRL GRPO Configuration Presets")
    print("=" * 70)
    
    if not TRL_AVAILABLE:
        print("❌ TRL is not installed!")
        return False
    
    presets = ["default", "fast", "quality", "memory_efficient"]
    
    print("\nAvailable configuration presets:")
    for i, preset_name in enumerate(presets, 1):
        try:
            config = get_grpo_preset(preset_name)
            print(f"\n{i}. {preset_name.upper()}")
            print(f"   - Learning rate: {config.learning_rate}")
            print(f"   - Num generations: {config.num_generations}")
            print(f"   - Temperature: {config.temperature}")
            print(f"   - Max new tokens: {config.max_new_tokens}")
            print(f"   - Beta (KL penalty): {config.beta}")
            print(f"   - Batch size: {config.per_device_train_batch_size}")
            print(f"   - Gradient accumulation: {config.gradient_accumulation_steps}")
            print(f"   - Gradient checkpointing: {config.gradient_checkpointing}")
            print(f"   - Reward weights: "
                  f"success={config.success_weight}, "
                  f"efficiency={config.efficiency_weight}, "
                  f"self_correction={config.self_correction_weight}")
            
        except Exception as e:
            print(f"   ❌ Error loading config '{preset_name}': {e}")
            return False
    
    print("\n" + "=" * 70)
    print("✓ All configuration presets loaded successfully!")
    print("=" * 70)
    return True


def test_trl_grpo_checkpoint():
    """Test saving and loading checkpoints."""
    print("\n" + "=" * 70)
    print("Testing TRL GRPO Checkpoint Saving/Loading")
    print("=" * 70)
    
    if not TRL_AVAILABLE:
        print("❌ TRL is not installed!")
        return False
    
    import tempfile
    import os
    
    print("\n1. Creating trainer...")
    try:
        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        config = get_grpo_preset("fast")
        config.verbose = 0  # Reduce verbosity
        config.num_generations = 2
        config.max_new_tokens = 16
        config.output_dir = "./test_checkpoint_output"
        
        trainer = TRLGRPOTrainer(model, tokenizer, config)
        print("   ✓ Trainer created")
        
    except Exception as e:
        print(f"   ❌ Error creating trainer: {e}")
        return False
    
    # Do a quick update to have some stats
    print("\n2. Performing quick training update...")
    try:
        judgment = BrowserJudgment(success=0.8, efficiency=0.7, self_correction=0.6)
        trainer.update_policy(
            ["Test prompt"],
            ["Test response"],
            judgment
        )
        print("   ✓ Training update completed")
    except Exception as e:
        print(f"   ⚠️  Training update failed: {e}")
        # Continue with checkpoint test even if update fails
    
    # Save checkpoint
    print("\n3. Saving checkpoint...")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "test_checkpoint")
            trainer.save_checkpoint(checkpoint_path)
            
            # Check if files exist
            assert os.path.exists(checkpoint_path), "Checkpoint directory not created"
            assert os.path.exists(os.path.join(checkpoint_path, "training_stats.json")), "Stats file not created"
            assert os.path.exists(os.path.join(checkpoint_path, "grpo_config.json")), "Config file not created"
            
            print(f"   ✓ Checkpoint saved to: {checkpoint_path}")
            print(f"   ✓ Files created:")
            for file in os.listdir(checkpoint_path):
                print(f"      - {file}")
            
            # Test loading
            print("\n4. Loading checkpoint...")
            new_trainer = TRLGRPOTrainer(model, tokenizer, config)
            new_trainer.load_checkpoint(checkpoint_path)
            print("   ✓ Checkpoint loaded successfully")
            
            # Verify stats were loaded
            print(f"   ✓ Loaded stats: total_updates={new_trainer.training_stats['total_updates']}")
        
    except Exception as e:
        print(f"   ❌ Error with checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 70)
    print("✓ Checkpoint tests passed!")
    print("=" * 70)
    return True


def test_trl_grpo_multiple_updates():
    """Test multiple policy updates to verify statistics tracking."""
    print("\n" + "=" * 70)
    print("Testing TRL GRPO Multiple Updates")
    print("=" * 70)
    
    if not TRL_AVAILABLE:
        print("❌ TRL is not installed!")
        return False
    
    print("\n1. Creating trainer...")
    try:
        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        config = get_grpo_preset("fast")
        config.num_generations = 2
        config.max_new_tokens = 16
        config.verbose = 0
        config.output_dir = "./test_multiple_output"
        
        trainer = TRLGRPOTrainer(model, tokenizer, config)
        print("   ✓ Trainer created")
        
    except Exception as e:
        print(f"   ❌ Error creating trainer: {e}")
        return False
    
    # Perform multiple updates
    print("\n2. Performing 3 training updates...")
    episodes = [
        {
            "prompts": ["Navigate to page"],
            "responses": ["Navigating..."],
            "judgment": BrowserJudgment(success=0.7, efficiency=0.6, self_correction=0.5),
        },
        {
            "prompts": ["Click button"],
            "responses": ["Clicking..."],
            "judgment": BrowserJudgment(success=0.9, efficiency=0.8, self_correction=0.7),
        },
        {
            "prompts": ["Enter text"],
            "responses": ["Typing..."],
            "judgment": BrowserJudgment(success=0.85, efficiency=0.75, self_correction=0.65),
        },
    ]
    
    try:
        for i, episode in enumerate(episodes, 1):
            print(f"\n   Episode {i}:")
            result = trainer.update_policy(
                episode["prompts"],
                episode["responses"],
                episode["judgment"]
            )
            print(f"      Reward: {result.get('trajectory_reward', 0.0):.3f}")
            if 'avg_reward' in result:
                print(f"      Avg Reward: {result['avg_reward']:.3f}")
            if 'total_updates' in result:
                print(f"      Total Updates: {result['total_updates']}")
            if 'error' in result:
                print(f"      ⚠️  Error: {result['error']}")
        
        # Verify statistics (only if no errors)
        print("\n3. Verifying statistics...")
        if trainer.training_stats["total_updates"] > 0:
            assert trainer.training_stats["total_updates"] == 3, "Update count mismatch"
            assert trainer.training_stats["total_episodes"] == 3, "Episode count mismatch"
            print("   ✓ Statistics verified")
        else:
            print("   ⚠️  No successful updates recorded")
            return False
        
    except Exception as e:
        print(f"   ❌ Error during updates: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 70)
    print("✓ Multiple updates test passed!")
    print("=" * 70)
    return True


def test_trl_grpo_custom_config():
    """Test creating a custom configuration."""
    print("\n" + "=" * 70)
    print("Testing TRL GRPO Custom Configuration")
    print("=" * 70)
    
    if not TRL_AVAILABLE:
        print("❌ TRL is not installed!")
        return False
    
    print("\n1. Creating custom configuration...")
    try:
        custom_config = TRLGRPOConfig(
            learning_rate=2e-5,
            num_generations=6,
            temperature=0.85,
            max_new_tokens=256,
            beta=0.08,
            success_weight=0.6,
            efficiency_weight=0.25,
            self_correction_weight=0.15,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            gradient_checkpointing=False,
            output_dir="./test_custom_output"
        )
        
        print("   ✓ Custom config created")
        print(f"   - Learning rate: {custom_config.learning_rate}")
        print(f"   - Num generations: {custom_config.num_generations}")
        print(f"   - Temperature: {custom_config.temperature}")
        print(f"   - Beta: {custom_config.beta}")
        print(f"   - Reward weights: {custom_config.success_weight}, "
              f"{custom_config.efficiency_weight}, {custom_config.self_correction_weight}")
        
    except Exception as e:
        print(f"   ❌ Error creating custom config: {e}")
        return False
    
    print("\n2. Testing trainer creation with custom config...")
    try:
        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Override for faster testing
        custom_config.num_generations = 2
        custom_config.max_new_tokens = 16
        custom_config.verbose = 0
        
        trainer = create_trl_grpo_trainer(model, tokenizer, custom_config)
        print("   ✓ Trainer created with custom config")
        
    except Exception as e:
        print(f"   ❌ Error creating trainer: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("✓ Custom configuration test passed!")
    print("=" * 70)
    return True


def test_trl_grpo_interface_compatibility():
    """Test that TRL GRPO has the same interface as other RL algorithms."""
    print("\n" + "=" * 70)
    print("Testing TRL GRPO Interface Compatibility")
    print("=" * 70)
    
    if not TRL_AVAILABLE:
        print("❌ TRL is not installed!")
        return False
    
    print("\n1. Checking required methods...")
    try:
        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        config = get_grpo_preset("fast")
        config.output_dir = "./test_interface_output"
        trainer = TRLGRPOTrainer(model, tokenizer, config)
        
        # Check required methods exist
        required_methods = [
            "update_policy",
            "save_checkpoint",
            "load_checkpoint",
        ]
        
        for method_name in required_methods:
            assert hasattr(trainer, method_name), f"Missing method: {method_name}"
            print(f"   ✓ Method exists: {method_name}")
        
        # Check required attributes
        required_attrs = [
            "model",
            "tokenizer",
            "config",
            "training_stats",
        ]
        
        for attr_name in required_attrs:
            assert hasattr(trainer, attr_name), f"Missing attribute: {attr_name}"
            print(f"   ✓ Attribute exists: {attr_name}")
        
    except Exception as e:
        print(f"   ❌ Interface check failed: {e}")
        return False
    
    print("\n2. Verifying update_policy signature...")
    try:
        import inspect
        sig = inspect.signature(trainer.update_policy)
        params = list(sig.parameters.keys())
        
        expected_params = ["trajectory_prompts", "trajectory_responses", "judgment"]
        for param in expected_params:
            assert param in params, f"Missing parameter: {param}"
            print(f"   ✓ Parameter exists: {param}")
        
    except Exception as e:
        print(f"   ❌ Signature check failed: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("✓ Interface compatibility test passed!")
    print("=" * 70)
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("TRL GRPO IMPLEMENTATION TEST SUITE")
    print("=" * 70)
    
    tests = [
        ("Basic Functionality", test_trl_grpo_basic),
        ("Configuration Presets", test_trl_grpo_configs),
        ("Checkpoint Save/Load", test_trl_grpo_checkpoint),
        ("Multiple Updates", test_trl_grpo_multiple_updates),
        ("Custom Configuration", test_trl_grpo_custom_config),
        ("Interface Compatibility", test_trl_grpo_interface_compatibility),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Print summary
    print("\n\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    for test_name, success in results:
        status = "✓ PASSED" if success else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    all_passed = all(success for _, success in results)
    print("=" * 70)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
