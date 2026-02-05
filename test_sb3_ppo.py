"""
Test script for Stable Baselines3 PPO implementation.

This script demonstrates how to use the SB3 PPO trainer with
mock data, similar to test_rl_mock.py for custom RL algorithms.
"""

import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from insta.configs.judge_config import BrowserJudgment

# Import SB3 PPO components
try:
    from rl_sb3_ppo import SB3PPOTrainer, SB3PPOConfig, SB3_AVAILABLE
    from rl_sb3_config_examples import get_config, CONFIG_PRESETS
except ImportError as e:
    print(f"Error importing SB3 PPO modules: {e}")
    print("Make sure stable-baselines3 is installed: pip install stable-baselines3")
    sys.exit(1)


def test_sb3_ppo_basic():
    """Test basic SB3 PPO functionality with mock data."""
    print("=" * 70)
    print("Testing Stable Baselines3 PPO - Basic Functionality")
    print("=" * 70)
    
    if not SB3_AVAILABLE:
        print("❌ Stable Baselines3 is not installed!")
        print("   Install with: pip install stable-baselines3")
        return False
    
    print("\n✓ Stable Baselines3 is available")
    
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
    
    # Create SB3 PPO configuration
    print("\n2. Creating SB3 PPO configuration...")
    try:
        # Use low memory config for testing
        config = get_config("low_memory")
        config.verbose = 1
        config.n_steps = 128  # Very small for quick testing
        config.batch_size = 32
        config.n_epochs = 2
        
        print(f"   ✓ Config: learning_rate={config.learning_rate}")
        print(f"   ✓ Config: n_steps={config.n_steps}, batch_size={config.batch_size}")
        print(f"   ✓ Config: clip_range={config.clip_range}, ent_coef={config.ent_coef}")
        
    except Exception as e:
        print(f"   ❌ Error creating config: {e}")
        return False
    
    # Create SB3 PPO trainer
    print("\n3. Initializing SB3 PPO trainer...")
    try:
        trainer = SB3PPOTrainer(
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
        print(f"   ✓ Average reward: {stats['avg_reward']:.4f}")
        print(f"   ✓ Average loss: {stats['avg_loss']:.4f}")
        
    except Exception as e:
        print(f"   ❌ Error checking stats: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("✓ All basic tests passed!")
    print("=" * 70)
    return True


def test_sb3_ppo_configs():
    """Test different configuration presets."""
    print("\n" + "=" * 70)
    print("Testing SB3 PPO Configuration Presets")
    print("=" * 70)
    
    if not SB3_AVAILABLE:
        print("❌ Stable Baselines3 is not installed!")
        return False
    
    print("\nAvailable configuration presets:")
    for i, preset_name in enumerate(CONFIG_PRESETS.keys(), 1):
        try:
            config = get_config(preset_name)
            print(f"\n{i}. {preset_name.upper()}")
            print(f"   - Learning rate: {config.learning_rate}")
            print(f"   - N steps: {config.n_steps}")
            print(f"   - Batch size: {config.batch_size}")
            print(f"   - Clip range: {config.clip_range}")
            print(f"   - Entropy coef: {config.ent_coef}")
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


def test_sb3_ppo_checkpoint():
    """Test saving and loading checkpoints."""
    print("\n" + "=" * 70)
    print("Testing SB3 PPO Checkpoint Saving/Loading")
    print("=" * 70)
    
    if not SB3_AVAILABLE:
        print("❌ Stable Baselines3 is not installed!")
        return False
    
    import tempfile
    import os
    
    print("\n1. Creating trainer...")
    try:
        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        config = get_config("low_memory")
        config.verbose = 0  # Reduce verbosity
        
        trainer = SB3PPOTrainer(model, tokenizer, config)
        print("   ✓ Trainer created")
        
    except Exception as e:
        print(f"   ❌ Error creating trainer: {e}")
        return False
    
    # Save checkpoint
    print("\n2. Saving checkpoint...")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "test_checkpoint")
            trainer.save_checkpoint(checkpoint_path)
            
            # Check if files exist
            assert os.path.exists(checkpoint_path), "Checkpoint directory not created"
            assert os.path.exists(os.path.join(checkpoint_path, "training_stats.json")), "Stats file not created"
            
            print(f"   ✓ Checkpoint saved to: {checkpoint_path}")
            print(f"   ✓ Files created:")
            for file in os.listdir(checkpoint_path):
                print(f"      - {file}")
            
            # Test loading (just verify the method works)
            print("\n3. Loading checkpoint...")
            trainer.load_checkpoint(checkpoint_path)
            print("   ✓ Checkpoint loaded successfully")
        
    except Exception as e:
        print(f"   ❌ Error with checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 70)
    print("✓ Checkpoint tests passed!")
    print("=" * 70)
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("STABLE BASELINES3 PPO IMPLEMENTATION TEST SUITE")
    print("=" * 70)
    
    tests = [
        ("Basic Functionality", test_sb3_ppo_basic),
        ("Configuration Presets", test_sb3_ppo_configs),
        ("Checkpoint Save/Load", test_sb3_ppo_checkpoint),
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
