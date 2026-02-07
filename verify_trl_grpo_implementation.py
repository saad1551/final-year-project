#!/usr/bin/env python3
"""
Quick verification test for TRL GRPO multi-execution implementation.
Tests that the code can at least be imported and basic logic works.
"""

import sys
sys.path.insert(0, '/Users/saadashraf/fyp/final-year-project')

def test_imports():
    """Test that all necessary imports work."""
    print("Testing imports...")
    try:
        from rl_trl_grpo import TRLGRPOTrainer, get_grpo_preset, TRLGRPOConfig
        from insta.configs.judge_config import BrowserJudgment
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_backward_compatibility():
    """Test that update_policy accepts both single and multiple judgments."""
    print("\nTesting backward compatibility...")
    try:
        from rl_trl_grpo import TRLGRPOTrainer, TRLGRPOConfig
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from insta.configs.judge_config import BrowserJudgment
        import torch
        
        # Create mock model and tokenizer
        print("  Loading tiny model for testing...")
        model_name = "gpt2"  # Small model for quick test
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Create trainer
        config = TRLGRPOConfig(learning_rate=1e-5)
        trainer = TRLGRPOTrainer(model, tokenizer, config)
        
        # Test single judgment (backward compatible)
        print("  Testing single judgment mode...")
        judgment = BrowserJudgment(success=0.8, efficiency=0.7, self_correction=0.6)
        
        result = trainer.update_policy(
            trajectory_prompts=["Test prompt"],
            trajectory_responses=["Test response"],
            judgment=judgment  # Single judgment
        )
        
        assert 'trajectory_reward' in result
        assert result['multi_execution'] == False
        print("  ✅ Single judgment mode works")
        
        # Test multiple judgments (new feature)
        print("  Testing multiple judgments mode...")
        judgments = [
            BrowserJudgment(success=0.8, efficiency=0.7, self_correction=0.6),
            BrowserJudgment(success=0.6, efficiency=0.5, self_correction=0.4),
            BrowserJudgment(success=0.9, efficiency=0.8, self_correction=0.7),
        ]
        
        result = trainer.update_policy(
            trajectory_prompts=["Test prompt"],
            trajectory_responses=["Test response"],
            judgments=judgments  # Multiple judgments
        )
        
        assert 'trajectory_reward' in result
        assert result['multi_execution'] == True
        assert 'num_attempts' in result
        assert result['num_attempts'] == 3
        assert 'min_reward' in result
        assert 'max_reward' in result
        assert 'reward_std' in result
        print("  ✅ Multiple judgments mode works")
        
        print("✅ Backward compatibility verified")
        return True
        
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pipeline_args():
    """Test that pipeline has the new arguments."""
    print("\nTesting pipeline arguments...")
    try:
        import argparse
        import importlib.util
        
        # Load pipeline_in_steps as a module
        spec = importlib.util.spec_from_file_location(
            "pipeline", 
            "/Users/saadashraf/fyp/final-year-project/pipeline_in_steps.py"
        )
        # Just check the file can be parsed
        with open("/Users/saadashraf/fyp/final-year-project/pipeline_in_steps.py") as f:
            content = f.read()
            
        assert 'trl_grpo' in content
        assert 'trl_grpo_preset' in content
        assert 'trl_grpo_num_generations' in content
        assert 'needs_multi_execution' in content
        assert 'all_judgments' in content
        
        print("✅ Pipeline has multi-execution logic")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test error: {e}")
        return False

def main():
    print("="*60)
    print("TRL GRPO Multi-Execution Verification Test")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Backward Compatibility", test_backward_compatibility()))
    results.append(("Pipeline Arguments", test_pipeline_args()))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(r[1] for r in results)
    print("\n" + "="*60)
    if all_passed:
        print("🎉 All tests passed! Implementation verified.")
    else:
        print("⚠️  Some tests failed. Review the errors above.")
    print("="*60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
