# Testing TRL GRPO Implementation

This document provides instructions for testing the TRL GRPO implementation.

## Quick Start

### 1. Install Dependencies

First, ensure you have all required dependencies:

```bash
pip install trl>=0.8.0 datasets>=2.14.0 transformers torch
```

Or install from requirements.txt:

```bash
pip install -r requirements.txt
```

### 2. Run the Test Suite

Run all tests:

```bash
python test_trl_grpo.py
```

This will execute 6 test cases:
1. ✅ Basic Functionality
2. ✅ Configuration Presets
3. ✅ Checkpoint Save/Load
4. ✅ Multiple Updates
5. ✅ Custom Configuration
6. ✅ Interface Compatibility

## Test Details

### Test 1: Basic Functionality
Tests core functionality including:
- Model and tokenizer loading
- Trainer initialization
- Policy update with mock data
- Training statistics tracking

**Expected output:**
```
✓ Trainer initialized successfully
✓ Policy update completed
✓ Total updates: 1
```

### Test 2: Configuration Presets
Verifies all preset configurations load correctly:
- `default` - Balanced settings
- `fast` - Quick experiments (2 generations)
- `quality` - Best results (8 generations)
- `memory_efficient` - Low memory usage

**Expected output:**
```
✓ All configuration presets loaded successfully!
```

### Test 3: Checkpoint Save/Load
Tests checkpoint persistence:
- Save checkpoint with training stats
- Load checkpoint into new trainer
- Verify stats are restored

**Expected output:**
```
✓ Checkpoint saved to: <path>
✓ Checkpoint loaded successfully
```

### Test 4: Multiple Updates
Verifies statistics tracking across multiple episodes:
- 3 consecutive policy updates
- Statistics accumulation
- Reward tracking

**Expected output:**
```
Episode 1: Reward: 0.650
Episode 2: Reward: 0.820
Episode 3: Reward: 0.758
✓ Statistics verified
```

### Test 5: Custom Configuration
Tests creating custom configurations:
- Custom reward weights
- Custom hyperparameters
- Trainer initialization with custom config

**Expected output:**
```
✓ Custom config created
✓ Trainer created with custom config
```

### Test 6: Interface Compatibility
Verifies interface matches other RL algorithms:
- Required methods exist
- Required attributes exist
- Method signatures are correct

**Expected output:**
```
✓ Method exists: update_policy
✓ Method exists: save_checkpoint
✓ Method exists: load_checkpoint
```

## Running Individual Tests

You can modify `test_trl_grpo.py` to run specific tests:

```python
if __name__ == "__main__":
    # Run only basic test
    test_trl_grpo_basic()
    
    # Or run only config test
    test_trl_grpo_configs()
```

## Performance Notes

### Test Duration

- **Basic Functionality**: ~15-30 seconds (includes model generation)
- **Configuration Presets**: <1 second (just config validation)
- **Checkpoint Save/Load**: ~5-10 seconds
- **Multiple Updates**: ~30-60 seconds (3 generations)
- **Custom Configuration**: ~5-10 seconds
- **Interface Compatibility**: <1 second

**Total suite runtime**: ~1-2 minutes

### Memory Requirements

The tests use GPT-2 (small model) for quick testing:
- Model size: ~124M parameters
- Memory usage: ~500MB-1GB
- Recommended: At least 2GB available RAM

## Troubleshooting

### TRL Not Found

```
❌ TRL is not installed!
```

**Solution:**
```bash
pip install trl>=0.8.0 datasets>=2.14.0
```

### CUDA Out of Memory

If you get OOM errors:

```python
# In test file, add this before model loading:
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU

# Or reduce generations:
config.num_generations = 1
config.max_new_tokens = 8
```

### Tests Take Too Long

Speed up tests by reducing parameters in `test_trl_grpo.py`:

```python
config.num_generations = 1  # Reduce from 2
config.max_new_tokens = 8   # Reduce from 16
```

### Import Errors

If you see import errors for `insta.configs.judge_config`:

```python
# Make sure you're in the correct directory
cd /Users/saadashraf/fyp/final-year-project

# Or check PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/Users/saadashraf/fyp/final-year-project"
```

## Expected Test Output

Successful test run should look like:

```
======================================================================
TRL GRPO IMPLEMENTATION TEST SUITE
======================================================================

======================================================================
Testing TRL GRPO - Basic Functionality
======================================================================

✓ TRL is available

1. Creating mock model and tokenizer...
   ✓ Model loaded: gpt2
   ✓ Model size: 124.44M parameters

2. Creating TRL GRPO configuration...
   ✓ Config: learning_rate=3e-05
   ✓ Config: num_generations=2

[... more output ...]

======================================================================
TEST SUMMARY
======================================================================
✓ PASSED: Basic Functionality
✓ PASSED: Configuration Presets
✓ PASSED: Checkpoint Save/Load
✓ PASSED: Multiple Updates
✓ PASSED: Custom Configuration
✓ PASSED: Interface Compatibility
======================================================================
✓ ALL TESTS PASSED!
======================================================================
```

## Customizing Tests

### Use Your Own Model

Replace the model in tests:

```python
# Instead of:
model_name = "gpt2"

# Use:
model_name = "Qwen/Qwen2-0.5B-Instruct"
# Or your custom model path
```

### Add More Test Cases

Add new test functions following the pattern:

```python
def test_your_feature():
    """Test description."""
    print("\n" + "=" * 70)
    print("Testing Your Feature")
    print("=" * 70)
    
    # Your test code here
    
    return True  # or False if failed

# Add to tests list in main():
tests = [
    # ... existing tests
    ("Your Feature", test_your_feature),
]
```

## Integration Testing

To test with actual pipeline integration:

1. First run unit tests: `python test_trl_grpo.py`
2. Then test with pipeline (once integrated):
   ```bash
   python pipeline_in_steps.py --algorithm trl_grpo --preset fast
   ```

## Continuous Testing

Run tests before committing changes:

```bash
# Quick smoke test (basic functionality only)
python -c "from test_trl_grpo import test_trl_grpo_basic; test_trl_grpo_basic()"

# Full test suite
python test_trl_grpo.py
```

## Comparison with Other RL Implementations

Run comparison tests:

```bash
# Test SB3 PPO
python test_sb3_ppo.py

# Test TRL GRPO
python test_trl_grpo.py

# Test custom RL (if available)
python test_rl_mock.py
```

All should show similar interface and behavior!

## Next Steps

After tests pass:
1. ✅ Unit tests pass - You're here!
2. ⏭️ Integration tests - Integrate into pipeline
3. ⏭️ End-to-end tests - Test with real browser tasks
4. ⏭️ Performance comparison - Compare with other algorithms

## Getting Help

If tests fail:
1. Check the error message carefully
2. Review the test output for specific failures
3. Check `TRL_GRPO_README.md` for configuration help
4. Review `GRPO_COMPARISON.md` for algorithm details
5. Check TRL documentation: https://huggingface.co/docs/trl/grpo_trainer
