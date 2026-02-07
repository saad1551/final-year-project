# Testing the Pipeline with Mock Data

## ✅ Updated: SB3 PPO Support Added!

The mock test (`test_rl_mock.py`) now supports **all 4 algorithms** including SB3 PPO.

## Quick Test Commands

### Test Custom Algorithms (No Dependencies Required)

```bash
# Test REINFORCE
python3 test_rl_mock.py --algorithm reinforce --num_trajectories 5

# Test custom PPO
python3 test_rl_mock.py --algorithm ppo --num_trajectories 5

# Test GRPO
python3 test_rl_mock.py --algorithm grpo --num_trajectories 5
```

### Test SB3 PPO (Requires stable-baselines3)

```bash
# Install dependencies first
pip install stable-baselines3 gymnasium

# Test SB3 PPO with default preset
python3 test_rl_mock.py --algorithm sb3_ppo --num_trajectories 5

# Test with low_memory preset
python3 test_rl_mock.py --algorithm sb3_ppo --sb3_preset low_memory --num_trajectories 5

# Test with aggressive preset
python3 test_rl_mock.py --algorithm sb3_ppo --sb3_preset aggressive --num_trajectories 5
```

## What the Mock Test Does

The mock test (`test_rl_mock.py`) **simulates the entire RL training pipeline** without requiring:
- ❌ GPU (uses mock model, not real LLM)
- ❌ Browser (uses pre-generated trajectories)
- ❌ Judge LLM (generates random but realistic scores)

It tests:
- ✅ Trainer initialization
- ✅ Policy updates with mock trajectories
- ✅ Loss computation
- ✅ Gradient backpropagation
- ✅ Training statistics tracking
- ✅ All 4 algorithms (reinforce, ppo, grpo, sb3_ppo)

## Example Output

```bash
$ python3 test_rl_mock.py --algorithm sb3_ppo --num_trajectories 3

============================================================
MOCK RL TRAINING TEST
============================================================
Algorithm: SB3_PPO
Trajectories: 3
Steps per trajectory: 3
Debug mode: True
============================================================

[SETUP] Creating mock model and tokenizer...
[SETUP] Mock model created with 8,448,000 parameters (8,448,000 trainable)

[SETUP] Initializing SB3_PPO trainer...
[SETUP] SB3 PPO trainer initialized (default preset)

============================================================
STARTING TRAINING LOOP
============================================================

────────────────────────────────────────────────
TRAJECTORY 1/3
────────────────────────────────────────────────
[TRAJECTORY] Generated 3 steps (difficulty: medium)
[TRAJECTORY] Judgment: success=0.712, efficiency=0.589, self_correction=0.634

[UPDATE] Performing SB3_PPO update...

[RESULT] Trajectory 1 complete:
  • Reward: 0.6548
  • Policy Loss: 0.0123
  • Total Loss: 0.0145
...

============================================================
TRAINING COMPLETE
============================================================
Total trajectories: 3
Total updates: 3
Average reward: 0.6213
Average loss: 0.0156
Min reward: 0.5892
Max reward: 0.6548

Reward progression:
  Trajectory 1: 0.655 |█████████████
  Trajectory 2: 0.589 |███████████
  Trajectory 3: 0.621 |████████████

[SUCCESS] Mock RL training completed successfully! ✓
```

## All Test Options

```bash
python3 test_rl_mock.py --help

Options:
  --algorithm {reinforce,ppo,grpo,sb3_ppo}
                        RL algorithm to test
  --num_trajectories N  Number of mock trajectories
  --steps N             Steps per trajectory
  --debug               Enable debug logging (default: True)
  --no-debug            Disable debug logging
  --sb3_preset {default,low_memory,aggressive,conservative,exploration}
                        SB3 PPO preset (only used with sb3_ppo)
```

## Comparison Testing

Test all algorithms to compare:

```bash
# Custom algorithms
python3 test_rl_mock.py --algorithm reinforce --num_trajectories 10
python3 test_rl_mock.py --algorithm ppo --num_trajectories 10
python3 test_rl_mock.py --algorithm grpo --num_trajectories 10

# SB3 PPO
python3 test_rl_mock.py --algorithm sb3_ppo --num_trajectories 10
```

## Next Steps

Once the mock test passes, you can run the **full pipeline** with real browser and judge:

```bash
# Full pipeline with SB3 PPO
python3 pipeline_in_steps.py --algorithm sb3_ppo --num_trajectories 1
```

## Summary

| Test Type | File | Requires | Tests |
|-----------|------|----------|-------|
| **Mock Test** | `test_rl_mock.py` | None (torch only) | RL algorithms with fake data |
| **Full Pipeline** | `pipeline_in_steps.py` | GPU, Browser, Judge | Complete end-to-end flow |

**Recommended workflow:**
1. Run mock test first to verify algorithm works
2. Then run full pipeline for real training

Both now support all 4 algorithms including SB3 PPO! 🎉
