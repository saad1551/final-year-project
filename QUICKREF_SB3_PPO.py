"""
QUICK REFERENCE: Stable Baselines3 PPO Implementation
=====================================================

INSTALLATION
-----------
pip install stable-baselines3 gymnasium

BASIC USAGE
----------
from rl_sb3_ppo import SB3PPOTrainer
from rl_sb3_config_examples import get_config

config = get_config("default")  # or low_memory, aggressive, etc.
trainer = SB3PPOTrainer(model, tokenizer, config)

result = trainer.update_policy(prompts, responses, judgment)
trainer.save_checkpoint("path/to/checkpoint")

CONFIGURATION PRESETS
-------------------
default          - Standard PPO settings
low_memory       - For limited GPU memory (512 steps, batch=32)
aggressive       - Fast learning (lr=1e-3, batch=128)
conservative     - Stable learning (lr=1e-4, clip=0.1)
exploration      - High exploration (ent_coef=0.05, SDE)
success_focused  - Success weight=0.8
efficiency_focused - Efficiency weight=0.5

CUSTOM CONFIG
------------
from rl_sb3_ppo import SB3PPOConfig

config = SB3PPOConfig(
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    clip_range=0.2,
    ent_coef=0.01,
    success_weight=0.5,
    efficiency_weight=0.3,
    self_correction_weight=0.2
)

KEY PARAMETERS
-------------
Learning:
  - learning_rate: 3e-4 (default)
  - n_steps: 2048 (steps per update)
  - batch_size: 64 (minibatch size)
  - n_epochs: 10 (epochs per update)
  - gamma: 0.99 (discount factor)

PPO-Specific:
  - clip_range: 0.2 (PPO clipping)
  - ent_coef: 0.01 (entropy coefficient)
  - vf_coef: 0.5 (value function coefficient)
  - max_grad_norm: 0.5 (gradient clipping)

Rewards:
  - success_weight: 0.5
  - efficiency_weight: 0.3
  - self_correction_weight: 0.2

INTERFACE (same as custom RL algorithms)
---------------------------------------
trainer.update_policy(prompts, responses, judgment)
  -> Returns dict with metrics

trainer.save_checkpoint(path)
  -> Saves model and stats

trainer.load_checkpoint(path)
  -> Loads model and stats

trainer.training_stats
  -> Dict with total_updates, avg_reward, avg_loss

INTEGRATION EXAMPLE
------------------
# Add to existing pipeline
if algorithm == "sb3_ppo":
    config = get_config("default")
    trainer = SB3PPOTrainer(model, tokenizer, config)
elif algorithm in ["reinforce", "ppo", "grpo"]:
    config = RLConfig(algorithm=algorithm)
    trainer = OnPolicyTrainer(model, tokenizer, config)

# Both use same interface!
result = trainer.update_policy(prompts, responses, judgment)

TESTING
-------
python test_sb3_ppo.py

DOCUMENTATION
------------
README_SB3_PPO.md                 - Full user guide
SUMMARY_SB3_PPO.md                - Implementation summary
rl_sb3_integration_examples.py    - Integration examples
test_sb3_ppo.py                   - Test suite

FILES
-----
rl_sb3_ppo.py                     - Main implementation
rl_sb3_config_examples.py         - Configuration presets
rl_sb3_integration_examples.py    - Integration examples
test_sb3_ppo.py                   - Tests
README_SB3_PPO.md                 - Documentation
SUMMARY_SB3_PPO.md                - Summary

TROUBLESHOOTING
--------------
Issue: ImportError stable-baselines3
Fix:   pip install stable-baselines3

Issue: CUDA out of memory
Fix:   Use get_config("low_memory")

Issue: Slow training
Fix:   Use get_config("aggressive")

Issue: Unstable training
Fix:   Use get_config("conservative")

COMPARISON
---------
SB3 PPO:
  ✓ Well-tested, production-ready
  ✓ Advanced features (SDE, tensorboard)
  ✓ Community support
  ✗ Additional dependencies

Custom PPO:
  ✓ Full control over implementation
  ✓ Tighter integration
  ✓ Fewer dependencies
  ✗ Less tested than SB3
"""
