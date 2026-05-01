# Experimental — SB3 PPO path

This directory contains an alternative training path that wraps Stable-Baselines3 (SB3) PPO around a Gym environment. **It is not the path used in the published experiments** in this repository, and we recommend against using it for new work. Two reasons:

1. **It does not train on real trajectories.** As `pipeline_in_steps.py` itself warns when you select `--algorithm sb3_ppo`:

   > *"SB3 PPO trains on a dummy environment, NOT your real trajectory data."*

   The wrapper feeds SB3 a synthetic rollout signal rather than the actual browser-agent reward stream produced by the trajectory loop, so the gradient updates are not connected to agent behavior in any meaningful way.

2. **The custom PPO + reference-KL implementation in `rl_trainer.py` is the actual training path** — it handles real trajectories with sparse terminal rewards, applies the LoRA + reference-KL anchor we describe in §4 of the report, and is what produced the results in our paper.

Files retained here for reference / future work:

- `rl_sb3_ppo.py` — wraps SB3 PPO around a Gym environment
- `rl_sb3_config_examples.py` — hyperparameter presets for the SB3 path
- `rl_sb3_integration_examples.py` — example invocations
- `gym_env.py` — the dummy Gym environment SB3 uses
- `test_sb3_ppo.py` — tests for the above

## Using the SB3 path despite the caveat

If you really want to run it (e.g. to reproduce our negative observation that this path doesn't train usefully), you'll need to revert the import refactor in `pipeline_in_steps.py` — the imports of these modules are now lazy and only fire when `--algorithm sb3_ppo` is passed. From the repo root:

```bash
python pipeline_in_steps.py --algorithm sb3_ppo --sb3_preset aggressive --num_trajectories 50
```
