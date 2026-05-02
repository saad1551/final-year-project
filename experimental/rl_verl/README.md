# Experimental — VeRL / GRPO training path

This directory was previously `rl/` at the repo root. It contains a parallel training pipeline based on the **VeRL** framework targeting **Qwen2.5-1.5B with GRPO**, distinct from the PPO + LoRA path that the published experiments use (`pipeline_in_steps.py`).

It is preserved here for reference, not used in the paper's experiments.

## Why it's separated

- The main pipeline (`pipeline_in_steps.py`) implements custom PPO with a reference-KL anchor on a Qwen3-1.7B SFT model.
- This directory implements GRPO via VeRL on Qwen2.5-1.5B, going through InSTA's own `InstaPipeline` rather than our trajectory loop.
- The two share no code paths — neither imports from the other.

## Contents

| File | Purpose |
|---|---|
| `insta_pipeline.py` | Wraps InSTA's `InstaPipeline` for training trajectories |
| `reward_func.py` | Reward function for VeRL/GRPO |
| `create_verl_dataset.py` | Convert InSTA tasks to VeRL training format |
| `verl_to_huggingface.py` | Convert VeRL outputs back to HF-style checkpoints |
| `train_grpo_qwen2.5-1.5b.sh` | Launch script for the GRPO training run |
| `training_loop.sh` | Orchestrates trajectory collection + GRPO update |
| `start_*.sh` | Component launchers (LLM server, rollout pipeline, eval pipeline, slurm sbatch) |
| `run_qwen_1_7B.ipynb` | Exploratory notebook |

## Running it (advanced)

These scripts assume:
- A running InSTA pipeline (start with `start_insta_pipeline.sbatch` on a slurm cluster, or `start_llm_server.sh` + `start_rollout_pipeline.sh` for a local setup)
- VeRL installed alongside the standard `requirements.txt`
- A separate Qwen2.5-1.5B SFT checkpoint as the warm-start

We do not maintain or test this path. If it needs updates after the move (e.g., for `from rl.foo` imports), they live in the surrounding shell scripts only — Python imports inside this directory don't reference the repo root.
