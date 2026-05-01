# Reproducibility

This document gives the exact commands to reproduce each experiment in `report/main.md`. Three sections, one per experiment:

1. **Section 3 — Feasibility audit** (Figure 1)
2. **Section 4 — Continued training** (training run + checkpoints)
3. **Section 4.5 — Held-out evaluation** (Table 1, Figure 3)

All commands assume you have followed `INSTALLATION.md`. For sections 2 and 3, a GCP VM with an L4 (or T4 — slightly slower) GPU is required.

---

## 1. Section 3 — Feasibility audit

### Inputs

- `data/insta-150k-test.csv` (~2 MB) — the InSTA test split, shipped in the repo.
- `data/insta-150k-train.csv` (~102 MB) — **not shipped**, fetched from Hugging Face:
  ```bash
  python data/download.py
  ```
  Requires the `datasets` package (already in `requirements.txt`). Re-run with `--force` to re-download.
- A Vertex-AI-enabled GCP project, OR a Gemini API key from https://aistudio.google.com/apikey.

### Run the feasibility classifier

```bash
# Using Vertex AI (preferred):
python sample_feasible_tasks.py \
  --vertex_project <YOUR_GCP_PROJECT> \
  --target 200 \
  --min_confidence 0.95 \
  --seed 42

# Using AI Studio API key:
JUDGE_API_KEY=<your_key> python sample_feasible_tasks.py \
  --target 200 \
  --min_confidence 0.95 \
  --seed 42
```

This iterates over the test CSV (shuffled with the given seed), calls Gemini-2.5-Flash with URL-context grounding, and accepts tasks classified as `FEASIBLE` with confidence ≥ 0.95 until 200 are collected. Output: a timestamped CSV under `feasibility_results/feasible_sample_<ts>.csv` plus rejected tasks under `rejected_sample_<ts>.csv`.

The feasibility audit numbers in §3 of the report are produced by **a separate, larger run** that processed 2,598 tasks. To reproduce that exact run is not feasible (Gemini's url_context tool is non-deterministic; specific website states have changed since), but the methodology and seed are documented so subsequent runs are interpretable.

### Reproduce Figure 1 (dataset-decay breakdown)

The figure is regenerated from a parsed log file:

```bash
# Parse the v2 sampling log into per-task records:
python feasibility_results/parse_v2_log.py
# -> writes feasibility_results/v2_log_parsed.csv (2,598 rows)

# Render the two-panel figure:
python feasibility_results/figure1_dataset_decay.py
# -> writes figure1_dataset_decay.{png,pdf}
```

### Reproduce the sanity-check sample (Appendix A)

```bash
python feasibility_results/sample_for_sanity_check.py
# -> writes feasibility_results/sanity_check_sample.md (28 spot-check tasks, seed=42)
```

---

## 2. Section 4 — Continued training

### Inputs

- A LoRA adapter to warm-start from. We used `checkpoints/checkpoint_trajectory_600/` from a prior training round. Training from `btrabucco/Insta-Qwen3-1.7B-SFT` directly (no warm-start) is also supported — point `RESUME_FROM` at any LoRA checkpoint.
- The feasibility-filtered training task CSV: `feasibility_results/feasible_sample_20260324_195836.csv` (2,068 tasks).
- A Vertex-AI-enabled GCP project (for the judge during RL rollouts).
- An L4 (or T4) GPU VM provisioned per `INSTALLATION.md`.

### On the VM

```bash
ssh into the VM, then:
cd ~/final-year-project

# Bring up the Playwright server with watchdog.
screen -S watchdog -dm bash ~/final-year-project/gcp/playwright_watchdog.sh

# Launch training. Defaults: 500 trajectories, save every 25, ref-KL every 5,
# min_reward=0.0, screenshots disabled, output to checkpoints_feasible/.
RESUME_DATASET_IDX=0 \
NUM_TRAJECTORIES=500 \
CHECKPOINT_DIR=checkpoints_feasible \
MIN_REWARD=0.0 \
RESUME_FROM=checkpoints/checkpoint_trajectory_600 \
TRAIN_CSV=feasibility_results/feasible_sample_20260324_195836.csv \
bash gcp/launch_training.sh
```

Training runs in a detached `tmux` session named `fyp-train`. To monitor:

```bash
# Attach (Ctrl-b d to detach):
tmux attach -t fyp-train

# Or tail the log:
tail -f ~/final-year-project/training_logs/training_log_*.csv
```

Estimated runtime: **~30 hours** for 500 trajectories on an L4 (varies with website-load latency, model improvement over time).

### Continue training from a saved checkpoint

```bash
# After the first 500 trajectories have completed, e.g. trajectory 1000 target:
RESUME_FROM=checkpoints_feasible/checkpoint_trajectory_500 \
NUM_TRAJECTORIES=1000 \
CHECKPOINT_DIR=checkpoints_feasible \
MIN_REWARD=0.0 \
bash gcp/launch_training.sh
```

The script auto-resumes from the checkpoint's metadata (no need to set `RESUME_DATASET_IDX` after the first run — it's preserved in the checkpoint).

### Reproduce Figure 2 (training progress)

After training, with the CSV available locally:

```bash
python monitor_training.py \
  --csv training_logs/training_log_<ts>.csv \
  --window 25 \
  --plot
# -> writes <csv>.png with rolling reward / success / steps over trajectories
```

---

## 3. Section 4.5 — Held-out evaluation

### Inputs

- The three checkpoints to compare:
  - `btrabucco/Insta-Qwen3-1.7B-SFT` (Base SFT, no LoRA — auto-loaded by `evaluate_checkpoint.py --compare`)
  - `checkpoints/checkpoint_trajectory_600/` (RL-on-raw)
  - `checkpoints_feasible/checkpoint_trajectory_500/` (RL-on-filtered)
- The held-out test CSV: `feasibility_results/feasible_sample_20260424_124844.csv` (200 feasibility-filtered tasks).
- Same Vertex-AI/Gemini setup as training.
- An L4/T4 GPU VM.

### Run the full evaluation

```bash
ssh into the VM, then:
cd ~/final-year-project
# Watchdog should already be up; if not:
screen -S watchdog -dm bash ~/final-year-project/gcp/playwright_watchdog.sh

# Filtered-only eval (~30h):
bash eval/run_full_eval.sh

# OR include the small unfiltered validation experiment (~+7h):
RUN_UNFILTERED=1 N_UNFILTERED=30 bash eval/run_full_eval.sh
```

The script runs 4 invocations of `evaluate_checkpoint.py` (or 2 if unfiltered is disabled), one per (checkpoint × test set) cell-pair. Same seed=42 across all invocations so cross-checkpoint comparisons are paired.

### Analyze and render Table 1 + Figure 3

```bash
python eval/analyze_eval_results.py \
  --run_dir eval_results/run_<timestamp>
# -> stdout: Markdown Table 1 with means and 95% bootstrap CIs
# -> writes eval_results/run_<ts>/summary.md   (paste-ready into the report)
# -> writes eval_results/run_<ts>/summary.csv  (data for downstream plotting)
# -> writes eval_results/run_<ts>/figure3.{png,pdf}  (grouped bar chart)
```

The analyzer computes:
- Per-cell mean ± 95% bootstrap CI (10,000 resamples) for reward and success
- Paired Wilcoxon p-values on per-task reward differences
- McNemar p-values on per-task binary success indicators
- Mean delta-reward bootstrap CIs for the four key comparisons:
  - RL-on-raw vs Base SFT (per eval set)
  - RL-on-filtered vs Base SFT (per eval set)
  - RL-on-filtered vs RL-on-raw (per eval set) — the **incremental cleaning attribution**

If the unfiltered invocations are absent (default), the table collapses to filtered-only columns automatically.

---

## Hardware and compute footprint

| Experiment | Compute | Wall time | Approx \$ on GCP |
|---|---|---|---|
| §3 Feasibility audit (2,598 tasks) | CPU + Vertex API | ~6h | ~\$5 |
| §4 Continued training (500 trajectories) | 1× L4 + Vertex API | ~30h | ~\$25 |
| §4.5 Held-out eval (filtered only) | 1× L4 + Vertex API | ~30h | ~\$25 |
| §4.5 Held-out eval (+ unfiltered validation) | 1× L4 + Vertex API | +7h | +\$5 |

All on-demand prices in `us-east4-c` as of April 2026. Spot pricing is ~3× cheaper but preemption-prone for L4 in busy zones.

## Notes on non-determinism

- **Trajectory-level RL is not deterministic** under the same seed because (a) the LLM samples actions with `temperature=0.7` + `top_p=0.9`, (b) live websites change state between visits.
- **Vertex Gemini judge calls are not deterministic** even with `temperature=0.5` (the judge has its own internal sampling).
- **Bootstrap CIs** in `analyze_eval_results.py` use seed 0 internally so the *post-hoc analysis* of a fixed result file is reproducible.
- **`SEED=42` everywhere** controls only task selection (which 200 tasks land in your eval slice), not the trajectories themselves.

For these reasons, exact reproduction of our point estimates is not expected. Reproduction of the *direction and magnitude* of the effects (RL-on-filtered > RL-on-raw > Base SFT, with ~2× success rate gain) should hold.

## Citation

See `README.md` for the BibTeX entry.
