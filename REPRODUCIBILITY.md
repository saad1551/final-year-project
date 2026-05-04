# Reproducibility

This document gives the exact commands to reproduce each experiment. Three sections, one per experiment:

1. **Section 3 — Feasibility audit** (Figure 1)
2. **Section 4 — RL training** (training run + checkpoints)
3. **Section 4.5 — Held-out evaluation** (Table 1, Figure 3)

All commands assume you have followed `INSTALLATION.md`. For sections 2 and 3, a GCP VM with an L4 (or T4 — slightly slower) GPU is required.

---

## For reviewers — fastest path to running the held-out evaluation

If you only want to verify our headline result (Base SFT vs RL-on-filtered on the held-out set), you do **not** need to retrain. Skip Section 2 and:

1. Follow `INSTALLATION.md` Path 2 to provision a VM and set up the environment.
2. From the repo root on the VM, run:
   ```bash
   bash scripts/download_checkpoint.sh    # fetches our LoRA adapter (~21 MB) from the GitHub Release
   bash eval/run_full_eval.sh             # runs the held-out eval (~20h on an L4)
   python eval/analyze_eval_results.py --run_dir eval_results/run_<timestamp>
   ```

The download script extracts to `checkpoints_feasible/final_checkpoint/`, which is exactly where `eval/run_full_eval.sh` expects to find the LoRA adapter. The base SFT model (the other arm of the comparison) is auto-downloaded from Hugging Face on first run.

If you want to retrain from scratch and verify §4 first, follow Section 2 below before Section 3.

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

The 28.3% headline feasibility number comes from **a separate, larger run** that processed 2,598 tasks (parsed log preserved at `feasibility_results/v2_log_parsed.csv`, rendered as `feasibility_results/figure1_dataset_decay.{png,pdf}`). To reproduce that exact run is not feasible (Gemini's `url_context` tool is non-deterministic; specific website states have changed since), but the methodology and seed are documented so subsequent runs are interpretable.

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

### Alternative: lenient doable-task collector

`filter_doable_tasks.py` and `filter_doable_tasks_eval.py` are two additional scripts that take a different approach to the same problem. They call Gemini with both the `url_context` and `google_search` tools and ask for a single binary `doable / not doable` decision instead of the strict four-way classification used above. The prompt is intentionally more permissive ("be decisive — if the website loads and the general type of action is still possible, mark it as doable"), so they produce a larger pool of accepted tasks per Gemini call.

These were **not** used to produce the §3 headline (28.3% feasibility) — that number comes from `sample_feasible_tasks.py` with the strict four-way schema and confidence ≥ 0.95. The doable-task scripts are kept here as a faster, looser alternative for collecting training pools when paper-grade strictness isn't needed.

```bash
# Collect 200 doable tasks from the test split (default target):
python filter_doable_tasks_eval.py --target 200 --output data/doable_tasks_eval.csv

# Or pull a larger training pool from the train split:
python filter_doable_tasks.py --target 2000 --output data/doable_tasks.csv
```

Both read `GEMINI_API_KEY` or `GOOGLE_API_KEY` from the environment.

---

## 2. Section 4 — RL training

### Inputs

- The base SFT model: `btrabucco/Insta-Qwen3-1.7B-SFT` (auto-downloaded on first run).
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
NUM_TRAJECTORIES=500 \
CHECKPOINT_DIR=checkpoints_feasible \
MIN_REWARD=0.0 \
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

### Resume training after interruption

If training is interrupted (preemption, manual stop, crash), you can resume from the last saved checkpoint:

```bash
# E.g. if checkpoint_trajectory_300 is the latest one saved, target 500 total:
RESUME_FROM=checkpoints_feasible/checkpoint_trajectory_300 \
NUM_TRAJECTORIES=500 \
CHECKPOINT_DIR=checkpoints_feasible \
MIN_REWARD=0.0 \
bash gcp/launch_training.sh
```

The script auto-resumes from the checkpoint's metadata (no need to set `RESUME_DATASET_IDX` after the first run — it's preserved in the checkpoint).

### Reproduce Figure 2 (training progress)

After training, with the CSV available locally:

```bash
python scripts/monitor_training.py \
  --csv training_logs/training_log_<ts>.csv \
  --window 25 \
  --plot
# -> writes <csv>.png with rolling reward / success / steps over trajectories
```

---

## 3. Section 4.5 — Held-out evaluation

### Inputs

- The two checkpoints to compare:
  - `btrabucco/Insta-Qwen3-1.7B-SFT` (Base SFT, no LoRA — auto-loaded by `evaluate_checkpoint.py --compare`)
  - A LoRA adapter checkpoint. Two options for getting it onto the eval VM:
    - **Use ours** (recommended for reviewers): `bash scripts/download_checkpoint.sh` fetches it from the GitHub Release and extracts to `checkpoints_feasible/final_checkpoint/`. ~21 MB download, ~35 MB extracted.
    - **Use one you trained yourself**: produced by §2 at `checkpoints_feasible/checkpoint_trajectory_<N>/`. See "Selecting which checkpoint to evaluate" below for picking the best one.
- The held-out test CSV: `data/doable_tasks_eval.csv` (200 feasibility-filtered tasks produced by `filter_doable_tasks_eval.py` — the lenient-binary collector documented in §1; ships in the repo). The eval below samples 100 of them via `--sample_size 100 --seed 42`, which gives a deterministic and reproducible per-task list.
- Same Vertex-AI/Gemini setup as training.
- An L4/T4 GPU VM.

### Where checkpoints live

`gcp/launch_training.sh` writes LoRA adapter checkpoints to `$CHECKPOINT_DIR/checkpoint_trajectory_<N>/` (default `CHECKPOINT_DIR=checkpoints_feasible`) every `SAVE_EVERY` (default 25) trajectories. After a 500-trajectory run you have `checkpoint_trajectory_25/`, `_50/`, …, `_500/`. Each directory is a small (~70 MB) PEFT adapter — `adapter_config.json` + `adapter_model.safetensors` + tokenizer files; the base SFT weights are *not* included.

If your training run was on a different machine than your eval VM, copy the chosen checkpoint directory over and place it under `<repo_root>/checkpoints_feasible/` on the eval VM, e.g.:

```bash
# On the machine where training ran:
rsync -avz checkpoints_feasible/checkpoint_trajectory_300/ \
  <user>@<eval-vm>:final-year-project/checkpoints_feasible/checkpoint_trajectory_300/
```

Or skip the rsync and pass the checkpoint path directly to the eval via `RL_CHECKPOINT=<absolute-or-relative-path> bash eval/run_full_eval.sh`.

### Selecting which checkpoint to evaluate

Two strategies, in order of decreasing rigor:

1. **Best-checkpoint search (preferred).** Pick K candidate trajectory IDs (default `250 300 325`) and evaluate each on the same small held-out subset with the same seed. Compare paired rewards / success rates and pick the winner.

   ```bash
   CKPTS="250 300 325" \
   N_TASKS=25 \
   bash eval/best_checkpoint_search.sh
   # -> writes eval_results/best_ckpt_search_<ts>/ckpt_<N>/checkpoint_results_<ts>.json
   #
   # Then analyze:
   python eval/best_checkpoint_compare.py --run_dir eval_results/best_ckpt_search_<ts>
   # -> prints per-checkpoint stats + pairwise paired comparisons + a recommendation
   ```

   ~6 hours for the default 3 checkpoints × 25 tasks on an L4. The output recommends the trajectory ID with the highest mean reward (ties broken by success rate) — that's the one to feed into the full eval.

2. **Just use the latest.** If you don't have time for a search, evaluate the final checkpoint (e.g. `checkpoint_trajectory_500`). Caveat: late-stage RL can drift, so the latest isn't always the best.

### Run the full evaluation

```bash
ssh into the VM, then:
cd ~/final-year-project
# Watchdog should already be up; if not:
screen -S watchdog -dm bash ~/final-year-project/gcp/playwright_watchdog.sh

# Run the filtered held-out eval (~20h on an L4):
bash eval/run_full_eval.sh
# Or point at a specific checkpoint trajectory (e.g. the one selected by
# the best-checkpoint search):
RL_CHECKPOINT=checkpoints_feasible/checkpoint_trajectory_300 \
  bash eval/run_full_eval.sh
# Other overrides: TEST_CSV=<path>  N_TASKS=<N>  SEED=<N>  OUT=<dir>
```

This wraps `evaluate_checkpoint.py --compare`, which runs the LoRA-adapted checkpoint and the base SFT model on the same task set in sequence. Same seed across both arms means the per-task comparisons are paired. Two output files are produced under `$OUT`:
- `checkpoint_results_<ts>.json` — RL-on-filtered trajectories
- `base_results_<ts>.json` — Base SFT trajectories

### Analyze and render Table 1 + Figure 3

```bash
python eval/analyze_eval_results.py \
  --run_dir eval_results/run_<timestamp>
# -> stdout: Markdown Table 1 with means and 95% bootstrap CIs
# -> writes eval_results/run_<ts>/summary.md   (Markdown table of the results)
# -> writes eval_results/run_<ts>/summary.csv  (data for downstream plotting)
# -> writes eval_results/run_<ts>/figure3.{png,pdf}  (bar chart)
```

The analyzer computes:
- Per-cell mean ± 95% bootstrap CI (10,000 resamples) for reward and success
- Paired Wilcoxon p-value on per-task reward differences
- McNemar p-value on per-task binary success indicators
- Mean delta-reward with 95% bootstrap CI for the headline comparison: RL-on-filtered vs Base SFT.

---

## Hardware and compute footprint

| Experiment | Compute | Wall time | Approx \$ on GCP |
|---|---|---|---|
| §3 Feasibility audit (2,598 tasks) | CPU + Vertex API | ~6h | ~\$5 |
| §4 RL training (500 trajectories) | 1× L4 + Vertex API | ~30h | ~\$25 |
| §4.5 Held-out eval (Base SFT vs RL-on-filtered, 100 tasks each via `--compare`) | 1× L4 + Vertex API | ~20h | ~\$15 |

All on-demand prices in `us-east4-c` as of April 2026. Spot pricing is ~3× cheaper but preemption-prone for L4 in busy zones.

## Notes on non-determinism

- **Trajectory-level RL is not deterministic** under the same seed because (a) the LLM samples actions with `temperature=0.7` + `top_p=0.9`, (b) live websites change state between visits.
- **Vertex Gemini judge calls are not deterministic** even with `temperature=0.5` (the judge has its own internal sampling).
- **Bootstrap CIs** in `analyze_eval_results.py` use seed 0 internally so the *post-hoc analysis* of a fixed result file is reproducible.
- **`SEED=42` everywhere** controls only task selection (which 100 tasks land in your eval slice), not the trajectories themselves.

For these reasons, exact reproduction of our point estimates is not expected. Reproduction of the *direction and magnitude* of the effect (RL-on-filtered > Base SFT) should hold.

## Citation

See `README.md` for the BibTeX entry.
