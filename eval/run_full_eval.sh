#!/usr/bin/env bash
# Orchestrate the held-out evaluation: Base SFT vs RL-on-filtered, filtered
# eval set only. One invocation of evaluate_checkpoint.py with --compare runs
# both arms on the same task list (paired comparison). Same seed across the
# two arms means per-task results are aligned for Wilcoxon / McNemar tests.
#
# Outputs (under $OUT):
#   checkpoint_results_<ts>.json  -- RL-on-filtered (LoRA-adapted) trajectories
#   base_results_<ts>.json        -- Base SFT (no adapter) trajectories

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/final-year-project}"
# Default points at the checkpoint that scripts/download_checkpoint.sh fetches
# from the project's GitHub Release. If you trained your own checkpoint via
# gcp/launch_training.sh, point this at e.g. checkpoints_feasible/checkpoint_trajectory_<N>
RL_CHECKPOINT="${RL_CHECKPOINT:-checkpoints_feasible/final_checkpoint}"
TEST_CSV="${TEST_CSV:-data/doable_tasks_eval.csv}"
N_TASKS="${N_TASKS:-100}"
SEED="${SEED:-42}"
OUT="${OUT:-eval_results/run_$(date +%Y%m%d_%H%M%S)}"

# Vertex AI is the default judge backend on the VM.
export JUDGE_USE_VERTEX="${JUDGE_USE_VERTEX:-1}"
export JUDGE_VERTEX_LOCATION="${JUDGE_VERTEX_LOCATION:-us-central1}"

cd "$REPO_DIR"
mkdir -p "$OUT"

# Sanity: required files exist
if [[ ! -d "$RL_CHECKPOINT" ]]; then
  echo "ERROR: missing checkpoint dir: $RL_CHECKPOINT" >&2; exit 1
fi
if [[ ! -f "$TEST_CSV" ]]; then
  echo "ERROR: missing dataset: $TEST_CSV" >&2; exit 1
fi
# Playwright server must be up (the watchdog should be running)
if ! curl -sS -m 3 -o /dev/null "http://localhost:3000/" 2>/dev/null; then
  echo "ERROR: Playwright server not responding on :3000" >&2
  echo "       Start it with: bash start_playwright_server.sh" >&2
  echo "       (or ensure the watchdog screen is running)" >&2
  exit 1
fi

echo "===================================================================="
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) -- held-out eval"
echo "  checkpoint : $RL_CHECKPOINT"
echo "  test csv   : $TEST_CSV"
echo "  tasks      : $N_TASKS  (seed=$SEED, paired)"
echo "  output     : $OUT"
echo "===================================================================="

python3 evaluate_checkpoint.py \
  --checkpoint_dir "$RL_CHECKPOINT" \
  --dataset "$TEST_CSV" \
  --sample_size "$N_TASKS" \
  --seed "$SEED" \
  --compare \
  --output_dir "$OUT"

echo
echo "===================================================================="
echo "Eval done. Result JSONs are under: $OUT"
echo "Run analysis with:  python3 eval/analyze_eval_results.py --run_dir $OUT"
echo "===================================================================="
