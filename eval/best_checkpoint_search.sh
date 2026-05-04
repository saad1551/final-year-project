#!/usr/bin/env bash
# Pass-1 best-checkpoint search.
#
# Compares K checkpoints on the same paired held-out subset to determine
# which one to use for the full filtered eval. The point is to disambiguate
# "Q4 training-set drop = real regression" from "Q4 = task-difficulty
# variance" — see report §4.5 / RLHF literature on checkpoint selection.
#
# Default: 3 checkpoints x 25 tasks = 75 trajectories (~6.25h on L4).
# Same --seed everywhere → paired comparisons.

set -euo pipefail

# Auto-detect repo root from the script's location; override with REPO_DIR=...
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CKPT_DIR="${CKPT_DIR:-checkpoints_feasible}"
EVAL_CSV="${EVAL_CSV:-data/doable_tasks_eval.csv}"
N_TASKS="${N_TASKS:-25}"
SEED="${SEED:-42}"
OUT="${OUT:-eval_results/best_ckpt_search_$(date +%Y%m%d_%H%M%S)}"
# Space-separated list of checkpoint trajectory ids to compare
CKPTS="${CKPTS:-250 300 325}"

export JUDGE_USE_VERTEX="${JUDGE_USE_VERTEX:-1}"
export JUDGE_VERTEX_LOCATION="${JUDGE_VERTEX_LOCATION:-us-central1}"
export DISABLE_SCREENSHOTS="${DISABLE_SCREENSHOTS:-1}"

cd "$REPO_DIR"
mkdir -p "$OUT"

# Pre-flight
for c in $CKPTS; do
  d="$CKPT_DIR/checkpoint_trajectory_$c"
  [[ -d "$d" ]] || { echo "ERROR: missing $d" >&2; exit 1; }
done
[[ -f "$EVAL_CSV" ]] || { echo "ERROR: missing $EVAL_CSV" >&2; exit 1; }
if ! curl -sS -m 3 -o /dev/null http://localhost:3000/ 2>/dev/null; then
  echo "ERROR: Playwright server not on :3000. Start watchdog first." >&2
  exit 1
fi

echo "===================================================================="
echo "Best-checkpoint search"
echo "  checkpoints: $CKPTS"
echo "  tasks/ckpt : $N_TASKS  (seed=$SEED → paired across checkpoints)"
echo "  output     : $OUT"
echo "===================================================================="

i=0
total=$(echo $CKPTS | wc -w)
for c in $CKPTS; do
  i=$((i+1))
  echo
  echo "--- [$i/$total] checkpoint_trajectory_$c  $(date -u +%H:%M:%S) ---"
  python3 evaluate_checkpoint.py \
    --checkpoint_dir "$CKPT_DIR/checkpoint_trajectory_$c" \
    --dataset "$EVAL_CSV" \
    --sample_size "$N_TASKS" \
    --seed "$SEED" \
    --output_dir "$OUT/ckpt_$c" \
    2>&1 | tee "$OUT/ckpt_$c.log" | grep -E "TASK [0-9]+/|Success|Reward|Error" | tail -10 || true
done

echo
echo "===================================================================="
echo "Search complete. Results under: $OUT"
echo "Run analysis:"
echo "  python3 eval/best_checkpoint_compare.py --run_dir $OUT"
echo "===================================================================="
