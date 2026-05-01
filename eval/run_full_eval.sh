#!/usr/bin/env bash
# Orchestrate the 3 x 2 = 6-cell held-out evaluation described in
# report/main.md §4.5.
#
# 3 checkpoints:
#   - Base SFT (loaded automatically by --compare; it's the LoRA's base)
#   - RL-on-raw    : checkpoints/checkpoint_trajectory_600
#   - RL-on-filtered : checkpoints_feasible/checkpoint_trajectory_500
#
# 2 test sets:
#   - Filtered   : feasibility_results/feasible_sample_20260424_124844.csv
#   - Unfiltered : data/insta-150k-test.csv (random sample, same seed)
#
# Per cell: 200 tasks (filtered) or 150 tasks (unfiltered). Same seed across
# all evaluations so cross-checkpoint comparisons are paired (each task is
# evaluated by every checkpoint, enabling Wilcoxon/McNemar paired tests).
#
# We use evaluate_checkpoint.py's --compare flag, which evaluates the supplied
# adapter checkpoint AND the underlying base SFT model on the same task list.
# That makes one invocation cover 2 cells (e.g. Base SFT + RL-on-filtered on
# Filtered eval). We need 4 invocations total to cover all 6 cells:
#
#   inv 1: --compare with RL-on-raw      on Filtered    -> cells "Base SFT/Filt" + "RL-on-raw/Filt"
#   inv 2: (no --compare) RL-on-filtered on Filtered    -> cell  "RL-on-filtered/Filt"
#   inv 3: --compare with RL-on-raw      on Unfiltered  -> cells "Base SFT/Unfilt" + "RL-on-raw/Unfilt"
#   inv 4: (no --compare) RL-on-filtered on Unfiltered  -> cell  "RL-on-filtered/Unfilt"
#
# Why split it this way: Base SFT is the same model regardless of which
# adapter we point at, so we only need to evaluate it once per test set.
# Pairing it with the RL-on-raw call (inv 1, 3) keeps the cell count minimal.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/final-year-project}"
RL_RAW="${RL_RAW:-checkpoints/checkpoint_trajectory_600}"
RL_FILT="${RL_FILT:-checkpoints_feasible/checkpoint_trajectory_500}"
FILT_CSV="${FILT_CSV:-feasibility_results/feasible_sample_20260424_124844.csv}"
UNFILT_CSV="${UNFILT_CSV:-data/insta-150k-test.csv}"
N_FILTERED="${N_FILTERED:-200}"
N_UNFILTERED="${N_UNFILTERED:-150}"
SEED="${SEED:-42}"
OUT="${OUT:-eval_results/run_$(date +%Y%m%d_%H%M%S)}"

# Only require Vertex on the VM where this is meant to run.
export JUDGE_USE_VERTEX="${JUDGE_USE_VERTEX:-1}"
export JUDGE_VERTEX_LOCATION="${JUDGE_VERTEX_LOCATION:-us-central1}"

cd "$REPO_DIR"
mkdir -p "$OUT"

# Sanity: required files exist
for d in "$RL_RAW" "$RL_FILT"; do
  [[ -d "$d" ]] || { echo "ERROR: missing checkpoint dir: $d" >&2; exit 1; }
done
for f in "$FILT_CSV" "$UNFILT_CSV"; do
  [[ -f "$f" ]] || { echo "ERROR: missing dataset: $f" >&2; exit 1; }
done
# Playwright server must be up (the watchdog should be running)
if ! curl -sS -m 3 -o /dev/null "http://localhost:3000/" 2>/dev/null; then
  echo "ERROR: Playwright server not responding on :3000" >&2
  echo "       Start it with: bash start_playwright_server.sh" >&2
  echo "       (or ensure the watchdog screen is running)" >&2
  exit 1
fi

run () {
  local label="$1"
  shift
  echo
  echo "===================================================================="
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) — $label"
  echo "===================================================================="
  printf '  %s\n' "$@"
  echo "===================================================================="
  "$@"
}

# inv 1: Base SFT + RL-on-raw on Filtered
run "Inv 1/4: Base SFT + RL-on-raw on Filtered" \
  python3 evaluate_checkpoint.py \
    --checkpoint_dir "$RL_RAW" \
    --dataset "$FILT_CSV" \
    --sample_size "$N_FILTERED" \
    --seed "$SEED" \
    --compare \
    --output_dir "$OUT/inv1_filtered_base_and_raw"

# inv 2: RL-on-filtered on Filtered
run "Inv 2/4: RL-on-filtered on Filtered" \
  python3 evaluate_checkpoint.py \
    --checkpoint_dir "$RL_FILT" \
    --dataset "$FILT_CSV" \
    --sample_size "$N_FILTERED" \
    --seed "$SEED" \
    --output_dir "$OUT/inv2_filtered_rl_filtered"

# inv 3: Base SFT + RL-on-raw on Unfiltered
run "Inv 3/4: Base SFT + RL-on-raw on Unfiltered" \
  python3 evaluate_checkpoint.py \
    --checkpoint_dir "$RL_RAW" \
    --dataset "$UNFILT_CSV" \
    --sample_size "$N_UNFILTERED" \
    --seed "$SEED" \
    --compare \
    --output_dir "$OUT/inv3_unfiltered_base_and_raw"

# inv 4: RL-on-filtered on Unfiltered
run "Inv 4/4: RL-on-filtered on Unfiltered" \
  python3 evaluate_checkpoint.py \
    --checkpoint_dir "$RL_FILT" \
    --dataset "$UNFILT_CSV" \
    --sample_size "$N_UNFILTERED" \
    --seed "$SEED" \
    --output_dir "$OUT/inv4_unfiltered_rl_filtered"

echo
echo "===================================================================="
echo "All 4 invocations done. Result JSONs are under: $OUT"
echo "Run analysis with:  python3 eval/analyze_eval_results.py --run_dir $OUT"
echo "===================================================================="
