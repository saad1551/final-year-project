#!/usr/bin/env bash
# Run ON the VM (inside ~/final-year-project) to launch training under tmux.
# Trains on the 2,068 feasible-train tasks starting from the base SFT model.
# Set RESUME_FROM=<path> to warm-start from an existing LoRA checkpoint.

set -euo pipefail

# Auto-detect repo root from the script's location; override with REPO_DIR=...
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_NAME="${ENV_NAME:-insta}"
SESSION="${SESSION:-fyp-train}"
# Empty by default = train a fresh LoRA adapter on top of the base SFT model.
# Set RESUME_FROM=<path-to-checkpoint-dir> to warm-start from an existing LoRA
# adapter (e.g. to resume an interrupted run).
RESUME_FROM="${RESUME_FROM:-}"
TRAIN_CSV="${TRAIN_CSV:-feasibility_results/feasible_sample_20260324_195836.csv}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints_feasible}"
# num_trajectories is the *target* trajectory_id (not an increment). When
# starting from scratch the counter starts at 1, so NUM_TRAJECTORIES=500
# means trajectories 1..500. When resuming, the counter continues from
# whatever the checkpoint's metadata says.
NUM_TRAJECTORIES="${NUM_TRAJECTORIES:-500}"
SAVE_EVERY="${SAVE_EVERY:-25}"
REF_KL_FREQUENCY="${REF_KL_FREQUENCY:-5}"
# min_reward_for_update threshold: trajectories with reward below this are
# skipped from the RL update. Default 0.0 = train on every completed
# trajectory. The 0.1 default in the python script was set when the dataset
# was raw InSTA (full of broken tasks); now that we're on the
# feasibility-filtered CSV, low rewards reflect real agent failures whose
# negative gradient is signal, not noise. PPO clip + ref-KL + grad-norm
# already bound destabilization.
MIN_REWARD="${MIN_REWARD:-0.0}"
# Optional override: where in the CSV to start. Useful when the warm-start
# checkpoint was trained on a different dataset (e.g. insta-150k-train.csv)
# and the inferred resume index doesn't match the new CSV. Leave unset to
# let the script auto-resume from checkpoint metadata.
RESUME_DATASET_IDX="${RESUME_DATASET_IDX:-}"
LOG_FILE="${LOG_FILE:-training_$(date +%Y%m%d_%H%M%S).log}"

cd "$REPO_DIR"

if ! command -v tmux >/dev/null; then
  echo "Installing tmux..."
  sudo apt-get update -y && sudo apt-get install -y tmux
fi

# Pre-flight: dataset and (optional) checkpoint sanity
if [[ ! -f "$TRAIN_CSV" ]]; then
  echo "ERROR: training CSV not found at $TRAIN_CSV" >&2; exit 1
fi
if [[ -n "$RESUME_FROM" && ! -d "$RESUME_FROM" ]]; then
  echo "ERROR: RESUME_FROM was set but directory not found: $RESUME_FROM" >&2; exit 1
fi

# Vertex AI is the default judge backend on the VM. The credit pool the user
# wants billed (GenAI app builder) only applies via Vertex / Gemini products,
# not Compute Engine. JUDGE_USE_VERTEX=1 + JUDGE_VERTEX_PROJECT (auto-detected
# via metadata server / gcloud) tells judge_integration.py to use Vertex.
export JUDGE_USE_VERTEX="${JUDGE_USE_VERTEX:-1}"
export JUDGE_VERTEX_LOCATION="${JUDGE_VERTEX_LOCATION:-us-central1}"
# JUDGE_VERTEX_PROJECT is optional — auto-detected from gcloud config or the
# GCE metadata server if unset.
export JUDGE_VERTEX_PROJECT="${JUDGE_VERTEX_PROJECT:-}"

# Disable per-step screenshot saving. Over 500 trajectories with ~10 steps
# each the visualization_output dir balloons to several GB and adds disk
# IO overhead per step. Set DISABLE_SCREENSHOTS=0 to re-enable for debugging.
export DISABLE_SCREENSHOTS="${DISABLE_SCREENSHOTS:-1}"

# Compose extra args for the python invocation (resolved by the outer shell
# before being baked into the inner heredoc).
EXTRA_ARGS=""
if [[ -n "$RESUME_FROM" ]]; then
  EXTRA_ARGS="$EXTRA_ARGS --resume_from $RESUME_FROM"
fi
if [[ -n "${RESUME_DATASET_IDX:-}" ]]; then
  EXTRA_ARGS="$EXTRA_ARGS --resume_dataset_idx $RESUME_DATASET_IDX"
fi

# Build the inner command. The DL VM image uses system Python directly
# (no conda env). Playwright server lifecycle is managed by the watchdog
# (gcp/playwright_watchdog.sh) — start it separately in a 'watchdog' screen
# before launching training. Here we just verify port 3000 is live.
INNER_CMD=$(cat <<EOF
set -euo pipefail

echo "[\$(date)] checking playwright server"
deadline=\$((SECONDS + 30))
while ! curl -sS -m 3 -o /dev/null http://localhost:3000/ 2>/dev/null; do
  if (( SECONDS > deadline )); then
    echo "ERROR: playwright server did not respond on :3000 within 30s" >&2
    echo "       Make sure the watchdog screen is running:" >&2
    echo "       screen -S watchdog -dm bash gcp/playwright_watchdog.sh" >&2
    exit 1
  fi
  sleep 2
done
echo "[\$(date)] playwright server up"

echo "[$(date)] starting training"
echo "  csv         : $TRAIN_CSV"
echo "  resume_from : ${RESUME_FROM:-<none, training fresh LoRA from base SFT>}"
echo "  trajectories: $NUM_TRAJECTORIES"

# JUDGE_USE_VERTEX, JUDGE_VERTEX_LOCATION, JUDGE_VERTEX_PROJECT are already
# exported by the outer script and inherited via tmux's env passthrough.

python3 pipeline_in_steps.py \
  --train_csv "$TRAIN_CSV" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --num_trajectories $NUM_TRAJECTORIES \
  --save_every $SAVE_EVERY \
  --algorithm ppo \
  --ref_kl_frequency $REF_KL_FREQUENCY \
  --min_reward $MIN_REWARD \
  $EXTRA_ARGS \
  2>&1 | tee "$LOG_FILE"
EOF
)

# Launch detached tmux so the SSH session can drop without killing training
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists. Attach with: tmux attach -t $SESSION"
  exit 1
fi

tmux new-session -d -s "$SESSION" "$INNER_CMD; bash"

echo "===================================================================="
echo "Training launched in tmux session '$SESSION'."
echo "  Attach   : tmux attach -t $SESSION    (Ctrl-b d to detach)"
echo "  Tail log : tail -f $REPO_DIR/$LOG_FILE"
echo "  Kill     : tmux kill-session -t $SESSION"
echo "===================================================================="
