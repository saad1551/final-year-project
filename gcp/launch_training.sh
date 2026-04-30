#!/usr/bin/env bash
# Run ON the VM (inside ~/final-year-project) to launch training under tmux.
# Resumes from final_checkpoint and trains on the 2,068 feasible-train tasks.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/final-year-project}"
ENV_NAME="${ENV_NAME:-insta}"
SESSION="${SESSION:-fyp-train}"
RESUME_FROM="${RESUME_FROM:-checkpoints/checkpoint_trajectory_600}"
TRAIN_CSV="${TRAIN_CSV:-feasibility_results/feasible_sample_20260324_195836.csv}"
# Save new checkpoints to a separate dir so they don't collide with the old
# warm-start checkpoints (checkpoints/checkpoint_trajectory_500/600/...).
# When RESUME_DATASET_IDX=0 resets the trajectory counter, names like
# "checkpoint_trajectory_500" would otherwise overwrite the originals.
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints_feasible}"
# num_trajectories is the *target* trajectory_id, not an increment. With
# RESUME_DATASET_IDX=0 the counter resets, so 500 means run trajectories 1..500
# of the feasible CSV from a model warm-started at checkpoint_trajectory_600.
NUM_TRAJECTORIES="${NUM_TRAJECTORIES:-500}"
SAVE_EVERY="${SAVE_EVERY:-25}"
REF_KL_FREQUENCY="${REF_KL_FREQUENCY:-5}"
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

# Pre-flight: dataset and checkpoint sanity
if [[ ! -f "$TRAIN_CSV" ]]; then
  echo "ERROR: training CSV not found at $TRAIN_CSV" >&2; exit 1
fi
if [[ ! -d "$RESUME_FROM" ]]; then
  echo "ERROR: warm-start checkpoint not found at $RESUME_FROM" >&2; exit 1
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

# Compose extra args for the python invocation (resolved by the outer shell
# before being baked into the inner heredoc).
EXTRA_ARGS=""
if [[ -n "${RESUME_DATASET_IDX:-}" ]]; then
  EXTRA_ARGS="--resume_dataset_idx $RESUME_DATASET_IDX"
fi

# Build the inner command. The DL VM image uses system Python directly
# (no conda env), so we just start the Playwright server and run training.
INNER_CMD=$(cat <<EOF
set -euo pipefail

# (Re)start the Playwright server in a detached screen, idempotent.
screen -S playwright -X quit 2>/dev/null || true
sleep 1
echo "[\$(date)] starting playwright server"
bash start_playwright_server.sh
sleep 8
# Sanity-check it's listening
if ! curl -sS -m 3 -o /dev/null -w "%{http_code}" http://localhost:3000/ | grep -qE "^[2-4]"; then
  echo "ERROR: playwright server did not come up on :3000" >&2
  exit 1
fi
echo "[\$(date)] playwright server up"

echo "[$(date)] starting training"
echo "  csv         : $TRAIN_CSV"
echo "  resume_from : $RESUME_FROM"
echo "  trajectories: $NUM_TRAJECTORIES"

# JUDGE_USE_VERTEX, JUDGE_VERTEX_LOCATION, JUDGE_VERTEX_PROJECT are already
# exported by the outer script and inherited via tmux's env passthrough.

python3 pipeline_in_steps.py \
  --train_csv "$TRAIN_CSV" \
  --resume_from "$RESUME_FROM" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --num_trajectories $NUM_TRAJECTORIES \
  --save_every $SAVE_EVERY \
  --algorithm ppo \
  --ref_kl_frequency $REF_KL_FREQUENCY \
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
