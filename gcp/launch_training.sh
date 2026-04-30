#!/usr/bin/env bash
# Run ON the VM (inside ~/final-year-project) to launch training under tmux.
# Resumes from final_checkpoint and trains on the 2,068 feasible-train tasks.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/final-year-project}"
ENV_NAME="${ENV_NAME:-insta}"
SESSION="${SESSION:-fyp-train}"
RESUME_FROM="${RESUME_FROM:-checkpoints/checkpoint_trajectory_600}"
TRAIN_CSV="${TRAIN_CSV:-feasibility_results/feasible_sample_20260324_195836.csv}"
NUM_TRAJECTORIES="${NUM_TRAJECTORIES:-500}"
SAVE_EVERY="${SAVE_EVERY:-25}"
REF_KL_FREQUENCY="${REF_KL_FREQUENCY:-5}"
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

# Build the inner command. Activate conda, start playwright server in
# background, then run the trainer.
INNER_CMD=$(cat <<EOF
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate $ENV_NAME

echo "[$(date)] starting playwright server"
nohup bash start_playwright_server.sh > playwright_server.log 2>&1 &
sleep 5

echo "[$(date)] starting training"
echo "  csv         : $TRAIN_CSV"
echo "  resume_from : $RESUME_FROM"
echo "  trajectories: $NUM_TRAJECTORIES"

export JUDGE_USE_VERTEX="$JUDGE_USE_VERTEX"
export JUDGE_VERTEX_LOCATION="$JUDGE_VERTEX_LOCATION"
if [[ -n "${JUDGE_VERTEX_PROJECT:-}" ]]; then
  export JUDGE_VERTEX_PROJECT="$JUDGE_VERTEX_PROJECT"
fi

python pipeline_in_steps.py \
  --train_csv "$TRAIN_CSV" \
  --resume_from "$RESUME_FROM" \
  --num_trajectories $NUM_TRAJECTORIES \
  --save_every $SAVE_EVERY \
  --algorithm ppo \
  --ref_kl_frequency $REF_KL_FREQUENCY \
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
