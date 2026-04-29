#!/usr/bin/env bash
# Run ON the VM (inside ~/final-year-project) to launch training under tmux.
# Resumes from final_checkpoint and trains on the 2,068 feasible-train tasks.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/final-year-project}"
ENV_NAME="${ENV_NAME:-insta}"
SESSION="${SESSION:-fyp-train}"
RESUME_FROM="${RESUME_FROM:-checkpoints/final_checkpoint}"
TRAIN_CSV="${TRAIN_CSV:-feasibility_results/feasible_sample_20260324_195836.csv}"
NUM_TRAJECTORIES="${NUM_TRAJECTORIES:-500}"
SAVE_EVERY="${SAVE_EVERY:-25}"
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

# Required env: Gemini judge API key (drawn from the GenAI credit pool).
: "${JUDGE_API_KEY:?Must set JUDGE_API_KEY (Gemini API key) before launching}"

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

export JUDGE_API_KEY="$JUDGE_API_KEY"

python pipeline_in_steps.py \
  --train_csv "$TRAIN_CSV" \
  --resume_from "$RESUME_FROM" \
  --num_trajectories $NUM_TRAJECTORIES \
  --save_every $SAVE_EVERY \
  --algorithm ppo \
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
