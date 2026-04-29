#!/usr/bin/env bash
# Run LOCALLY to push code, the feasible-train CSV, and the warm-start
# checkpoint to the VM. Uses gcloud compute scp (built on top of ssh).

set -euo pipefail

INSTANCE_NAME="${INSTANCE_NAME:-fyp-train-t4}"
ZONE="${ZONE:-us-central1-c}"
REMOTE_DIR="${REMOTE_DIR:-final-year-project}"
LOCAL_DIR="${LOCAL_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

echo "===================================================================="
echo "Transferring project to $INSTANCE_NAME ($ZONE):~/$REMOTE_DIR"
echo "===================================================================="

# 1. Code: clone via git on the VM is cleaner than rsyncing the working tree.
#    But we want this exact branch. Easiest: rsync only the tracked + selected
#    files we actually need.

EXCLUDES=(
  --exclude '.git'
  --exclude '__pycache__'
  --exclude '.DS_Store'
  --exclude 'checkpoints.zip'
  --exclude 'feasibility_results/feasibility_check_*.json'
  --exclude 'feasibility_results/*.log'
  --exclude '*.pyc'
  --exclude 'eval_results'
  --exclude 'wandb'
)

# Use gcloud's recommended "scp recursively" via tar streaming over ssh.
# (gcloud compute scp --recurse re-transfers everything every run; rsync over
#  the gcloud-compute-ssh wrapper gives incremental sync.)

GCLOUD_SSH="gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --tunnel-through-iap"

# Make remote dir
$GCLOUD_SSH --command "mkdir -p ~/$REMOTE_DIR" -- -q

# Build rsync command using gcloud as the ssh transport
RSYNC_RSH="gcloud compute ssh --zone=$ZONE --tunnel-through-iap --"
echo "Rsync code (excluding heavy/transient files)..."
rsync -avz "${EXCLUDES[@]}" \
  -e "$RSYNC_RSH" \
  "$LOCAL_DIR/" "$INSTANCE_NAME:~/$REMOTE_DIR/"

echo ""
echo "Transfer summary:"
$GCLOUD_SSH --command "du -sh ~/$REMOTE_DIR && ls ~/$REMOTE_DIR/checkpoints/ 2>/dev/null | head -10" -- -q

echo ""
echo "Done. Next: ssh in and run bash gcp/setup_vm.sh"
