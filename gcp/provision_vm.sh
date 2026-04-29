#!/usr/bin/env bash
# Provision a single T4 spot VM in the cheapest region with available quota.
#
# All knobs are env-var overridable; defaults are the cost-bounded path.
# Run with `--dry-run` (or DRY_RUN=1) to print the gcloud command without launching.
#
# Approximate billing on free trial:
#   T4 spot:           ~$0.11/hr  →  6 days continuous = ~$16
#   100 GB pd-balanced: ~$0.55/day → 6 days            = ~$3.30
#
# After training: run gcp/teardown.sh to stop or delete the VM.

set -euo pipefail

PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
INSTANCE_NAME="${INSTANCE_NAME:-fyp-train-t4}"
ZONE="${ZONE:-us-central1-c}"
MACHINE_TYPE="${MACHINE_TYPE:-n1-standard-4}"
GPU_TYPE="${GPU_TYPE:-nvidia-tesla-t4}"
GPU_COUNT="${GPU_COUNT:-1}"
DISK_SIZE_GB="${DISK_SIZE_GB:-150}"
DISK_TYPE="${DISK_TYPE:-pd-balanced}"
IMAGE_FAMILY="${IMAGE_FAMILY:-pytorch-latest-gpu}"
IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"

DRY_RUN=0
for arg in "$@"; do
  if [[ "$arg" == "--dry-run" ]]; then DRY_RUN=1; fi
done
if [[ "${DRY_RUN_ENV:-${DRY_RUN:-0}}" == "1" ]]; then DRY_RUN=1; fi

if [[ -z "$PROJECT" ]]; then
  echo "ERROR: no project set. Run: gcloud config set project YOUR_PROJECT_ID" >&2
  exit 1
fi

cmd=(
  gcloud compute instances create "$INSTANCE_NAME"
  --project="$PROJECT"
  --zone="$ZONE"
  --machine-type="$MACHINE_TYPE"
  --provisioning-model=SPOT
  --instance-termination-action=STOP
  --accelerator="type=$GPU_TYPE,count=$GPU_COUNT"
  --maintenance-policy=TERMINATE
  --image-family="$IMAGE_FAMILY"
  --image-project="$IMAGE_PROJECT"
  --boot-disk-size="${DISK_SIZE_GB}GB"
  --boot-disk-type="$DISK_TYPE"
  --metadata=install-nvidia-driver=True
  --scopes=cloud-platform
  --tags=fyp-train
)

echo "===================================================================="
echo "Provisioning command:"
echo "===================================================================="
printf '  %s\n' "${cmd[@]}"
echo "===================================================================="
echo "Project        : $PROJECT"
echo "Zone           : $ZONE"
echo "Machine type   : $MACHINE_TYPE  +  ${GPU_COUNT}x $GPU_TYPE  (SPOT)"
echo "Boot disk      : ${DISK_SIZE_GB} GB $DISK_TYPE  ($IMAGE_FAMILY)"
echo "Estimated cost : ~\$0.11/hr GPU + ~\$0.55/day disk"
echo "===================================================================="

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY-RUN: not launching. Re-run without --dry-run to provision."
  exit 0
fi

read -r -p "Launch VM now? [y/N] " confirm
if [[ "${confirm,,}" != "y" && "${confirm,,}" != "yes" ]]; then
  echo "Aborted."
  exit 1
fi

"${cmd[@]}"

echo ""
echo "VM '$INSTANCE_NAME' is up. Next steps:"
echo "  1. SSH in:        gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
echo "  2. Verify GPU:    nvidia-smi"
echo "  3. Run setup:     bash gcp/setup_vm.sh         (after rsync)"
echo "  4. Transfer code: bash gcp/transfer_to_vm.sh   (from local)"
echo "  5. Launch train:  bash gcp/launch_training.sh  (on VM)"
echo ""
echo "When done: bash gcp/teardown.sh   (stops or deletes the VM)"
