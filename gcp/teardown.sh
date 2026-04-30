#!/usr/bin/env bash
# Run LOCALLY to either STOP (preserves disk, no compute cost) or DELETE
# (frees everything, no further charges) the training VM.
#
# Usage:
#   bash gcp/teardown.sh stop      # default; preserves disk for resume
#   bash gcp/teardown.sh delete    # nukes VM AND boot disk

set -euo pipefail

INSTANCE_NAME="${INSTANCE_NAME:-fyp-train-t4}"
ZONE="${ZONE:-us-central1-c}"
ACTION="${1:-stop}"

case "$ACTION" in
  stop)
    echo "Stopping $INSTANCE_NAME (preserves disk; no compute cost while stopped)..."
    gcloud compute instances stop "$INSTANCE_NAME" --zone="$ZONE"
    echo "Stopped. Disk still costs ~\$0.55/day. To remove fully: $0 delete"
    ;;
  delete)
    read -r -p "Delete VM AND boot disk for '$INSTANCE_NAME'? This is irreversible. [y/N] " confirm
    if [[ ! "$confirm" =~ ^[yY] ]]; then
      echo "Aborted."
      exit 1
    fi
    gcloud compute instances delete "$INSTANCE_NAME" --zone="$ZONE" --quiet
    echo "Deleted. No further charges."
    ;;
  *)
    echo "Usage: $0 {stop|delete}" >&2
    exit 1
    ;;
esac
