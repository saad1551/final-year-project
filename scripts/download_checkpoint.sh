#!/usr/bin/env bash
# Download the held-out evaluation checkpoint from the project's GitHub Release.
#
# Run once after cloning. Fetches the LoRA adapter zip and extracts it to
# checkpoints_feasible/final_checkpoint/, which is the default path that
# eval/run_full_eval.sh expects.
#
# All env vars are overridable:
#   REPO=<owner>/<repo>     (default: saad1551/final-year-project)
#   TAG=<release-tag>       (default: v1.0-submission)
#   ASSET=<filename>        (default: final_checkpoint.zip)
#   DEST=<extract-target>   (default: checkpoints_feasible/final_checkpoint)
#
# Idempotent: if the extracted checkpoint dir already contains
# adapter_model.safetensors, the script is a no-op.

set -euo pipefail

REPO="${REPO:-saad1551/final-year-project}"
TAG="${TAG:-v1.0-submission}"
ASSET="${ASSET:-final_checkpoint.zip}"
DEST="${DEST:-checkpoints_feasible/final_checkpoint}"

URL="https://github.com/${REPO}/releases/download/${TAG}/${ASSET}"

# Pre-flight: tools we need
for tool in curl unzip; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "ERROR: '$tool' is required but not found on PATH." >&2
    echo "       Install it (e.g. 'sudo apt-get install -y $tool' or 'brew install $tool')." >&2
    exit 1
  fi
done

# If the checkpoint is already extracted, do nothing.
if [[ -f "$DEST/adapter_model.safetensors" ]]; then
  echo "[download_checkpoint] $DEST/adapter_model.safetensors already present."
  echo "[download_checkpoint] Nothing to do. Remove $DEST/ or set DEST=<other-path> to re-download."
  exit 0
fi

TMPZIP="$(mktemp -t fyp-ckpt-XXXXXX.zip)"
trap 'rm -f "$TMPZIP"' EXIT

echo "[download_checkpoint] fetching:"
echo "    $URL"
curl -fL --progress-bar -o "$TMPZIP" "$URL"

echo "[download_checkpoint] extracting to: $DEST"
mkdir -p "$DEST"
unzip -q -o "$TMPZIP" -d "$DEST"

echo
echo "[download_checkpoint] done. Checkpoint at: $DEST"
ls -1 "$DEST"
