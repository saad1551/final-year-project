# GCP training scripts

Scripts to provision a single T4 spot VM, run PPO + LoRA RL training on the 2,068 feasible-train tasks, and tear it down. All paths assume the project lives at `~/final-year-project` on the VM.

## Order of operations

```
# Local (you are here)
bash gcp/provision_vm.sh --dry-run   # review the gcloud command
bash gcp/provision_vm.sh             # actually launch (~$0.11/hr T4 spot)
bash gcp/transfer_to_vm.sh           # rsync code + data + checkpoint

# On the VM
gcloud compute ssh fyp-train-t4 --zone=us-central1-c
cd final-year-project
bash gcp/setup_vm.sh                 # conda env + Playwright

export JUDGE_API_KEY="<your Gemini key>"
bash gcp/launch_training.sh          # detaches into tmux

# When done (back on local)
bash gcp/teardown.sh stop            # pause (cheap)
bash gcp/teardown.sh delete          # nuke (zero ongoing cost)
```

## Cost-bounded defaults

| Resource | Spot price | 6 days | If stopped between runs |
|---|---|---|---|
| n1-standard-4 + T4 | ~$0.11/hr | ~$16 | ~$8 |
| 150 GB pd-balanced | ~$0.55/day | ~$4 | ~$4 |
| Egress | negligible | <$1 | <$1 |
| **Total** | | **~$21** | **~$13** |

Comfortably under the $243 free-trial credit. The $1000 GenAI credit covers Gemini judge API calls during RL rollouts (not Compute Engine).

## Knobs

All scripts accept env vars: `INSTANCE_NAME`, `ZONE`, `MACHINE_TYPE`, `GPU_TYPE`, `DISK_SIZE_GB`, `RESUME_FROM`, `TRAIN_CSV`, `NUM_TRAJECTORIES`, `JUDGE_API_KEY`. Defaults are in each script's header.

## Prerequisites

- `gcloud` authenticated to the right project: `gcloud config set project YOUR_ID`
- Quota: `GPUS_ALL_REGIONS = 1` (request via Console → IAM → Quotas)
- Quota: `PREEMPTIBLE_NVIDIA_T4_GPUS = 1` in `us-central1` (already 1 by default)
