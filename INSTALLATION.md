# Installation

This project has two installation paths: **local development** (small experiments, eval analysis, figure rendering) and **GCP VM** (full training runs, large-scale eval). For reproducing the experiments we ran, follow the GCP VM path.

---

## Path 1 — Local development

For the lighter parts: editing code, generating figures from existing CSVs, running the analysis scripts.

### Prerequisites

- Python 3.10
- Node.js 18+ (only if you want to run the Playwright server locally)
- ~10 GB free disk for the InSTA dataset and dependencies

### Setup

```bash
# Create a virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install the local insta package (provides judge prompts, configs)
pip install -e .

# Fetch the InSTA train split (~102 MB; the test split ships with the repo)
python data/download.py

# Optional: only if running browser sessions locally
playwright install chromium
cd javascript/server && npm install && cd ../..
```

### Verify

```bash
python -c "
import torch, transformers, peft, bitsandbytes
print(f'torch {torch.__version__}, cuda={torch.cuda.is_available()}')
print(f'transformers {transformers.__version__}')
print(f'peft {peft.__version__}')
print(f'bitsandbytes {bitsandbytes.__version__}')
"
```

For just rendering figures or running analysis, you don't need GPU/CUDA — the analysis scripts (`feasibility_results/figure1_dataset_decay.py`, `eval/analyze_eval_results.py`, `scripts/monitor_training.py`) all run on CPU.

---

## Path 2 — GCP VM (for training and full eval)

This is the path we used for the actual training run. The L4 + system-Python deep-learning image setup avoids a number of pitfalls (see notes below).

### Provision a VM

Prerequisites on your local machine:
- `gcloud` CLI authenticated to a GCP project with billing enabled
- GPU quota: `GPUS_ALL_REGIONS ≥ 1` (request via Console → IAM → Quotas)
- Vertex AI User role granted to the default Compute Engine service account (so the judge calls work without an API key)

```bash
# From your local repo root:
bash gcp/provision_vm.sh --dry-run    # review what'll be created
bash gcp/provision_vm.sh              # actually launch (you'll be prompted to confirm)
```

Default config (override via env vars; see `gcp/provision_vm.sh` header):
- `g2-standard-8` + 1× NVIDIA L4
- 150 GB pd-balanced boot disk
- `pytorch-2-9-cu129-ubuntu-2204-nvidia-580` deep-learning image
- On-demand pricing (~\$0.71/hr); set `--provisioning-model=SPOT` (~\$0.22/hr) at your own risk of preemption

### Transfer code

```bash
# Two options. Direct rsync via the gcloud-installed ssh key is fastest:
rsync -avz \
  -e "ssh -i ~/.ssh/google_compute_engine \
         -o StrictHostKeyChecking=accept-new \
         -o UserKnownHostsFile=~/.ssh/google_compute_known_hosts" \
  --exclude='.git/' --exclude='__pycache__/' --exclude='*.pyc' \
  --exclude='.DS_Store' --exclude='checkpoints.zip' \
  --exclude='eval_results/' --exclude='visualization_output/' \
  --exclude='feasibility_results/feasibility_check_2026*.json' \
  --exclude='javascript/server/node_modules/' \
  ./ <VM_USERNAME>@<VM_EXTERNAL_IP>:final-year-project/

# (Or use gcloud compute scp --recurse if you can't get rsync's -e wrapper to work.)
```

### Set up the VM environment

```bash
gcloud compute ssh <INSTANCE_NAME> --zone=<ZONE>
cd ~/final-year-project
bash gcp/setup_vm.sh
```

`setup_vm.sh` is **idempotent** — re-running it after a reboot or zone migration is safe. It:

1. Installs Node.js LTS (the Playwright server is a Node.js app)
2. Runs `npm install` in `javascript/server/`
3. Installs Playwright browsers **without sudo** so they land in the user's cache (the Node server runs as the user; sudo would put them in `/root/.cache` and break)
4. Installs Python deps via `pip` directly into the deep-learning image's system Python (which already ships with `torch + cu129`)
5. Editable-installs the local `insta` package
6. Runs smoke checks (torch+CUDA, transformers, peft, bitsandbytes)

### Start the Playwright server (with watchdog)

```bash
# Start the watchdog first — it monitors port 3000 and auto-restarts the
# Playwright server if it crashes. Without this, transient server failures
# can cause many failed trajectories.
screen -S watchdog -dm bash ~/final-year-project/gcp/playwright_watchdog.sh

# Watchdog will detect the missing server and start it on first check (~20s).
# Or start it manually first if you don't want to wait:
bash ~/final-year-project/start_playwright_server.sh

# Verify
curl -sS -X POST 'http://localhost:3000/start?width=1920&height=1080' -i | head -3
# Expect: HTTP/1.1 200 OK + a session id in the body
```

### Authenticate the judge to use Vertex AI

Two options:

**Option A — VM service account (preferred, no API key in source):**

```bash
# Confirm the SA has the Vertex AI User role:
gcloud projects get-iam-policy <YOUR_PROJECT> --format=json | grep -A1 'aiplatform'

# Default Compute Engine SA needs roles/aiplatform.user. To grant:
gcloud projects add-iam-policy-binding <YOUR_PROJECT> \
  --member="serviceAccount:$(gcloud iam service-accounts list \
                              --filter='email:*-compute@developer.gserviceaccount.com' \
                              --format='value(email)' | head -1)" \
  --role="roles/aiplatform.user"
```

Then set on the VM:

```bash
export JUDGE_USE_VERTEX=1
# JUDGE_VERTEX_PROJECT and JUDGE_VERTEX_LOCATION auto-detect from the
# metadata server — no need to set them on a GCE VM.
```

**Option B — AI Studio API key (fallback, not recommended for shared/published code):**

```bash
export JUDGE_API_KEY="<your Gemini API key from https://aistudio.google.com/apikey>"
```

`judge_integration.py` auto-selects between the two based on env vars. Vertex is preferred because (a) it bills against GCP credits rather than a per-key quota, (b) doesn't require committing keys to source.

### (Optional) Hugging Face token for higher download rate limits

```bash
huggingface-cli login --token <your_hf_token>
```

The base model `btrabucco/Insta-Qwen3-1.7B-SFT` is public, so this is purely a "no warning, faster downloads" thing.

### Launch a training run

```bash
# All env vars optional; defaults are sane (500 trajectories, save every 25,
# output to checkpoints_feasible/, screenshots disabled).
NUM_TRAJECTORIES=500 \
CHECKPOINT_DIR=checkpoints_feasible \
TRAIN_CSV=feasibility_results/feasible_sample_20260324_195836.csv \
bash gcp/launch_training.sh
```

This launches a detached `tmux` session named `fyp-train`. Attach with `tmux attach -t fyp-train`, detach with `Ctrl-b d`. Per-trajectory CSV log streams to `training_logs/training_log_<ts>.csv`. The `gcp/launch_training.sh` header documents every overridable env var.

---

## What's in the deep-learning image already

The `pytorch-2-9-cu129-ubuntu-2204-nvidia-580` image ships with:
- Python 3.10, pip
- PyTorch 2.9.1 + CUDA 12.9
- NVIDIA driver 580
- screen, tmux, git, build-essential

We don't use `conda` on the VM — system Python is sufficient and avoids re-downloading a duplicate PyTorch stack.

## Common pitfalls (we hit these so you don't have to)

- **Playwright "Executable doesn't exist at /home/.../.cache/ms-playwright/...".** You ran `sudo playwright install`. The browser landed in `/root/.cache`. Re-run **without sudo** so it lands in the user's cache. `setup_vm.sh` does this correctly.
- **Vertex 403 PERMISSION_DENIED on `aiplatform.endpoints.predict`.** The default Compute Engine service account doesn't have the Vertex AI User role. See "Authenticate the judge to use Vertex AI" above.
- **`set -u` in the launch script: "JUDGE_VERTEX_PROJECT: unbound variable".** Already fixed — `gcp/launch_training.sh` defaults the var to empty before referencing it.
- **Playwright server crashes mid-training, training keeps producing `trajectory_failed` rows.** Use the watchdog (`gcp/playwright_watchdog.sh`). It SIGSTOPs the python trainer during outages so trajectories don't fail in cascade.
- **bash 3.2 incompatibility on macOS launches.** `provision_vm.sh` and `teardown.sh` use `${var,,}` lowercasing — replaced with regex matches that work on bash 3.2+.
- **Spot preemption.** Spot T4/L4 in `us-central1-c` was unstable when we ran. We migrated to on-demand L4 in `us-east4-c` via snapshot. See "Provision a VM" overrides.

## Troubleshooting

### CUDA out of memory
Already optimized: 4-bit NF4 quantization, gradient checkpointing, micro-batching, LoRA (only 1.69% of params trainable). If still tight, in `pipeline_in_steps.py`:
- Lower `MAX_TRAJECTORY_STEPS` from 20 → 12
- Lower `JUDGE_LAST_OBS` and `JUDGE_LAST_ACTIONS` from 50 → 20

### `bitsandbytes` import errors on Windows
Use WSL2. `bitsandbytes` Windows builds are unreliable.

### `ImportError` for `insta.*` modules
Run from the project root, and ensure `pip install -e .` was run.
