# Adaptive Web Interaction: Leveraging Reinforcement Learning for Comprehensive Action Support

This repository contains the code, data, and analysis artifacts for a final-year-project investigating reinforcement-learning fine-tuning of small language models for browser-navigation tasks. The work makes two contributions:

1. **A feasibility audit of the InSTA-150k web-agent benchmark.** Using a Gemini-2.5-Flash judge with URL-context grounding, we classify 2,598 sampled InSTA test tasks into feasible / website-down / content-outdated / uncertain. **Only 28.3% of sampled tasks are reliably feasible.** The breakdown and methodology are in `feasibility_results/` and Section 3 of `report/main.md`.

2. **A continued-RL-training experiment on a feasibility-filtered subset.** We continue PPO + LoRA training of `btrabucco/Insta-Qwen3-1.7B-SFT` on a 2,068-task feasibility-filtered subset and evaluate on a held-out 200-task feasibility-filtered subset, comparing against (a) the base SFT model and (b) a prior checkpoint trained on the unfiltered InSTA-150k-train split. Code in `gcp/`, training pipeline in `pipeline_in_steps.py`, eval pipeline in `eval/`.

## Headline numbers

> **Section 3 — feasibility audit.** 28.3% of InSTA-150k-test tasks are FEASIBLE. 28.6% are WEBSITE_DOWN, 13.0% are CONTENT_LIKELY_OUTDATED, 30.1% are UNCERTAIN. **24.3% of "WEBSITE_DOWN" sites returned an HTTP 2xx**, demonstrating that reachability probing alone is insufficient.

> **Section 4 — continued training.** *(Final numbers TBD; updates after training + held-out eval complete. Latest mid-run quartile reward: Q1 = 0.22 → Q4 = ~0.45 over 150 trajectories — a roughly 2× improvement over training. See `report/main.md` §4.5 for held-out test-set results.)*

## Repository layout

```
.
├── pipeline_in_steps.py       # main training entry point (PPO + LoRA)
├── evaluate_checkpoint.py     # held-out evaluation entry point
├── sample_feasible_tasks.py   # feasibility audit entry point (Section 3)
│
├── src/                       # internal modules — imported as `src.<name>`
│   ├── rl_trainer.py          # RL algorithm implementations
│   ├── judge_integration.py   # Vertex AI Gemini judge wrapper
│   ├── client.py              # browser session client
│   ├── training_logger.py     # per-trajectory CSV logger
│   ├── observability.py       # per-step JSON observability
│   └── utils.py               # shared utilities
│
├── tests/                     # test suites
│   ├── test.py
│   ├── test_rl_mock.py
│   └── conftest.py            # adds repo root to sys.path
│
├── scripts/                   # standalone CLI helpers
│   ├── check_progress.py      # live training-log monitor (VM-aware)
│   ├── monitor_training.py    # render training curves from CSV
│   └── merge_lora.py          # merge LoRA adapter into base model
│
├── eval/                      # held-out evaluation harness
│   ├── run_full_eval.sh       # orchestrate the 3×{1,2}-cell evaluation
│   └── analyze_eval_results.py
│
├── gcp/                       # GCP provisioning + training launch
│   ├── provision_vm.sh
│   ├── setup_vm.sh
│   ├── transfer_to_vm.sh
│   ├── launch_training.sh
│   ├── playwright_watchdog.sh # auto-recovers a crashed Playwright server
│   ├── teardown.sh
│   └── README.md
│
├── feasibility_results/       # Section 3 artifacts
│   ├── feasible_sample_*.csv  # accepted task lists
│   ├── figure1_dataset_decay.{png,pdf}
│   ├── parse_v2_log.py
│   ├── figure1_dataset_decay.py
│   ├── sample_for_sanity_check.py
│   └── sanity_check_sample.md
│
├── report/                    # workshop-paper-style writeup (work in progress)
│   └── main.md
│
├── insta/                     # InSTA package (judge prompts, configs)
├── javascript/server/         # Node.js Playwright server
├── data/                      # datasets (InSTA train/test, see REPRODUCIBILITY.md)
├── checkpoints/               # warm-start LoRA checkpoints (prior training)
├── checkpoints_feasible/      # this paper's continued-training checkpoints
├── agent_prompts/             # prompts used by the agent
├── configs/                   # configuration objects
├── markdown/                  # HTML→Markdown utilities
└── observation_processors/
```

## Getting started

- **Install dependencies and verify:** see `INSTALLATION.md`.
- **Reproduce the headline numbers:** see `REPRODUCIBILITY.md`.
- **Read the writeup:** see `report/main.md`.

## Authors

This project is the joint work of three final-year undergraduate students at *(TBD: institution)*. Author names and contributions:

- *(TBD)*
- *(TBD)*
- *(TBD)*

Supervisor: *(TBD)*

## Citation

If you use this code or build on the dataset feasibility analysis, please cite:

```bibtex
@misc{adaptive-web-interaction-2026,
  title  = {Adaptive Web Interaction: Leveraging Reinforcement Learning for
            Comprehensive Action Support},
  author = {(TBD)},
  year   = {2026},
  note   = {Final-year project, (institution TBD).}
}
```

(BibTeX entry will be updated when the work is published.)

## License

MIT (see `LICENSE`).

## Acknowledgements

This project builds on:
- **InSTA** (Trabucco et al.) — the underlying SFT model and benchmark dataset.
- **Vertex AI Gemini 2.5 Flash** — used as the LLM judge for both feasibility classification and trajectory evaluation.
- The PEFT, transformers, bitsandbytes, and stable-baselines3 ecosystems.
