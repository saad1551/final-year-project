"""
Analyze pass-1 best-checkpoint search output.

Given a directory containing ckpt_<N>/checkpoint_results_<ts>.json files,
compute paired reward + success-rate stats per checkpoint and a pairwise
comparison table. Same seed across checkpoints means the per-task results
are aligned by task index for paired tests.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import statistics as st
from pathlib import Path

import numpy as np


SUCCESS_THRESHOLD = 0.5
BOOTSTRAP_N = 10_000


def load_ckpt_results(ckpt_dir: Path) -> list[dict]:
    """Load the most-recent checkpoint_results_*.json under ckpt_dir."""
    matches = sorted(ckpt_dir.glob("checkpoint_results_*.json"))
    if not matches:
        return []
    with matches[-1].open() as f:
        rows = json.load(f)
    return [r for r in rows if r is not None]


def per_task(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rew, succ, steps = [], [], []
    for r in rows:
        j = r.get("judgment")
        if not j:
            continue
        s = j.get("success") or 0.0
        e = j.get("efficiency") or 0.0
        sc = j.get("self_correction") or 0.0
        rew.append(0.7 * s + 0.2 * e + 0.1 * sc)
        succ.append(1.0 if s > SUCCESS_THRESHOLD else 0.0)
        steps.append(r.get("num_steps") or 0)
    return np.array(rew), np.array(succ), np.array(steps)


def boot_ci(values: np.ndarray, n: int = BOOTSTRAP_N, alpha: float = 0.05):
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    samples = values[idx].mean(axis=1)
    lo, hi = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(values.mean()), float(lo), float(hi)


def fmt(stat):
    m, lo, hi = stat
    if any(x != x for x in (m, lo, hi)):
        return "—"
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=Path, required=True,
                    help="Output dir from best_checkpoint_search.sh")
    args = ap.parse_args()

    rd = args.run_dir
    ckpt_dirs = sorted(d for d in rd.iterdir() if d.is_dir() and d.name.startswith("ckpt_"))
    if not ckpt_dirs:
        print(f"no ckpt_* subdirs under {rd}")
        return

    # Build per-checkpoint metrics
    metrics = {}
    for d in ckpt_dirs:
        m = re.match(r"ckpt_(\d+)", d.name)
        if not m:
            continue
        cid = int(m.group(1))
        rows = load_ckpt_results(d)
        rew, succ, steps = per_task(rows)
        metrics[cid] = {
            "n": len(rew),
            "reward": boot_ci(rew),
            "success_rate": boot_ci(succ),
            "mean_steps": float(np.mean(steps)) if len(steps) else float("nan"),
            "_rew": rew,
            "_succ": succ,
        }

    cids = sorted(metrics.keys())
    print(f"\n## Pass-1 best-checkpoint search ({rd.name})\n")
    print("| Checkpoint | n | Reward (mean [95% CI]) | Success rate | Mean steps |")
    print("|---|---|---|---|---|")
    for c in cids:
        m = metrics[c]
        print(f"| `_{c}` | {m['n']} | {fmt(m['reward'])} | {fmt(m['success_rate'])} | {m['mean_steps']:.1f} |")

    # Pairwise paired comparisons (same seed = same tasks across checkpoints)
    if len(cids) > 1:
        print("\n## Pairwise paired comparisons (Δ vs earlier checkpoint)\n")
        print("| Pair | Δreward (mean [95% CI]) | Δsuccess (pp) | Verdict |")
        print("|---|---|---|---|")
        for i in range(1, len(cids)):
            a, b = cids[i], cids[i - 1]
            ra, sa = metrics[a]["_rew"], metrics[a]["_succ"]
            rb, sb = metrics[b]["_rew"], metrics[b]["_succ"]
            n = min(len(ra), len(rb))
            if n == 0:
                print(f"| `_{a}` vs `_{b}` | — | — | no data |")
                continue
            d_rew = ra[:n] - rb[:n]
            mr, lo, hi = boot_ci(d_rew)
            d_succ_pp = (sa[:n].mean() - sb[:n].mean()) * 100
            # Verdict: does the CI exclude 0?
            if lo > 0:
                v = f"`_{a}` better"
            elif hi < 0:
                v = f"`_{b}` better"
            else:
                v = "tied within CI"
            print(f"| `_{a}` vs `_{b}` | {mr:+.3f} [{lo:+.3f}, {hi:+.3f}] | {d_succ_pp:+.1f} | {v} |")

    # Pick the winner: highest mean reward, ties broken by success rate
    best = max(cids, key=lambda c: (metrics[c]["reward"][0], metrics[c]["success_rate"][0]))
    print(f"\n## Recommended checkpoint for full eval\n")
    print(f"**`checkpoint_trajectory_{best}`** — mean reward {metrics[best]['reward'][0]:.3f}, "
          f"success rate {metrics[best]['success_rate'][0]:.3f}.")
    if len(cids) > 1:
        latest = max(cids)
        if best == latest:
            print(f"\n*Latest checkpoint ({best}) is also best → no regression evidence; "
                  f"if time permits, restart training from `_{latest}` to push further.*")
        else:
            print(f"\n*Earlier checkpoint ({best}) beats the latest ({latest}) → regression evidence; "
                  f"do not restart, use `_{best}` for the headline.*")

    # Write a CSV summary too
    csv_path = rd / "summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["checkpoint", "n", "reward_mean", "reward_lo", "reward_hi",
                    "success_mean", "success_lo", "success_hi", "mean_steps"])
        for c in cids:
            m = metrics[c]
            rm, rl, rh = m["reward"]
            sm, sl, sh = m["success_rate"]
            w.writerow([f"_{c}", m["n"], rm, rl, rh, sm, sl, sh, m["mean_steps"]])
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
