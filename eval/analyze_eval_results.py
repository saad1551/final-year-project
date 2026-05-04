"""
Analyze the JSON outputs of run_full_eval.sh.

Computes mean reward + success rate with 95% bootstrap CIs, plus paired
Wilcoxon (reward) and McNemar (binary success) tests for the comparison
RL-on-filtered vs Base SFT.

Inputs (per --run_dir):
  checkpoint_results_<ts>.json   <- RL-on-filtered (LoRA-adapted) trajectories
  base_results_<ts>.json         <- Base SFT trajectories

Outputs:
  - stdout: pretty-printed Markdown table
  - <run_dir>/summary.md:  same content as stdout
  - <run_dir>/summary.csv: per-cell rewards/success/CI for downstream plotting
  - <run_dir>/figure3.{png,pdf}: 2-bar comparison chart
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


SUCCESS_THRESHOLD = 0.5  # binary success: judge_success > 0.5
BOOTSTRAP_N = 10_000


def load_results(json_path: Path) -> list[dict]:
    """Load one evaluate_checkpoint.py JSON file. Returns the list of trajectory dicts."""
    with json_path.open() as f:
        rows = json.load(f)
    return [r for r in rows if r is not None]


def latest(glob_pattern: str, in_dir: Path) -> Path:
    matches = sorted(in_dir.glob(glob_pattern))
    if not matches:
        raise FileNotFoundError(f"no match for {glob_pattern} in {in_dir}")
    return matches[-1]


def per_task_metrics(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (rewards, successes_binary, num_steps).

    Reward is a weighted combination matching training (0.7s + 0.2e + 0.1sc).
    Tasks where the judge failed (None scores) are dropped.
    """
    rewards, successes, steps = [], [], []
    for r in rows:
        j = r.get("judgment")
        if not j:
            continue
        s = j.get("success")
        e = j.get("efficiency")
        sc = j.get("self_correction")
        if s is None and e is None and sc is None:
            continue
        s = s or 0.0
        e = e or 0.0
        sc = sc or 0.0
        rewards.append(0.7 * s + 0.2 * e + 0.1 * sc)
        successes.append(1.0 if s > SUCCESS_THRESHOLD else 0.0)
        steps.append(r.get("num_steps") or 0)
    return np.array(rewards), np.array(successes), np.array(steps)


def bootstrap_ci(values: np.ndarray, n: int = BOOTSTRAP_N, alpha: float = 0.05) -> tuple[float, float, float]:
    """Return (mean, lo, hi) where [lo, hi] is a 95% percentile bootstrap CI."""
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed=0)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    samples = values[idx].mean(axis=1)
    lo, hi = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(values.mean()), float(lo), float(hi)


def paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> float:
    """Return one-sided Wilcoxon signed-rank p-value for a > b."""
    try:
        from scipy.stats import wilcoxon
        diff = a - b
        nz = diff[diff != 0]
        if len(nz) < 5:
            return float("nan")
        _, p = wilcoxon(nz, alternative="greater")
        return float(p)
    except Exception:
        diff = a - b
        pos = int((diff > 0).sum())
        neg = int((diff < 0).sum())
        n = pos + neg
        if n == 0:
            return float("nan")
        from math import comb
        return float(sum(comb(n, k) for k in range(pos, n + 1)) / (2 ** n))


def mcnemar(a_succ: np.ndarray, b_succ: np.ndarray) -> float:
    """One-sided McNemar exact test: is a's success rate higher than b's?"""
    a01 = int(((a_succ == 1) & (b_succ == 0)).sum())  # a wins
    a10 = int(((a_succ == 0) & (b_succ == 1)).sum())  # b wins
    n = a01 + a10
    if n == 0:
        return float("nan")
    from math import comb
    return float(sum(comb(n, k) for k in range(a01, n + 1)) / (2 ** n))


def fmt_ci(stat: tuple[float, float, float]) -> str:
    m, lo, hi = stat
    if any(map(lambda v: v != v, [m, lo, hi])):
        return "—"
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def fmt_p(p: float) -> str:
    if p != p:
        return "—"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=Path, required=True,
                    help="Output directory used by run_full_eval.sh "
                         "(contains checkpoint_results_*.json and base_results_*.json)")
    args = ap.parse_args()

    rd = args.run_dir
    ckpt_rows = load_results(latest("checkpoint_results_*.json", rd))
    base_rows = load_results(latest("base_results_*.json", rd))

    # Per-task metrics. Same seed across both arms means rows are paired by index.
    ckpt_rew, ckpt_succ, ckpt_steps = per_task_metrics(ckpt_rows)
    base_rew, base_succ, base_steps = per_task_metrics(base_rows)

    # Truncate to common length for paired comparisons (defends against one arm
    # finishing more tasks than the other if eval was killed mid-run).
    n = min(len(ckpt_rew), len(base_rew))

    metrics = {
        "RL-on-filtered": {
            "n": len(ckpt_rew),
            "reward": bootstrap_ci(ckpt_rew),
            "success": bootstrap_ci(ckpt_succ),
            "mean_steps": float(np.mean(ckpt_steps)) if len(ckpt_steps) else float("nan"),
        },
        "Base SFT": {
            "n": len(base_rew),
            "reward": bootstrap_ci(base_rew),
            "success": bootstrap_ci(base_succ),
            "mean_steps": float(np.mean(base_steps)) if len(base_steps) else float("nan"),
        },
    }

    # Markdown table
    md = ["## Held-out evaluation (mean [95% bootstrap CI])\n",
          "| Checkpoint | n | Reward | Success rate | Mean steps |",
          "|---|---|---|---|---|"]
    for ckpt in ["Base SFT", "RL-on-filtered"]:
        m = metrics[ckpt]
        md.append(
            f"| **{ckpt}** | {m['n']} | {fmt_ci(m['reward'])} | "
            f"{fmt_ci(m['success'])} | {m['mean_steps']:.1f} |"
        )
    md.append("")

    # Pairwise test: RL-on-filtered vs Base SFT
    if n > 0:
        d_rew = ckpt_rew[:n] - base_rew[:n]
        m_d, lo_d, hi_d = bootstrap_ci(d_rew)
        d_succ_pp = (ckpt_succ[:n].mean() - base_succ[:n].mean()) * 100
        p_w = paired_wilcoxon(ckpt_rew[:n], base_rew[:n])
        p_m = mcnemar(ckpt_succ[:n], base_succ[:n])
        md.append("## Headline pair (RL-on-filtered vs Base SFT, paired)\n")
        md.append(f"- n paired           : {n}")
        md.append(f"- Δ reward (mean CI) : {m_d:+.3f} [{lo_d:+.3f}, {hi_d:+.3f}]")
        md.append(f"- Δ success rate     : {d_succ_pp:+.1f} percentage points")
        md.append(f"- Wilcoxon (reward)  : p = {fmt_p(p_w)}")
        md.append(f"- McNemar (success)  : p = {fmt_p(p_m)}")
        md.append("")

    text = "\n".join(md)
    print(text)
    (rd / "summary.md").write_text(text)

    # CSV summary for plotting
    with (rd / "summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["checkpoint", "n", "reward_mean", "reward_lo", "reward_hi",
                    "success_mean", "success_lo", "success_hi", "mean_steps"])
        for ckpt in ["Base SFT", "RL-on-filtered"]:
            m = metrics[ckpt]
            rm, rl, rh = m["reward"]
            sm, sl, sh = m["success"]
            w.writerow([ckpt, m["n"], rm, rl, rh, sm, sl, sh, m["mean_steps"]])

    # Figure 3: 2-bar comparison (success rate + reward)
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.2))
        labels = ["Base SFT", "RL-on-filtered"]
        x = np.arange(len(labels))
        for ax, metric_key, title in [(axes[0], "success", "Success rate (judge > 0.5)"),
                                       (axes[1], "reward", "Mean reward")]:
            means = [metrics[c][metric_key][0] for c in labels]
            los = [metrics[c][metric_key][1] for c in labels]
            his = [metrics[c][metric_key][2] for c in labels]
            err = [
                [m - lo for m, lo in zip(means, los)],
                [hi - m for m, hi in zip(means, his)],
            ]
            ax.bar(x, means, 0.55, yerr=err, capsize=4,
                   color=["#264653", "#2a9d8f"],
                   edgecolor="black", linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_title(title)
            ax.grid(alpha=0.3, axis="y")
        fig.suptitle("Held-out evaluation (filtered, paired)", fontsize=12, y=1.02)
        fig.tight_layout()
        fig.savefig(rd / "figure3.png", dpi=200, bbox_inches="tight")
        fig.savefig(rd / "figure3.pdf", bbox_inches="tight")
        print(f"\n[figure] wrote {rd / 'figure3.png'} and figure3.pdf")
    except Exception as e:
        print(f"[figure] skipped (matplotlib not available): {e}")


if __name__ == "__main__":
    main()
