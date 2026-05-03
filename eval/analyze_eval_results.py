"""
Analyze the JSON outputs of run_full_eval.sh.

Builds the held-out evaluation summary table with mean, 95% bootstrap CIs,
paired Wilcoxon p-values for reward, and McNemar p-values for binary
success rate.

Inputs (per --run_dir):
  inv1_filtered_base_and_raw/
      checkpoint_results_<ts>.json   <- RL-on-raw on Filtered
      base_results_<ts>.json         <- Base SFT  on Filtered
  inv2_filtered_rl_filtered/
      checkpoint_results_<ts>.json   <- RL-on-filtered on Filtered
  inv3_unfiltered_base_and_raw/
      checkpoint_results_<ts>.json   <- RL-on-raw on Unfiltered
      base_results_<ts>.json         <- Base SFT  on Unfiltered
  inv4_unfiltered_rl_filtered/
      checkpoint_results_<ts>.json   <- RL-on-filtered on Unfiltered

Outputs:
  - stdout: pretty-printed Markdown table ready to paste into the report
  - <run_dir>/summary.csv: per-cell rewards/success/CI for downstream plotting
  - <run_dir>/summary.md:  same content as stdout
  - <run_dir>/figure3.png: grouped bar chart with error bars
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

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
    """Return (rewards, successes_binary, num_steps) aligned to the row order.

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
    """Return p-value of a one-sided Wilcoxon signed-rank test for a > b.

    Uses scipy if available, otherwise a manual paired sign-test fallback.
    """
    try:
        from scipy.stats import wilcoxon
        diff = a - b
        nz = diff[diff != 0]
        if len(nz) < 5:
            return float("nan")
        stat, p_two = wilcoxon(nz, alternative="greater")
        return float(p_two)
    except Exception:
        # Sign-test fallback (very conservative)
        diff = a - b
        pos = int((diff > 0).sum())
        neg = int((diff < 0).sum())
        n = pos + neg
        if n == 0:
            return float("nan")
        from math import comb
        p = sum(comb(n, k) for k in range(pos, n + 1)) / (2 ** n)
        return float(p)


def mcnemar(a_succ: np.ndarray, b_succ: np.ndarray) -> float:
    """One-sided McNemar exact test: is a's success rate higher than b's?"""
    a01 = int(((a_succ == 1) & (b_succ == 0)).sum())  # a wins
    a10 = int(((a_succ == 0) & (b_succ == 1)).sum())  # b wins
    n = a01 + a10
    if n == 0:
        return float("nan")
    # Exact binomial: P(X >= a01) under p=0.5
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
                    help="Output directory used by run_full_eval.sh (contains inv1-4 subdirs)")
    args = ap.parse_args()

    rd = args.run_dir
    inv1 = rd / "inv1_filtered_base_and_raw"
    inv2 = rd / "inv2_filtered_rl_filtered"
    inv3 = rd / "inv3_unfiltered_base_and_raw"
    inv4 = rd / "inv4_unfiltered_rl_filtered"

    # Filtered cells are required; unfiltered are optional (only present if
    # RUN_UNFILTERED=1 was set in run_full_eval.sh).
    cells: dict[tuple[str, str], list[dict]] = {
        ("Base SFT", "Filtered"):       load_results(latest("base_results_*.json", inv1)),
        ("RL-on-raw", "Filtered"):      load_results(latest("checkpoint_results_*.json", inv1)),
        ("RL-on-filtered", "Filtered"): load_results(latest("checkpoint_results_*.json", inv2)),
    }
    has_unfiltered = inv3.exists() and inv4.exists()
    if has_unfiltered:
        try:
            cells.update({
                ("Base SFT", "Unfiltered"):       load_results(latest("base_results_*.json", inv3)),
                ("RL-on-raw", "Unfiltered"):      load_results(latest("checkpoint_results_*.json", inv3)),
                ("RL-on-filtered", "Unfiltered"): load_results(latest("checkpoint_results_*.json", inv4)),
            })
        except FileNotFoundError:
            has_unfiltered = False

    # Compute per-cell metrics (paired across checkpoints by row order — eval script uses same seed)
    metrics: dict[tuple[str, str], dict] = {}
    for (ckpt, eval_set), rows in cells.items():
        rew, succ, steps = per_task_metrics(rows)
        metrics[(ckpt, eval_set)] = {
            "n": len(rew),
            "reward": bootstrap_ci(rew),
            "success_rate": bootstrap_ci(succ),
            "mean_steps": float(np.mean(steps)) if len(steps) else float("nan"),
            "_rew": rew,
            "_succ": succ,
        }

    # Render the main table
    rows = ["Base SFT", "RL-on-raw", "RL-on-filtered"]
    cols = ["Filtered", "Unfiltered"] if has_unfiltered else ["Filtered"]

    md = ["## Table 1 — held-out evaluation (mean [95% bootstrap CI])\n"]
    if has_unfiltered:
        md.append("| Checkpoint | Filtered Reward | Filtered Success | Unfiltered Reward | Unfiltered Success | Mean steps (Filt / Unfilt) |")
        md.append("|---|---|---|---|---|---|")
        for ckpt in rows:
            f = metrics[(ckpt, "Filtered")]
            u = metrics[(ckpt, "Unfiltered")]
            md.append(
                f"| **{ckpt}** | {fmt_ci(f['reward'])} | {fmt_ci(f['success_rate'])} | "
                f"{fmt_ci(u['reward'])} | {fmt_ci(u['success_rate'])} | "
                f"{f['mean_steps']:.1f} / {u['mean_steps']:.1f} |"
            )
    else:
        md.append("| Checkpoint | Filtered Reward | Filtered Success | Mean steps |")
        md.append("|---|---|---|---|")
        for ckpt in rows:
            f = metrics[(ckpt, "Filtered")]
            md.append(
                f"| **{ckpt}** | {fmt_ci(f['reward'])} | {fmt_ci(f['success_rate'])} | "
                f"{f['mean_steps']:.1f} |"
            )
    md.append("")

    # Pairwise tests (the comparisons that go in §4.5 prose)
    md.append("## Pairwise tests (one-sided: row > Base SFT)\n")
    md.append("| Pair | Test set | Δreward (mean [95% CI]) | Δsuccess (pp) | Wilcoxon p | McNemar p |")
    md.append("|---|---|---|---|---|---|")

    def pair_test(a_key, b_key) -> tuple[str, str, str, str]:
        a = metrics[a_key]
        b = metrics[b_key]
        rew_a, rew_b = a["_rew"], b["_rew"]
        succ_a, succ_b = a["_succ"], b["_succ"]
        # Truncate to common length (paired)
        n = min(len(rew_a), len(rew_b))
        if n == 0:
            return "—", "—", "—", "—"
        rew_a, rew_b = rew_a[:n], rew_b[:n]
        succ_a, succ_b = succ_a[:n], succ_b[:n]
        d_rew = rew_a - rew_b
        m, lo, hi = bootstrap_ci(d_rew)
        d_succ_pp = (succ_a.mean() - succ_b.mean()) * 100
        return (f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]",
                f"{d_succ_pp:+.1f}",
                fmt_p(paired_wilcoxon(rew_a, rew_b)),
                fmt_p(mcnemar(succ_a, succ_b)))

    for ckpt in ["RL-on-raw", "RL-on-filtered"]:
        for eval_set in cols:
            d_rew, d_succ, p_w, p_m = pair_test((ckpt, eval_set), ("Base SFT", eval_set))
            md.append(f"| {ckpt} vs Base SFT | {eval_set} | {d_rew} | {d_succ} | {p_w} | {p_m} |")
    # The main attribution claim
    for eval_set in cols:
        d_rew, d_succ, p_w, p_m = pair_test(("RL-on-filtered", eval_set), ("RL-on-raw", eval_set))
        md.append(f"| RL-on-filtered vs RL-on-raw | {eval_set} | {d_rew} | {d_succ} | {p_w} | {p_m} |")
    md.append("")

    text = "\n".join(md)
    print(text)
    (rd / "summary.md").write_text(text)

    # CSV summary for plotting
    with (rd / "summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["checkpoint", "eval_set", "n", "reward_mean", "reward_lo", "reward_hi",
                    "success_mean", "success_lo", "success_hi", "mean_steps"])
        for ckpt in rows:
            for eval_set in cols:
                m = metrics[(ckpt, eval_set)]
                rm, rl, rh = m["reward"]
                sm, sl, sh = m["success_rate"]
                w.writerow([ckpt, eval_set, m["n"], rm, rl, rh, sm, sl, sh, m["mean_steps"]])

    # Figure 3 (matplotlib, optional — skip if mpl missing)
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        x = np.arange(len(rows))
        width = 0.35
        for ax, metric_key, title in [(axes[0], "success_rate", "Success rate (judge > 0.5)"),
                                       (axes[1], "reward", "Mean reward")]:
            for i, eval_set in enumerate(cols):
                means = [metrics[(c, eval_set)][metric_key][0] for c in rows]
                los = [metrics[(c, eval_set)][metric_key][1] for c in rows]
                his = [metrics[(c, eval_set)][metric_key][2] for c in rows]
                err = [
                    [m - lo for m, lo in zip(means, los)],
                    [hi - m for m, hi in zip(means, his)],
                ]
                ax.bar(x + (i - 0.5) * width, means, width, yerr=err,
                       capsize=3, label=eval_set,
                       color=("#2a9d8f" if i == 0 else "#e76f51"),
                       edgecolor="black", linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(rows, rotation=15)
            ax.set_title(title)
            ax.grid(alpha=0.3, axis="y")
            ax.legend(loc="upper left")
        fig.suptitle("Held-out evaluation: 3 checkpoints × 2 test sets", fontsize=12, y=1.02)
        fig.tight_layout()
        fig.savefig(rd / "figure3.png", dpi=200, bbox_inches="tight")
        fig.savefig(rd / "figure3.pdf", bbox_inches="tight")
        print(f"\n[figure] wrote {rd / 'figure3.png'} and figure3.pdf")
    except Exception as e:
        print(f"[figure] skipped (matplotlib not available): {e}")


if __name__ == "__main__":
    main()
