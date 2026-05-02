"""
Live training monitor.

Reads the trajectory-level CSV produced by training_logger.py and prints a
one-line status plus a small text dashboard of rolling metrics. Optionally
renders a matplotlib figure of reward / success / steps over trajectories.

Typical use during a GCP training run:

  # 1) Periodically rsync the CSV down from the VM (or watch a mounted path)
  rsync -avz fyp-train-t4:~/final-year-project/training_logs/  ./training_logs/

  # 2) Print a status line
  python scripts/monitor_training.py --csv training_logs/<latest>.csv

  # 3) Plot to PNG
  python scripts/monitor_training.py --csv training_logs/<latest>.csv --plot

The CSV columns expected (per training_logger.py):
  timestamp, trajectory_id, dataset_index, website, instruction,
  num_steps, trajectory_completed, judge_success, judge_efficiency,
  judge_self_correction, reward, used_for_optimization, skip_reason,
  policy_loss, entropy, total_loss
"""

import argparse
import csv
from pathlib import Path
from typing import List, Dict, Optional


def load_rows(csv_path: Path) -> List[Dict]:
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    # Coerce numerics
    for r in rows:
        for k in ("trajectory_id", "dataset_index", "num_steps"):
            r[k] = int(r[k]) if r.get(k) not in (None, "", "None") else None
        for k in ("judge_success", "judge_efficiency", "judge_self_correction",
                  "reward", "policy_loss", "entropy", "total_loss"):
            v = r.get(k)
            r[k] = float(v) if v not in (None, "", "None") else None
        r["trajectory_completed"] = (r.get("trajectory_completed", "").lower() == "true")
        r["used_for_optimization"] = (r.get("used_for_optimization", "").lower() == "true")
    return rows


def rolling(values: List[Optional[float]], window: int):
    out = []
    buf = []
    for v in values:
        if v is None:
            out.append(None)
            continue
        buf.append(v)
        if len(buf) > window:
            buf.pop(0)
        out.append(sum(buf) / len(buf))
    return out


def status_line(rows: List[Dict], window: int = 25) -> str:
    if not rows:
        return "no trajectories yet"
    last_n = rows[-window:]
    rewards = [r["reward"] for r in last_n if r["reward"] is not None]
    succ = [r["judge_success"] for r in last_n if r["judge_success"] is not None]
    steps = [r["num_steps"] for r in last_n if r["num_steps"] is not None]
    used = sum(1 for r in last_n if r["used_for_optimization"])

    def stats(vs):
        if not vs: return "—"
        m = sum(vs) / len(vs)
        if len(vs) > 1:
            var = sum((v - m) ** 2 for v in vs) / len(vs)
            sd = var ** 0.5
        else:
            sd = 0.0
        return f"{m:.3f}±{sd:.3f}"

    return (
        f"trajectories={len(rows)}  "
        f"reward(last{window})={stats(rewards)}  "
        f"success(last{window})={stats(succ)}  "
        f"steps(last{window})={stats(steps)}  "
        f"used_for_update={used}/{len(last_n)}"
    )


def text_table(rows: List[Dict], window: int = 25) -> str:
    """Print a small table of the last ~10 rolling-window points."""
    rewards = rolling([r["reward"] for r in rows], window)
    succ = rolling([r["judge_success"] for r in rows], window)
    steps = rolling([r["num_steps"] for r in rows], window)
    n = len(rows)
    if n == 0:
        return "(no data)"
    # Sample evenly: 10 points
    idxs = sorted({int(round(i * (n - 1) / 9)) for i in range(10)}) if n >= 10 else list(range(n))
    lines = [f"  {'idx':>5} {'rwd':>8} {'succ':>8} {'steps':>8}"]
    for i in idxs:
        rid = rows[i]["trajectory_id"]
        rw = f"{rewards[i]:.3f}" if rewards[i] is not None else "—"
        su = f"{succ[i]:.3f}" if succ[i] is not None else "—"
        st = f"{steps[i]:.1f}" if steps[i] is not None else "—"
        lines.append(f"  {rid if rid is not None else i:>5} {rw:>8} {su:>8} {st:>8}")
    return "\n".join(lines)


def make_plot(rows: List[Dict], window: int, out_path: Path):
    import matplotlib.pyplot as plt
    if not rows:
        print("no rows; skipping plot")
        return

    xs = [r["trajectory_id"] for r in rows]
    rew_raw = [r["reward"] for r in rows]
    succ_raw = [r["judge_success"] for r in rows]
    steps_raw = [r["num_steps"] for r in rows]
    rew_roll = rolling(rew_raw, window)
    succ_roll = rolling(succ_raw, window)
    steps_roll = rolling(steps_raw, window)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].scatter(xs, rew_raw, s=6, alpha=0.25, color="#1f77b4", label="raw")
    axes[0].plot(xs, rew_roll, color="#1f77b4", lw=2, label=f"rolling mean (w={window})")
    axes[0].set_ylabel("reward")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)

    axes[1].scatter(xs, succ_raw, s=6, alpha=0.25, color="#2a9d8f", label="raw")
    axes[1].plot(xs, succ_roll, color="#2a9d8f", lw=2, label=f"rolling mean (w={window})")
    axes[1].set_ylabel("judge_success")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(loc="lower right")
    axes[1].grid(alpha=0.3)

    axes[2].scatter(xs, steps_raw, s=6, alpha=0.25, color="#e76f51", label="raw")
    axes[2].plot(xs, steps_roll, color="#e76f51", lw=2, label=f"rolling mean (w={window})")
    axes[2].set_ylabel("num_steps")
    axes[2].set_xlabel("trajectory_id")
    axes[2].legend(loc="upper right")
    axes[2].grid(alpha=0.3)

    fig.suptitle("Training progress", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description="Live monitor for training_logger.py CSV output")
    p.add_argument("--csv", type=Path, required=True, help="Path to training-log CSV")
    p.add_argument("--window", type=int, default=25, help="Rolling-mean window (default: 25)")
    p.add_argument("--plot", action="store_true", help="Also render a PNG to <csv>.png")
    p.add_argument("--quiet", action="store_true", help="Only print the one-line status (good for watch)")
    args = p.parse_args()

    if not args.csv.exists():
        print(f"CSV not found: {args.csv}")
        return 1

    rows = load_rows(args.csv)
    print(status_line(rows, args.window))
    if not args.quiet:
        print()
        print(text_table(rows, args.window))

    if args.plot:
        out = args.csv.with_suffix(".png")
        make_plot(rows, args.window, out)


if __name__ == "__main__":
    main()
