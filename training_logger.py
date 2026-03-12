"""
Training Logger for RL Pipeline.

Logs all trajectory information to a CSV file for analysis and tracking.
Each row represents one trajectory with all relevant metrics.
"""

import os
import csv
from datetime import datetime
from typing import Optional, Dict, Any


class TrainingLogger:
    """
    Logs training trajectory information to a CSV file.

    The CSV is updated after every trajectory, making it easy to:
    - Track progress during training
    - Analyze which tasks were skipped vs used for optimization
    - Compare metrics across trajectories
    - Resume analysis from any point
    """

    CSV_COLUMNS = [
        # Identification
        "timestamp",
        "trajectory_id",
        "dataset_index",
        # Task info
        "website",
        "instruction",
        # Trajectory metrics
        "num_steps",
        "trajectory_completed",  # True if trajectory ran to completion (not failed)
        # Judge evaluation scores
        "judge_success",
        "judge_efficiency",
        "judge_self_correction",
        "reward",  # Computed weighted reward
        # Optimization info
        "used_for_optimization",  # Whether this trajectory updated the model
        "skip_reason",  # Why it was skipped (if applicable): "low_reward", "empty", "failed", etc.
        # RL metrics (only if used for optimization)
        "policy_loss",
        "entropy",
        "kl_divergence",
        "ref_kl_divergence",
        "total_loss",
        "baseline",
        "ppo_epochs_run",
        "grad_norm",
        # Cumulative stats
        "total_updates",
        "avg_reward_ema",  # Exponential moving average
        "avg_loss_ema",
        # Config
        "algorithm",
        "learning_rate",
        "min_reward_threshold",
        # Per-trajectory phase profiling (mean seconds per step; None when --profile not used)
        "prof_observation_mean",
        "prof_markdown_mean",
        "prof_inference_mean",
        "prof_action_exec_mean",
        "prof_step_total_mean",
    ]

    def __init__(self, log_dir: str = "training_logs", filename: str = None):
        """
        Initialize the training logger.

        Args:
            log_dir: Directory to store log files
            filename: Optional specific filename. If None, uses timestamp-based name.
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_log_{timestamp}.csv"

        self.log_path = os.path.join(log_dir, filename)
        self._initialize_csv()

        print(f"Training logger initialized: {self.log_path}")

    def _initialize_csv(self):
        """Create CSV file with headers if it doesn't exist."""
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.CSV_COLUMNS)
                writer.writeheader()

    def log_trajectory(
        self,
        trajectory_id: int,
        dataset_index: int,
        website: str,
        instruction: str,
        num_steps: int,
        trajectory_completed: bool,
        judgment: Optional[Any],  # BrowserJudgment or None
        rl_stats: Optional[Dict[str, Any]],
        trainer_stats: Optional[Dict[str, Any]],
        algorithm: str,
        learning_rate: float,
        min_reward_threshold: float = 0.1,
        prof_timings: Optional[Dict[str, float]] = None,
    ):
        """
        Log a single trajectory to the CSV file.

        Args:
            trajectory_id: Sequential trajectory number
            dataset_index: Index in the training dataset
            website: URL of the task
            instruction: Task instruction
            num_steps: Number of steps taken in trajectory
            trajectory_completed: Whether trajectory completed successfully
            judgment: BrowserJudgment object (or None if failed)
            rl_stats: Dictionary from trainer.update_policy() (or None)
            trainer_stats: Dictionary from trainer.training_stats
            algorithm: RL algorithm name
            learning_rate: Learning rate used
            min_reward_threshold: Minimum reward threshold for updates
        """
        # Extract judge scores
        judge_success = None
        judge_efficiency = None
        judge_self_correction = None
        reward = None

        if judgment is not None:
            judge_success = judgment.success
            judge_efficiency = judgment.efficiency
            judge_self_correction = judgment.self_correction

            # Compute reward (same formula as in rl_trainer)
            success = judge_success if judge_success is not None else 0.0
            efficiency = judge_efficiency if judge_efficiency is not None else 0.0
            self_correction = (
                judge_self_correction if judge_self_correction is not None else 0.0
            )
            reward = 0.7 * success + 0.2 * efficiency + 0.1 * self_correction

        # Determine if used for optimization
        used_for_optimization = False
        skip_reason = None

        if not trajectory_completed:
            skip_reason = "trajectory_failed"
        elif judgment is None:
            skip_reason = "no_judgment"
        elif rl_stats is None:
            skip_reason = "no_rl_update"
        elif rl_stats.get("skipped", False):
            skip_reason = rl_stats.get("skip_reason", "unknown")
        else:
            used_for_optimization = True

        # Extract RL metrics
        policy_loss = rl_stats.get("policy_loss") if rl_stats else None
        entropy = rl_stats.get("entropy") if rl_stats else None
        kl_divergence = rl_stats.get("kl_divergence") if rl_stats else None
        ref_kl_divergence = rl_stats.get("ref_kl_divergence") if rl_stats else None
        total_loss = rl_stats.get("total_loss") if rl_stats else None
        baseline = rl_stats.get("baseline") if rl_stats else None
        ppo_epochs_run = rl_stats.get("ppo_epochs_run") if rl_stats else None
        grad_norm = rl_stats.get("grad_norm") if rl_stats else None

        # Extract cumulative stats
        total_updates = trainer_stats.get("total_updates", 0) if trainer_stats else 0
        avg_reward_ema = trainer_stats.get("avg_reward", 0.0) if trainer_stats else 0.0
        avg_loss_ema = trainer_stats.get("avg_loss", 0.0) if trainer_stats else 0.0

        # Build row
        row = {
            "timestamp": datetime.now().isoformat(),
            "trajectory_id": trajectory_id,
            "dataset_index": dataset_index,
            "website": website,
            "instruction": instruction[:200] + "..."
            if len(instruction) > 200
            else instruction,
            "num_steps": num_steps,
            "trajectory_completed": trajectory_completed,
            "judge_success": judge_success,
            "judge_efficiency": judge_efficiency,
            "judge_self_correction": judge_self_correction,
            "reward": round(reward, 4) if reward is not None else None,
            "used_for_optimization": used_for_optimization,
            "skip_reason": skip_reason,
            "policy_loss": round(policy_loss, 6) if policy_loss is not None else None,
            "entropy": round(entropy, 6) if entropy is not None else None,
            "kl_divergence": round(kl_divergence, 6)
            if kl_divergence is not None
            else None,
            "ref_kl_divergence": round(ref_kl_divergence, 6)
            if ref_kl_divergence is not None
            else None,
            "total_loss": round(total_loss, 6) if total_loss is not None else None,
            "baseline": round(baseline, 6) if baseline is not None else None,
            "ppo_epochs_run": ppo_epochs_run,
            "grad_norm": round(grad_norm, 6) if grad_norm is not None else None,
            "total_updates": total_updates,
            "avg_reward_ema": round(avg_reward_ema, 4),
            "avg_loss_ema": round(avg_loss_ema, 6),
            "algorithm": algorithm,
            "learning_rate": learning_rate,
            "min_reward_threshold": min_reward_threshold,
            # Profiling (None when --profile not used)
            "prof_observation_mean": round(prof_timings["observation"], 4)
            if prof_timings and "observation" in prof_timings
            else None,
            "prof_markdown_mean": round(prof_timings["markdown"], 4)
            if prof_timings and "markdown" in prof_timings
            else None,
            "prof_inference_mean": round(prof_timings["inference"], 4)
            if prof_timings and "inference" in prof_timings
            else None,
            "prof_action_exec_mean": round(prof_timings["action_exec"], 4)
            if prof_timings and "action_exec" in prof_timings
            else None,
            "prof_step_total_mean": round(prof_timings["step_total"], 4)
            if prof_timings and "step_total" in prof_timings
            else None,
        }

        # Append to CSV
        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_COLUMNS)
            writer.writerow(row)

        # Print summary
        opt_status = "✓ USED" if used_for_optimization else f"✗ SKIPPED ({skip_reason})"
        reward_str = f"{reward:.4f}" if reward is not None else "N/A"
        print(
            f"[LOG] Trajectory {trajectory_id}: reward={reward_str}, "
            f"steps={num_steps}, {opt_status}"
        )

    def log_failed_trajectory(
        self,
        trajectory_id: int,
        dataset_index: int,
        website: str,
        instruction: str,
        algorithm: str,
        learning_rate: float,
        min_reward_threshold: float = 0.1,
        error_reason: str = "trajectory_failed",
    ):
        """
        Log a failed trajectory (one that couldn't complete).

        Args:
            trajectory_id: Sequential trajectory number
            dataset_index: Index in the training dataset
            website: URL of the task
            instruction: Task instruction
            algorithm: RL algorithm name
            learning_rate: Learning rate used
            min_reward_threshold: Minimum reward threshold
            error_reason: Why the trajectory failed
        """
        row = {
            "timestamp": datetime.now().isoformat(),
            "trajectory_id": trajectory_id,
            "dataset_index": dataset_index,
            "website": website,
            "instruction": instruction[:200] + "..."
            if len(instruction) > 200
            else instruction,
            "num_steps": 0,
            "trajectory_completed": False,
            "judge_success": None,
            "judge_efficiency": None,
            "judge_self_correction": None,
            "reward": None,
            "used_for_optimization": False,
            "skip_reason": error_reason,
            "policy_loss": None,
            "entropy": None,
            "kl_divergence": None,
            "ref_kl_divergence": None,
            "total_loss": None,
            "baseline": None,
            "ppo_epochs_run": None,
            "grad_norm": None,
            "total_updates": None,
            "avg_reward_ema": None,
            "avg_loss_ema": None,
            "algorithm": algorithm,
            "learning_rate": learning_rate,
            "min_reward_threshold": min_reward_threshold,
        }

        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_COLUMNS)
            writer.writerow(row)

        print(f"[LOG] Trajectory {trajectory_id}: FAILED ({error_reason})")

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of training progress from the log file.

        Returns:
            Dictionary with summary statistics.
        """
        import pandas as pd

        if not os.path.exists(self.log_path):
            return {"error": "Log file not found"}

        df = pd.read_csv(self.log_path)

        if len(df) == 0:
            return {"total_trajectories": 0}

        total = len(df)
        completed = df["trajectory_completed"].sum()
        used_for_opt = df["used_for_optimization"].sum()

        # Skip reasons breakdown
        skip_reasons = (
            df[df["used_for_optimization"] == False]["skip_reason"]
            .value_counts()
            .to_dict()
        )

        # Reward statistics (only for completed trajectories)
        completed_df = df[df["trajectory_completed"] == True]

        summary = {
            "total_trajectories": total,
            "completed_trajectories": int(completed),
            "failed_trajectories": total - int(completed),
            "used_for_optimization": int(used_for_opt),
            "skipped_trajectories": total - int(used_for_opt),
            "optimization_rate": round(used_for_opt / total * 100, 1)
            if total > 0
            else 0,
            "skip_reasons": skip_reasons,
        }

        if len(completed_df) > 0:
            summary["avg_reward"] = round(completed_df["reward"].mean(), 4)
            summary["avg_judge_success"] = round(
                completed_df["judge_success"].mean(), 4
            )
            summary["avg_steps"] = round(completed_df["num_steps"].mean(), 1)

        # RL metrics for optimized trajectories
        opt_df = df[df["used_for_optimization"] == True]
        if len(opt_df) > 0:
            summary["avg_policy_loss"] = round(opt_df["policy_loss"].mean(), 6)
            summary["avg_kl_divergence"] = round(opt_df["kl_divergence"].mean(), 6)
            if "ref_kl_divergence" in opt_df.columns:
                ref_kl = opt_df["ref_kl_divergence"].dropna()
                if len(ref_kl) > 0:
                    summary["avg_ref_kl_divergence"] = round(ref_kl.mean(), 6)

        # Profiling stats — only included when prof columns are present and non-empty
        prof_cols = {
            "observation": "prof_observation_mean",
            "markdown": "prof_markdown_mean",
            "inference": "prof_inference_mean",
            "action_exec": "prof_action_exec_mean",
            "step_total": "prof_step_total_mean",
        }
        prof_summary = {}
        for phase, col in prof_cols.items():
            if col in df.columns:
                series = df[col].dropna()
                if len(series) > 0:
                    prof_summary[phase] = {
                        "mean": round(float(series.mean()), 4),
                        "min": round(float(series.min()), 4),
                        "max": round(float(series.max()), 4),
                        "n": len(series),
                    }
        if prof_summary:
            summary["profiling"] = prof_summary

        return summary

    def print_summary(self):
        """Print a formatted summary of training progress."""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("TRAINING LOG SUMMARY")
        print("=" * 60)
        print(f"Log file: {self.log_path}")
        print(f"Total trajectories: {summary.get('total_trajectories', 0)}")
        print(f"  Completed: {summary.get('completed_trajectories', 0)}")
        print(f"  Failed: {summary.get('failed_trajectories', 0)}")
        print(
            f"Used for optimization: {summary.get('used_for_optimization', 0)} "
            f"({summary.get('optimization_rate', 0)}%)"
        )

        if "skip_reasons" in summary and summary["skip_reasons"]:
            print("\nSkip reasons:")
            for reason, count in summary["skip_reasons"].items():
                print(f"  {reason}: {count}")

        if "avg_reward" in summary:
            print(f"\nAvg reward (completed): {summary['avg_reward']}")
            print(f"Avg judge success: {summary.get('avg_judge_success', 'N/A')}")
            print(f"Avg steps: {summary.get('avg_steps', 'N/A')}")

        if "avg_policy_loss" in summary:
            print(f"\nRL Metrics (optimized trajectories):")
            print(f"  Avg policy loss: {summary['avg_policy_loss']}")
            print(f"  Avg KL divergence: {summary.get('avg_kl_divergence', 'N/A')}")
            if "avg_ref_kl_divergence" in summary:
                print(f"  Avg ref KL divergence: {summary['avg_ref_kl_divergence']}")

        if "profiling" in summary:
            prof = summary["profiling"]
            sep = "─" * 62
            n_trajs = prof.get("step_total", {}).get("n", "?")
            print(f"\n{sep}")
            print(f"  PHASE TIMING SUMMARY  ({n_trajs} trajectories)")
            print(sep)
            print(f"  {'Phase':<16}  {'mean/step':>9}  {'min':>9}  {'max':>9}")
            print(f"  {'-' * 14}  {'-' * 9}  {'-' * 9}  {'-' * 9}")
            for phase in (
                "observation",
                "markdown",
                "inference",
                "action_exec",
                "step_total",
            ):
                if phase not in prof:
                    continue
                p = prof[phase]
                label = phase if phase != "step_total" else "── total ──"
                print(
                    f"  {label:<16}  {p['mean']:>8.3f}s  {p['min']:>8.3f}s  {p['max']:>8.3f}s"
                )
            print(sep)

        print("=" * 60 + "\n")
