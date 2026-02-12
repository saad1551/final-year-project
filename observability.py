"""
Observability module for tracking agent performance and improvement.

This module provides logging capabilities to track:
- Textual summaries of agent steps in each trajectory
- Judge scores (success, efficiency, self-correction)
- Training metrics and RL statistics
- Overall agent improvement over time
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict


@dataclass
class StepSummary:
    """Summary of a single agent step."""
    step_number: int
    observation_preview: str  # First 200 chars of markdown observation
    action_type: str  # e.g., "click", "type", "goto", "stop"
    action_details: str  # JSON string of the action
    timestamp: str


@dataclass
class TrajectoryLog:
    """Complete log of a trajectory execution."""
    trajectory_id: int
    dataset_index: int
    task_instruction: str
    website: str
    timestamp: str
    
    # Execution details
    total_steps: int
    step_summaries: List[StepSummary]
    
    # Judge scores
    judge_success: float
    judge_efficiency: float
    judge_self_correction: float
    
    # RL metrics (if available)
    trajectory_reward: Optional[float] = None
    policy_loss: Optional[float] = None
    entropy: Optional[float] = None
    total_loss: Optional[float] = None
    baseline: Optional[float] = None
    grad_norm: Optional[float] = None
    
    # Training context
    algorithm: Optional[str] = None
    learning_rate: Optional[float] = None
    total_updates: Optional[int] = None


class ObservabilityLogger:
    """Logger for tracking agent performance and improvement."""
    
    def __init__(self, log_dir: str = "observability_logs"):
        """
        Initialize the observability logger.
        
        Args:
            log_dir: Directory to store log files
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Create subdirectories
        self.trajectories_dir = os.path.join(log_dir, "trajectories")
        self.summaries_dir = os.path.join(log_dir, "summaries")
        os.makedirs(self.trajectories_dir, exist_ok=True)
        os.makedirs(self.summaries_dir, exist_ok=True)
        
        # Session log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_file = os.path.join(log_dir, f"session_{timestamp}.jsonl")
        self.session_summary_file = os.path.join(log_dir, f"session_{timestamp}_summary.txt")
        
        # In-memory tracking
        self.trajectory_logs: List[TrajectoryLog] = []
        
    def log_trajectory(
        self,
        trajectory_id: int,
        dataset_index: int,
        task_instruction: str,
        website: str,
        observations: List[str],
        action_jsons: List[str],
        judgment: Any,  # BrowserJudgment object
        rl_stats: Optional[Dict] = None,
        trainer_stats: Optional[Dict] = None,
        algorithm: Optional[str] = None,
        learning_rate: Optional[float] = None
    ) -> TrajectoryLog:
        """
        Log a complete trajectory with all relevant information.
        
        Args:
            trajectory_id: Sequential ID for this trajectory
            dataset_index: Index in the dataset
            task_instruction: The task the agent was trying to complete
            website: Starting website URL
            observations: List of markdown observations
            action_jsons: List of action JSON strings
            judgment: BrowserJudgment object with scores
            rl_stats: Optional RL update statistics
            trainer_stats: Optional trainer statistics
            algorithm: RL algorithm used
            learning_rate: Learning rate used
            
        Returns:
            TrajectoryLog object
        """
        timestamp = datetime.now().isoformat()
        
        # Create step summaries
        step_summaries = []
        for i, (obs, action_json) in enumerate(zip(observations, action_jsons)):
            # Extract action type from JSON
            try:
                action_dict = json.loads(action_json)
                action_type = action_dict.get("action_key", "unknown")
            except json.JSONDecodeError:
                action_type = "parse_error"
            
            step_summary = StepSummary(
                step_number=i + 1,
                observation_preview=obs[:200] if obs else "(empty)",
                action_type=action_type,
                action_details=action_json,
                timestamp=datetime.now().isoformat()
            )
            step_summaries.append(step_summary)
        
        # Extract RL metrics
        trajectory_reward = None
        policy_loss = None
        entropy = None
        total_loss = None
        baseline = None
        grad_norm = None
        total_updates = None
        
        if rl_stats:
            trajectory_reward = rl_stats.get('trajectory_reward')
            policy_loss = rl_stats.get('policy_loss')
            entropy = rl_stats.get('entropy')
            total_loss = rl_stats.get('total_loss')
            baseline = rl_stats.get('baseline')
            grad_norm = rl_stats.get('grad_norm')
        
        if trainer_stats:
            total_updates = trainer_stats.get('total_updates')
        
        # Create trajectory log
        traj_log = TrajectoryLog(
            trajectory_id=trajectory_id,
            dataset_index=dataset_index,
            task_instruction=task_instruction,
            website=website,
            timestamp=timestamp,
            total_steps=len(action_jsons),
            step_summaries=step_summaries,
            judge_success=judgment.success,
            judge_efficiency=judgment.efficiency,
            judge_self_correction=judgment.self_correction,
            trajectory_reward=trajectory_reward,
            policy_loss=policy_loss,
            entropy=entropy,
            total_loss=total_loss,
            baseline=baseline,
            grad_norm=grad_norm,
            algorithm=algorithm,
            learning_rate=learning_rate,
            total_updates=total_updates
        )
        
        # Store in memory
        self.trajectory_logs.append(traj_log)
        
        # Write to individual trajectory file
        self._write_trajectory_file(traj_log)
        
        # Append to session JSONL
        self._append_to_session(traj_log)
        
        # Update session summary
        self._update_session_summary()
        
        return traj_log
    
    def _write_trajectory_file(self, traj_log: TrajectoryLog):
        """Write a detailed trajectory log to a text file."""
        filename = f"trajectory_{traj_log.trajectory_id:04d}_idx{traj_log.dataset_index}.txt"
        filepath = os.path.join(self.trajectories_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"TRAJECTORY #{traj_log.trajectory_id} (Dataset Index: {traj_log.dataset_index})\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Timestamp: {traj_log.timestamp}\n")
            f.write(f"Task: {traj_log.task_instruction}\n")
            f.write(f"Website: {traj_log.website}\n")
            f.write(f"Total Steps: {traj_log.total_steps}\n\n")
            
            if traj_log.algorithm:
                f.write(f"Algorithm: {traj_log.algorithm.upper()}\n")
                f.write(f"Learning Rate: {traj_log.learning_rate}\n")
                f.write(f"Total Updates: {traj_log.total_updates}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("JUDGE SCORES\n")
            f.write("-" * 80 + "\n")
            f.write(f"Success:         {traj_log.judge_success:.4f}\n")
            f.write(f"Efficiency:      {traj_log.judge_efficiency:.4f}\n")
            f.write(f"Self-correction: {traj_log.judge_self_correction:.4f}\n\n")
            
            if traj_log.trajectory_reward is not None:
                f.write("-" * 80 + "\n")
                f.write("RL METRICS\n")
                f.write("-" * 80 + "\n")
                f.write(f"Trajectory Reward: {traj_log.trajectory_reward:.4f}\n")
                if traj_log.policy_loss is not None:
                    f.write(f"Policy Loss:       {traj_log.policy_loss:.4f}\n")
                if traj_log.entropy is not None:
                    f.write(f"Entropy:           {traj_log.entropy:.4f}\n")
                if traj_log.total_loss is not None:
                    f.write(f"Total Loss:        {traj_log.total_loss:.4f}\n")
                if traj_log.baseline is not None:
                    f.write(f"Baseline:          {traj_log.baseline:.4f}\n")
                if traj_log.grad_norm is not None:
                    f.write(f"Gradient Norm:     {traj_log.grad_norm:.4f}\n")
                f.write("\n")
            
            f.write("-" * 80 + "\n")
            f.write("STEP-BY-STEP SUMMARY\n")
            f.write("-" * 80 + "\n\n")
            
            for step in traj_log.step_summaries:
                f.write(f"Step {step.step_number}: {step.action_type.upper()}\n")
                f.write(f"  Action: {step.action_details}\n")
                f.write(f"  Observation Preview: {step.observation_preview}...\n\n")
            
            f.write("=" * 80 + "\n")
    
    def _append_to_session(self, traj_log: TrajectoryLog):
        """Append trajectory log to session JSONL file."""
        with open(self.session_file, 'a') as f:
            json.dump(asdict(traj_log), f)
            f.write('\n')
    
    def _update_session_summary(self):
        """Update the session summary with aggregate statistics."""
        if not self.trajectory_logs:
            return
        
        num_trajectories = len(self.trajectory_logs)
        
        # Calculate aggregate metrics
        avg_success = sum(t.judge_success for t in self.trajectory_logs) / num_trajectories
        avg_efficiency = sum(t.judge_efficiency for t in self.trajectory_logs) / num_trajectories
        avg_self_correction = sum(t.judge_self_correction for t in self.trajectory_logs) / num_trajectories
        avg_steps = sum(t.total_steps for t in self.trajectory_logs) / num_trajectories
        
        rewards = [t.trajectory_reward for t in self.trajectory_logs if t.trajectory_reward is not None]
        
        with open(self.session_summary_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("TRAINING SESSION SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Trajectories: {num_trajectories}\n\n")
            
            if self.trajectory_logs[0].algorithm:
                f.write(f"Algorithm: {self.trajectory_logs[0].algorithm.upper()}\n")
                f.write(f"Learning Rate: {self.trajectory_logs[0].learning_rate}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("AVERAGE JUDGE SCORES\n")
            f.write("-" * 80 + "\n")
            f.write(f"Success:         {avg_success:.4f}\n")
            f.write(f"Efficiency:      {avg_efficiency:.4f}\n")
            f.write(f"Self-correction: {avg_self_correction:.4f}\n")
            f.write(f"Steps per Trajectory: {avg_steps:.2f}\n\n")
            
            if rewards:
                f.write("-" * 80 + "\n")
                f.write("REWARD STATISTICS\n")
                f.write("-" * 80 + "\n")
                f.write(f"Average Reward: {sum(rewards) / len(rewards):.4f}\n")
                f.write(f"Max Reward:     {max(rewards):.4f}\n")
                f.write(f"Min Reward:     {min(rewards):.4f}\n\n")
                
                # Show improvement trend (last 10 vs first 10)
                if len(rewards) >= 20:
                    first_10_avg = sum(rewards[:10]) / 10
                    last_10_avg = sum(rewards[-10:]) / 10
                    improvement = ((last_10_avg - first_10_avg) / abs(first_10_avg)) * 100
                    f.write(f"First 10 avg:   {first_10_avg:.4f}\n")
                    f.write(f"Last 10 avg:    {last_10_avg:.4f}\n")
                    f.write(f"Improvement:    {improvement:+.2f}%\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("TRAJECTORY BREAKDOWN\n")
            f.write("-" * 80 + "\n\n")
            
            for traj in self.trajectory_logs:
                f.write(f"Trajectory #{traj.trajectory_id} (idx {traj.dataset_index})\n")
                f.write(f"  Task: {traj.task_instruction[:60]}...\n")
                f.write(f"  Steps: {traj.total_steps}\n")
                f.write(f"  Judge: S={traj.judge_success:.2f}, E={traj.judge_efficiency:.2f}, SC={traj.judge_self_correction:.2f}\n")
                if traj.trajectory_reward is not None:
                    f.write(f"  Reward: {traj.trajectory_reward:.4f}\n")
                f.write("\n")
            
            f.write("=" * 80 + "\n")
    
    def get_improvement_metrics(self) -> Dict[str, Any]:
        """
        Calculate improvement metrics across trajectories.
        
        Returns:
            Dictionary with improvement statistics
        """
        if len(self.trajectory_logs) < 2:
            return {"error": "Not enough trajectories to calculate improvement"}
        
        # Split into first half and second half
        mid = len(self.trajectory_logs) // 2
        first_half = self.trajectory_logs[:mid]
        second_half = self.trajectory_logs[mid:]
        
        def avg_metric(logs, attr):
            values = [getattr(log, attr) for log in logs if getattr(log, attr) is not None]
            return sum(values) / len(values) if values else 0
        
        return {
            "total_trajectories": len(self.trajectory_logs),
            "first_half": {
                "avg_success": avg_metric(first_half, 'judge_success'),
                "avg_efficiency": avg_metric(first_half, 'judge_efficiency'),
                "avg_self_correction": avg_metric(first_half, 'judge_self_correction'),
                "avg_reward": avg_metric(first_half, 'trajectory_reward'),
            },
            "second_half": {
                "avg_success": avg_metric(second_half, 'judge_success'),
                "avg_efficiency": avg_metric(second_half, 'judge_efficiency'),
                "avg_self_correction": avg_metric(second_half, 'judge_self_correction'),
                "avg_reward": avg_metric(second_half, 'trajectory_reward'),
            },
            "improvement": {
                "success": avg_metric(second_half, 'judge_success') - avg_metric(first_half, 'judge_success'),
                "efficiency": avg_metric(second_half, 'judge_efficiency') - avg_metric(first_half, 'judge_efficiency'),
                "self_correction": avg_metric(second_half, 'judge_self_correction') - avg_metric(first_half, 'judge_self_correction'),
                "reward": avg_metric(second_half, 'trajectory_reward') - avg_metric(first_half, 'trajectory_reward'),
            }
        }
    
    def print_summary(self):
        """Print a summary of the current session to console."""
        if not self.trajectory_logs:
            print("No trajectories logged yet.")
            return
        
        print("\n" + "=" * 80)
        print("OBSERVABILITY SUMMARY")
        print("=" * 80)
        print(f"Total Trajectories: {len(self.trajectory_logs)}")
        print(f"Session Log: {self.session_file}")
        print(f"Summary File: {self.session_summary_file}")
        print(f"Trajectory Details: {self.trajectories_dir}/")
        
        if len(self.trajectory_logs) >= 2:
            metrics = self.get_improvement_metrics()
            print("\nImprovement (Second Half vs First Half):")
            print(f"  Success:         {metrics['improvement']['success']:+.4f}")
            print(f"  Efficiency:      {metrics['improvement']['efficiency']:+.4f}")
            print(f"  Self-correction: {metrics['improvement']['self_correction']:+.4f}")
            if metrics['improvement']['reward']:
                print(f"  Reward:          {metrics['improvement']['reward']:+.4f}")
        
        print("=" * 80 + "\n")
