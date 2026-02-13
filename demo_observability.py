"""
Example script demonstrating the observability system with mock data.

This shows how the observability logger works without running a full training session.
"""

from observability import ObservabilityLogger
from insta.configs.judge_config import BrowserJudgment


def create_mock_judgment(success, efficiency, self_correction):
    """Create a mock BrowserJudgment object."""
    return BrowserJudgment(
        success=success,
        efficiency=efficiency,
        self_correction=self_correction
    )


def main():
    print("=" * 80)
    print("OBSERVABILITY SYSTEM DEMO")
    print("=" * 80)
    print("\nThis demo shows how the observability system logs agent performance.\n")
    
    # Create logger
    logger = ObservabilityLogger(log_dir="demo_observability_logs")
    print(f"✓ Logger initialized: {logger.log_dir}\n")
    
    # Simulate 5 trajectories with improving performance
    trajectories = [
        {
            "task": "Navigate to google.com and search for 'AI'",
            "website": "https://google.com",
            "observations": [
                "Google homepage loaded. Search bar visible.",
                "Search bar focused, ready for input.",
                "Typed 'AI' in search box.",
                "Search results page displayed."
            ],
            "actions": [
                '{"action_key": "goto", "url": "https://google.com"}',
                '{"action_key": "click", "element_id": "search-box"}',
                '{"action_key": "type", "text": "AI"}',
                '{"action_key": "click", "element_id": "search-button"}',
                '{"action_key": "stop"}'
            ],
            "judgment": create_mock_judgment(0.7, 0.6, 0.5),
            "rl_stats": {
                "trajectory_reward": 0.6333,
                "policy_loss": 3.2,
                "entropy": 3.5,
                "total_loss": 3.4,
                "baseline": 0.6,
            }
        },
        {
            "task": "Go to amazon.com and search for 'laptop'",
            "website": "https://amazon.com",
            "observations": [
                "Amazon homepage loaded.",
                "Search field clicked.",
                "Typed 'laptop' in search.",
                "Search results displayed.",
                "Product listings visible."
            ],
            "actions": [
                '{"action_key": "goto", "url": "https://amazon.com"}',
                '{"action_key": "click", "element_id": "nav-search"}',
                '{"action_key": "type", "text": "laptop"}',
                '{"action_key": "click", "element_id": "nav-search-submit"}',
                '{"action_key": "stop"}'
            ],
            "judgment": create_mock_judgment(0.75, 0.65, 0.55),
            "rl_stats": {
                "trajectory_reward": 0.6500,
                "policy_loss": 3.0,
                "entropy": 3.4,
                "total_loss": 3.2,
                "baseline": 0.65,
            }
        },
        {
            "task": "Navigate to wikipedia.org and search for 'Python'",
            "website": "https://wikipedia.org",
            "observations": [
                "Wikipedia homepage loaded.",
                "Search box focused.",
                "Typed 'Python' in search.",
                "Article page loaded."
            ],
            "actions": [
                '{"action_key": "goto", "url": "https://wikipedia.org"}',
                '{"action_key": "click", "element_id": "searchInput"}',
                '{"action_key": "type", "text": "Python"}',
                '{"action_key": "click", "element_id": "searchButton"}',
                '{"action_key": "stop"}'
            ],
            "judgment": create_mock_judgment(0.80, 0.70, 0.60),
            "rl_stats": {
                "trajectory_reward": 0.7000,
                "policy_loss": 2.8,
                "entropy": 3.3,
                "total_loss": 3.0,
                "baseline": 0.68,
            }
        },
        {
            "task": "Go to github.com and search for 'pytorch'",
            "website": "https://github.com",
            "observations": [
                "GitHub homepage loaded.",
                "Search bar activated.",
                "Typed 'pytorch' in search.",
                "Repository results shown."
            ],
            "actions": [
                '{"action_key": "goto", "url": "https://github.com"}',
                '{"action_key": "click", "element_id": "query-builder-test"}',
                '{"action_key": "type", "text": "pytorch"}',
                '{"action_key": "click", "element_id": "search-button"}',
                '{"action_key": "stop"}'
            ],
            "judgment": create_mock_judgment(0.85, 0.75, 0.65),
            "rl_stats": {
                "trajectory_reward": 0.7500,
                "policy_loss": 2.5,
                "entropy": 3.2,
                "total_loss": 2.8,
                "baseline": 0.72,
            }
        },
        {
            "task": "Navigate to stackoverflow.com and search for 'javascript'",
            "website": "https://stackoverflow.com",
            "observations": [
                "Stack Overflow homepage loaded.",
                "Search box selected.",
                "Entered 'javascript' in search.",
                "Question results displayed.",
                "Top relevant questions visible."
            ],
            "actions": [
                '{"action_key": "goto", "url": "https://stackoverflow.com"}',
                '{"action_key": "click", "element_id": "search"}',
                '{"action_key": "type", "text": "javascript"}',
                '{"action_key": "click", "element_id": "search-submit"}',
                '{"action_key": "stop"}'
            ],
            "judgment": create_mock_judgment(0.90, 0.80, 0.70),
            "rl_stats": {
                "trajectory_reward": 0.8000,
                "policy_loss": 2.2,
                "entropy": 3.1,
                "total_loss": 2.6,
                "baseline": 0.78,
            }
        }
    ]
    
    # Log each trajectory
    print("Logging trajectories...\n")
    for i, traj in enumerate(trajectories):
        print(f"  [{i+1}/5] {traj['task'][:50]}...")
        
        logger.log_trajectory(
            trajectory_id=i + 1,
            dataset_index=i,
            task_instruction=traj['task'],
            website=traj['website'],
            observations=traj['observations'],
            action_jsons=traj['actions'],
            judgment=traj['judgment'],
            rl_stats=traj['rl_stats'],
            trainer_stats={'total_updates': i + 1},
            algorithm='ppo',
            learning_rate=1e-5
        )
    
    print("\n✓ All trajectories logged!\n")
    
    # Print summary
    logger.print_summary()
    
    # Show improvement metrics
    metrics = logger.get_improvement_metrics()
    print("\n" + "=" * 80)
    print("DETAILED IMPROVEMENT METRICS")
    print("=" * 80)
    print(f"\nFirst Half:")
    print(f"  Avg Success:         {metrics['first_half']['avg_success']:.4f}")
    print(f"  Avg Efficiency:      {metrics['first_half']['avg_efficiency']:.4f}")
    print(f"  Avg Self-correction: {metrics['first_half']['avg_self_correction']:.4f}")
    print(f"  Avg Reward:          {metrics['first_half']['avg_reward']:.4f}")
    
    print(f"\nSecond Half:")
    print(f"  Avg Success:         {metrics['second_half']['avg_success']:.4f}")
    print(f"  Avg Efficiency:      {metrics['second_half']['avg_efficiency']:.4f}")
    print(f"  Avg Self-correction: {metrics['second_half']['avg_self_correction']:.4f}")
    print(f"  Avg Reward:          {metrics['second_half']['avg_reward']:.4f}")
    
    print(f"\nImprovement:")
    print(f"  Success:         {metrics['improvement']['success']:+.4f}")
    print(f"  Efficiency:      {metrics['improvement']['efficiency']:+.4f}")
    print(f"  Self-correction: {metrics['improvement']['self_correction']:+.4f}")
    print(f"  Reward:          {metrics['improvement']['reward']:+.4f}")
    
    print("\n" + "=" * 80)
    print("\nCheck the following files for detailed logs:")
    print(f"  • Session summary: {logger.session_summary_file}")
    print(f"  • Session log:     {logger.session_file}")
    print(f"  • Trajectories:    {logger.trajectories_dir}/")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
