"""
Mock test script for RL training pipeline.

This script simulates the entire RL training flow without requiring:
- GPU (no actual LLM inference)
- Browser (no Playwright server)
- Judge LLM (mocked responses)

Run with: python test_rl_mock.py --debug
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Dict, Optional
import random

# Mock the BrowserJudgment
@dataclass
class MockBrowserJudgment:
    success: float = None
    efficiency: float = None
    self_correction: float = None
    response: str = None
    matched_response: str = None


class MockTokenizerOutput:
    """Mock tokenizer output that supports .to() method and ** unpacking."""
    
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self._data = {"input_ids": input_ids, "attention_mask": attention_mask}
    
    def to(self, device):
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        self._data = {"input_ids": self.input_ids, "attention_mask": self.attention_mask}
        return self
    
    def __getitem__(self, key):
        return self._data[key]
    
    def keys(self):
        return self._data.keys()
    
    def values(self):
        return self._data.values()
    
    def items(self):
        return self._data.items()


class MockTokenizer:
    """Mock tokenizer that simulates tokenization without actual model."""
    
    def __init__(self):
        self.eos_token_id = 0
        self.vocab_size = 32000
    
    def __call__(self, text, return_tensors="pt", truncation=True, max_length=8192):
        # Simulate tokenization - create fake token IDs
        num_tokens = min(len(text) // 4, max_length)  # Rough approximation
        num_tokens = max(num_tokens, 10)  # At least 10 tokens
        
        input_ids = torch.randint(1, self.vocab_size, (1, num_tokens))
        attention_mask = torch.ones_like(input_ids)
        
        return MockTokenizerOutput(input_ids, attention_mask)
    
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        # Just concatenate message contents
        return " ".join([m["content"] for m in messages])
    
    def decode(self, tokens, skip_special_tokens=True):
        return "This is a mock decoded response."
    
    def save_pretrained(self, path):
        print(f"[MOCK] Tokenizer saved to {path}")


class MockModel(nn.Module):
    """
    Mock language model with trainable parameters.
    
    This model has real parameters that can be trained via backprop,
    but doesn't do actual language modeling - just returns random logits.
    """
    
    def __init__(self, vocab_size=32000, hidden_size=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # Create some trainable parameters
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, vocab_size)
        
        self._device = torch.device("cpu")
    
    @property
    def device(self):
        return self._device
    
    def to(self, device):
        self._device = device
        return super().to(device)
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        # Simple forward pass
        x = self.embedding(input_ids)
        x = torch.relu(self.linear1(x))
        logits = self.linear2(x)
        
        # Return object with logits attribute
        class Output:
            pass
        output = Output()
        output.logits = logits
        return output
    
    def generate(self, input_ids, max_new_tokens=512, pad_token_id=0, **kwargs):
        # Return input + some random tokens
        batch_size = input_ids.shape[0]
        new_tokens = torch.randint(1, self.vocab_size, (batch_size, max_new_tokens))
        return torch.cat([input_ids, new_tokens], dim=1)
    
    def save_pretrained(self, path):
        print(f"[MOCK] Model saved to {path}")
    
    def parameters(self):
        return super().parameters()


class MockTrajectoryGenerator:
    """Generates mock trajectory data for testing."""
    
    MOCK_PROMPTS = [
        "You are a helpful browser agent. Navigate to the search bar and type 'weather'.",
        "You are a helpful browser agent. Click on the login button.",
        "You are a helpful browser agent. Scroll down to see more content.",
        "You are a helpful browser agent. Fill in the email field.",
        "You are a helpful browser agent. Submit the form.",
    ]
    
    MOCK_RESPONSES = [
        '{"action_key": "type", "target_element_id": "search-box", "action_kwargs": {"text": "weather"}}',
        '{"action_key": "click", "target_element_id": "login-btn", "action_kwargs": {}}',
        '{"action_key": "scroll", "target_element_id": null, "action_kwargs": {"direction": "down"}}',
        '{"action_key": "type", "target_element_id": "email-input", "action_kwargs": {"text": "test@example.com"}}',
        '{"action_key": "click", "target_element_id": "submit-btn", "action_kwargs": {}}',
    ]
    
    @classmethod
    def generate_trajectory(cls, num_steps: int = 3) -> tuple:
        """Generate mock trajectory prompts and responses."""
        num_steps = min(num_steps, len(cls.MOCK_PROMPTS))
        
        prompts = cls.MOCK_PROMPTS[:num_steps]
        responses = cls.MOCK_RESPONSES[:num_steps]
        
        return prompts, responses
    
    @classmethod
    def generate_judgment(cls, task_difficulty: str = "medium") -> MockBrowserJudgment:
        """Generate mock judgment with realistic scores."""
        
        if task_difficulty == "easy":
            success = random.uniform(0.7, 1.0)
            efficiency = random.uniform(0.6, 0.9)
            self_correction = random.uniform(0.5, 0.8)
        elif task_difficulty == "hard":
            success = random.uniform(0.2, 0.6)
            efficiency = random.uniform(0.3, 0.6)
            self_correction = random.uniform(0.4, 0.7)
        else:  # medium
            success = random.uniform(0.4, 0.8)
            efficiency = random.uniform(0.4, 0.7)
            self_correction = random.uniform(0.4, 0.7)
        
        return MockBrowserJudgment(
            success=success,
            efficiency=efficiency,
            self_correction=self_correction,
            response="Mock judgment response",
            matched_response='{"success": ' + str(success) + '}'
        )


def run_mock_rl_training(
    algorithm: str = "reinforce",
    num_trajectories: int = 5,
    steps_per_trajectory: int = 3,
    debug: bool = True,
    sb3_preset: str = "default"
):
    """
    Run mock RL training to demonstrate the training loop.
    
    Args:
        algorithm: "reinforce", "ppo", "grpo", or "sb3_ppo"
        num_trajectories: Number of trajectories to simulate
        steps_per_trajectory: Steps per trajectory
        debug: Enable debug logging
        sb3_preset: SB3 PPO configuration preset (only used with sb3_ppo)
    """
    
    # Import and configure RL trainer
    import rl_trainer
    rl_trainer.DEBUG_RL = debug
    
    from rl_trainer import OnPolicyTrainer, RLConfig
    
    print("=" * 60)
    print("MOCK RL TRAINING TEST")
    print("=" * 60)
    print(f"Algorithm: {algorithm.upper()}")
    print(f"Trajectories: {num_trajectories}")
    print(f"Steps per trajectory: {steps_per_trajectory}")
    print(f"Debug mode: {debug}")
    print("=" * 60)
    
    # Create mock model and tokenizer
    print("\n[SETUP] Creating mock model and tokenizer...")
    model = MockModel(vocab_size=32000, hidden_size=256)
    tokenizer = MockTokenizer()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[SETUP] Mock model created with {total_params:,} parameters ({trainable_params:,} trainable)")
    
    # Create RL config and trainer
    print(f"\n[SETUP] Initializing {algorithm.upper()} trainer...")
    
    if algorithm == "sb3_ppo":
        # Use SB3 PPO trainer
        try:
            from rl_sb3_ppo import SB3PPOTrainer
            from rl_sb3_config_examples import get_config as get_sb3_config
            
            config = get_sb3_config(sb3_preset)
            config.learning_rate = 1e-4  # Override for mock testing
            trainer = SB3PPOTrainer(model, tokenizer, config)
            print(f"[SETUP] SB3 PPO trainer initialized ({sb3_preset} preset)")
        except ImportError as e:
            print(f"[ERROR] Could not import SB3 PPO: {e}")
            print("[ERROR] Install with: pip install stable-baselines3 gymnasium")
            return None, [], []
    else:
        # Use custom RL algorithms
        config = RLConfig(
            algorithm=algorithm,
            learning_rate=1e-4,  # Higher LR for mock testing
            entropy_coef=0.01,
            gamma=0.99,
        )
        
        trainer = OnPolicyTrainer(model, tokenizer, config)
        print(f"[SETUP] Custom {algorithm.upper()} trainer initialized")
    
    # Track metrics
    all_rewards = []
    all_losses = []
    
    # Training loop
    print("\n" + "=" * 60)
    print("STARTING TRAINING LOOP")
    print("=" * 60)
    
    for i in range(num_trajectories):
        print(f"\n{'─' * 40}")
        print(f"TRAJECTORY {i + 1}/{num_trajectories}")
        print(f"{'─' * 40}")
        
        # Generate mock trajectory
        prompts, responses = MockTrajectoryGenerator.generate_trajectory(steps_per_trajectory)
        
        # Randomly vary difficulty for more interesting training
        difficulties = ["easy", "medium", "hard"]
        difficulty = random.choice(difficulties)
        judgment = MockTrajectoryGenerator.generate_judgment(difficulty)
        
        print(f"[TRAJECTORY] Generated {len(prompts)} steps (difficulty: {difficulty})")
        print(f"[TRAJECTORY] Judgment: success={judgment.success:.3f}, "
              f"efficiency={judgment.efficiency:.3f}, "
              f"self_correction={judgment.self_correction:.3f}")
        
        # Perform RL update
        print(f"\n[UPDATE] Performing {algorithm.upper()} update...")
        stats = trainer.update_policy(
            trajectory_prompts=prompts,
            trajectory_responses=responses,
            judgment=judgment
        )
        
        # Record metrics
        all_rewards.append(stats["trajectory_reward"])
        # SB3 uses "avg_loss", custom uses "total_loss"
        loss_value = stats.get("total_loss", stats.get("avg_loss", 0.0))
        all_losses.append(loss_value)
        
        # Print summary (handle both custom and SB3 stats)
        print(f"\n[RESULT] Trajectory {i + 1} complete:")
        print(f"  • Reward: {stats['trajectory_reward']:.4f}")
        
        # Custom algorithms have policy_loss and total_loss
        if "policy_loss" in stats:
            print(f"  • Policy Loss: {stats['policy_loss']:.4f}")
        if "total_loss" in stats:
            print(f"  • Total Loss: {stats['total_loss']:.4f}")
        
        # SB3 has avg_loss and avg_reward
        if "avg_loss" in stats:
            print(f"  • Avg Loss: {stats['avg_loss']:.4f}")
        if "avg_reward" in stats:
            print(f"  • Avg Reward: {stats['avg_reward']:.4f}")
        
        # Common optional stats
        if "baseline" in stats:
            print(f"  • Baseline: {stats['baseline']:.4f}")
        if "grad_norm" in stats:
            print(f"  • Gradient Norm: {stats['grad_norm']:.4f}")
        if "total_updates" in stats:
            print(f"  • Total Updates: {stats['total_updates']}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total trajectories: {num_trajectories}")
    print(f"Total updates: {trainer.training_stats['total_updates']}")
    print(f"Average reward: {sum(all_rewards) / len(all_rewards):.4f}")
    print(f"Average loss: {sum(all_losses) / len(all_losses):.4f}")
    print(f"Min reward: {min(all_rewards):.4f}")
    print(f"Max reward: {max(all_rewards):.4f}")
    
    # Show reward progression
    print("\nReward progression:")
    for i, reward in enumerate(all_rewards):
        bar = "█" * int(reward * 20)
        print(f"  Trajectory {i + 1}: {reward:.3f} |{bar}")
    
    print("\n[SUCCESS] Mock RL training completed successfully! ✓")
    
    return trainer, all_rewards, all_losses


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test RL training with mock data")
    parser.add_argument("--algorithm", type=str, default="reinforce",
                        choices=["reinforce", "ppo", "grpo", "sb3_ppo"],
                        help="RL algorithm to test")
    parser.add_argument("--num_trajectories", type=int, default=5,
                        help="Number of mock trajectories")
    parser.add_argument("--steps", type=int, default=3,
                        help="Steps per trajectory")
    parser.add_argument("--debug", action="store_true", default=True,
                        help="Enable debug logging")
    parser.add_argument("--no-debug", action="store_true",
                        help="Disable debug logging")
    parser.add_argument("--sb3_preset", type=str, default="default",
                        choices=["default", "low_memory", "aggressive", "conservative", "exploration"],
                        help="SB3 PPO preset (only used with sb3_ppo)")
    
    args = parser.parse_args()
    
    debug = args.debug and not args.no_debug
    
    run_mock_rl_training(
        algorithm=args.algorithm,
        num_trajectories=args.num_trajectories,
        steps_per_trajectory=args.steps,
        debug=debug,
        sb3_preset=args.sb3_preset
    )
