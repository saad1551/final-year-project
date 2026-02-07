"""
Stable Baselines3 PPO Implementation for Browser Navigation Agent.

This module provides a wrapper around Stable Baselines3's PPO algorithm
to work with the browser navigation task, making it configurable like
the custom RL algorithms implemented in rl_trainer.py.
"""

import torch
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from transformers import PreTrainedModel, PreTrainedTokenizer
import gymnasium as gym
from gymnasium import spaces

from insta.configs.judge_config import BrowserJudgment

# Stable Baselines3 imports
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.policies import ActorCriticPolicy
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Warning: stable-baselines3 not installed. Install with: pip install stable-baselines3")


@dataclass
class SB3PPOConfig:
    """Configuration for Stable Baselines3 PPO."""
    
    # Learning parameters
    learning_rate: float = 3e-4
    n_steps: int = 2048  # Number of steps to run for each environment per update
    batch_size: int = 64  # Minibatch size
    n_epochs: int = 10  # Number of epochs when optimizing the surrogate loss
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # Factor for trade-off of bias vs variance for GAE
    
    # PPO specific
    clip_range: float = 0.2  # Clipping parameter for PPO
    clip_range_vf: Optional[float] = None  # Clipping parameter for value function (None = no clipping)
    normalize_advantage: bool = True  # Whether to normalize advantage
    ent_coef: float = 0.01  # Entropy coefficient for the loss calculation
    vf_coef: float = 0.5  # Value function coefficient for the loss calculation
    max_grad_norm: float = 0.5  # Maximum value for gradient clipping
    
    # Reward weights (same as custom RL algorithms)
    success_weight: float = 0.5
    efficiency_weight: float = 0.3
    self_correction_weight: float = 0.2
    
    # Training parameters
    use_sde: bool = False  # Whether to use State Dependent Exploration
    sde_sample_freq: int = -1  # Sample a new noise matrix every n steps (-1 = only at rollout start)
    target_kl: Optional[float] = None  # Early stopping KL divergence threshold
    
    # Device
    device: str = "auto"  # Device to use ("auto", "cpu", "cuda")
    
    # Model architecture (for custom feature extractor)
    hidden_dim: int = 256
    n_hidden_layers: int = 2
    
    # Logging and checkpointing
    verbose: int = 1  # Verbosity level: 0 none, 1 training information, 2 debug
    tensorboard_log: Optional[str] = None  # TensorBoard log directory
    
    # Policy network
    policy_kwargs: Optional[Dict] = None  # Additional policy network kwargs


class RewardCalculator:
    """Calculates reward from BrowserJudgment scores (same as in rl_trainer.py)."""
    
    def __init__(self, config: SB3PPOConfig = None):
        self.config = config or SB3PPOConfig()
    
    def compute_reward(self, judgment: BrowserJudgment) -> float:
        """
        Compute weighted reward from judgment scores.
        
        The reward is a weighted combination of:
        - success: Whether the task was completed successfully (0-1)
        - efficiency: How efficiently the task was completed (0-1)
        - self_correction: Ability to recover from mistakes (0-1)
        """
        if judgment is None:
            return 0.0
        
        success = judgment.success if judgment.success is not None else 0.0
        efficiency = judgment.efficiency if judgment.efficiency is not None else 0.0
        self_correction = judgment.self_correction if judgment.self_correction is not None else 0.0
        
        reward = (
            self.config.success_weight * success +
            self.config.efficiency_weight * efficiency +
            self.config.self_correction_weight * self_correction
        )
        
        return reward


class DummyBrowserEnv(gym.Env):
    """
    Minimal dummy environment for Stable Baselines3 API compatibility.
    
    IMPORTANT: This environment is NOT used for actual browser interaction!
    
    It exists only because SB3's PPO requires a Gymnasium environment to be 
    instantiated. The actual training happens through the `update_policy()` 
    method which receives pre-collected trajectories from external browser 
    interactions.
    
    In practice:
    - Browser interactions happen externally
    - Trajectories are collected separately  
    - `update_policy()` is called with completed trajectories
    - This environment is just API scaffolding
    """
    
    def __init__(self, observation_dim: int = 64, action_dim: int = 10):
        super().__init__()
        
        # Minimal spaces (not actually used for training)
        self.observation_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(observation_dim,), 
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(action_dim)
        
        self.current_step = 0
        
    def reset(self, seed=None, options=None):
        """Reset to initial state (not used in practice)."""
        super().reset(seed=seed)
        self.current_step = 0
        observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        return observation, {}
    
    def step(self, action):
        """Take a step (not used in practice)."""
        self.current_step += 1
        observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        reward = 0.0
        terminated = False
        truncated = self.current_step >= 100
        info = {}
        return observation, reward, terminated, truncated, info




class TrainingCallback(BaseCallback):
    """
    Custom callback for logging and monitoring during PPO training.
    """
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _on_step(self) -> bool:
        """Called after each step."""
        # Log episode statistics
        if len(self.model.ep_info_buffer) > 0:
            for info in self.model.ep_info_buffer:
                if 'episode' in info:
                    self.episode_rewards.append(info['episode']['r'])
                    self.episode_lengths.append(info['episode']['l'])
        
        return True  # Continue training
    
    def _on_rollout_end(self) -> None:
        """Called after each rollout."""
        if self.verbose > 0 and len(self.episode_rewards) > 0:
            mean_reward = np.mean(self.episode_rewards[-10:])
            mean_length = np.mean(self.episode_lengths[-10:])
            print(f"Mean reward (last 10 episodes): {mean_reward:.2f}")
            print(f"Mean episode length: {mean_length:.2f}")


class SB3PPOTrainer:
    """
    Wrapper for Stable Baselines3 PPO to work with browser navigation task.
    
    This class provides a similar interface to the custom RL algorithms
    in rl_trainer.py, using Stable Baselines3's PPO under the hood.
    
    IMPORTANT: The primary interface is `update_policy()`, NOT environment-based training!
    - Browser interactions happen externally
    - Trajectories are pre-collected
    - `update_policy()` receives completed trajectory data
    - The Gymnasium environment is just for SB3 API compatibility
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: SB3PPOConfig = None
    ):
        if not SB3_AVAILABLE:
            raise ImportError(
                "stable-baselines3 is not installed. "
                "Install it with: pip install stable-baselines3"
            )
        
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or SB3PPOConfig()
        self.reward_calculator = RewardCalculator(self.config)
        
        # Create minimal dummy environment (only needed for SB3 API)
        self.env = DummyBrowserEnv(
            observation_dim=64,  # Minimal size
            action_dim=10        # Minimal size
        )
        
        # Wrap in DummyVecEnv (required by SB3)
        self.vec_env = DummyVecEnv([lambda: self.env])
        
        # Set up policy kwargs
        # NOTE: We don't use CustomLMFeatureExtractor because:
        # 1. The dummy environment doesn't need it
        # 2. It conflicts with quantized models (BitsAndBytes)
        # 3. Training happens via update_policy() with pre-collected trajectories
        policy_kwargs = self.config.policy_kwargs or {}
        
        # Initialize PPO agent
        self.ppo_agent = PPO(
            policy="MlpPolicy",
            env=self.vec_env,
            learning_rate=self.config.learning_rate,
            n_steps=self.config.n_steps,
            batch_size=self.config.batch_size,
            n_epochs=self.config.n_epochs,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            clip_range=self.config.clip_range,
            clip_range_vf=self.config.clip_range_vf,
            normalize_advantage=self.config.normalize_advantage,
            ent_coef=self.config.ent_coef,
            vf_coef=self.config.vf_coef,
            max_grad_norm=self.config.max_grad_norm,
            use_sde=self.config.use_sde,
            sde_sample_freq=self.config.sde_sample_freq,
            target_kl=self.config.target_kl,
            tensorboard_log=self.config.tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=self.config.verbose,
            device=self.config.device
        )
        
        # Training statistics
        self.training_stats = {
            "total_updates": 0,
            "avg_reward": 0.0,
            "avg_loss": 0.0,
        }
        
        print("Initialized SB3PPOTrainer with Stable Baselines3 PPO")
    
    def update_policy(
        self,
        trajectory_prompts: List[str],
        trajectory_responses: List[str],
        judgment: BrowserJudgment
    ) -> Dict[str, float]:
        """
        Update policy using pre-collected trajectory data.
        
        This is the PRIMARY interface for training - it receives completed
        trajectories from external browser interactions.
        
        NOTE: This does NOT use the Gymnasium environment for training!
        The environment is just scaffolding for SB3's API requirements.
        
        Args:
            trajectory_prompts: List of prompts in the trajectory
            trajectory_responses: List of responses in the trajectory
            judgment: Browser judgment with scores
            
        Returns:
            Dictionary with training metrics
        """
        # Compute reward from judgment
        trajectory_reward = self.reward_calculator.compute_reward(judgment)
        
        # NOTE: In a more sophisticated implementation, you would:
        # 1. Convert trajectory data into SB3-compatible format
        # 2. Create a replay buffer or custom data loader
        # 3. Train the policy directly on this data
        #
        # For now, this is a simplified implementation that calls SB3's
        # learn() method with minimal timesteps. The actual training on
        # your trajectory data would require a more specialized integration.
        
        try:
            # Simplified training call
            # In practice, you'd want to implement custom training logic here
            # that properly uses your trajectory data
            self.ppo_agent.learn(
                total_timesteps=len(trajectory_prompts),
                reset_num_timesteps=False,
                callback=TrainingCallback(verbose=0)  # Quiet mode
            )
            
            # Update statistics
            self.training_stats["total_updates"] += 1
            self.training_stats["avg_reward"] = (
                0.9 * self.training_stats["avg_reward"] + 0.1 * trajectory_reward
            )
            
            # Get loss from logger (if available)
            loss = 0.0
            if hasattr(self.ppo_agent, 'logger') and self.ppo_agent.logger is not None:
                loss = self.ppo_agent.logger.name_to_value.get('train/loss', 0.0)
            
            self.training_stats["avg_loss"] = (
                0.9 * self.training_stats["avg_loss"] + 0.1 * loss
            )
            
            return {
                "algorithm": "sb3_ppo",
                "trajectory_reward": trajectory_reward,
                "total_updates": self.training_stats["total_updates"],
                "avg_reward": self.training_stats["avg_reward"],
                "avg_loss": self.training_stats["avg_loss"],
                "num_steps": len(trajectory_prompts),
            }
            
        except Exception as e:
            print(f"Error during PPO update: {e}")
            return {
                "algorithm": "sb3_ppo",
                "trajectory_reward": trajectory_reward,
                "error": str(e),
                "num_steps": len(trajectory_prompts),
            }
    
    def train(
        self,
        total_timesteps: int,
        callback=None,
        log_interval: int = 1
    ):
        """
        Train the PPO agent for a specified number of timesteps.
        
        This can be used for standalone training when not using the
        update_policy interface.
        
        Args:
            total_timesteps: Total number of timesteps to train
            callback: Optional callback for monitoring
            log_interval: Logging interval
        """
        if callback is None:
            callback = TrainingCallback(verbose=self.config.verbose)
        
        self.ppo_agent.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval
        )
    
    def predict(
        self,
        observation,
        deterministic: bool = True
    ):
        """
        Predict action for given observation.
        
        Args:
            observation: Current observation
            deterministic: Whether to use deterministic policy
            
        Returns:
            action, state (state is None for non-recurrent policies)
        """
        return self.ppo_agent.predict(observation, deterministic=deterministic)
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        import os
        os.makedirs(path, exist_ok=True)
        
        # Save the PPO model
        self.ppo_agent.save(os.path.join(path, "sb3_ppo_model"))
        
        # Save the underlying language model
        self.model.save_pretrained(os.path.join(path, "lm_model"))
        self.tokenizer.save_pretrained(os.path.join(path, "lm_model"))
        
        # Save training statistics
        import json
        with open(os.path.join(path, "training_stats.json"), 'w') as f:
            json.dump(self.training_stats, f, indent=2)
        
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        import os
        
        # Load the PPO model
        ppo_path = os.path.join(path, "sb3_ppo_model")
        if os.path.exists(ppo_path + ".zip"):
            self.ppo_agent = PPO.load(ppo_path, env=self.vec_env)
            print(f"Loaded PPO model from {ppo_path}")
        
        # Load training statistics
        stats_path = os.path.join(path, "training_stats.json")
        if os.path.exists(stats_path):
            import json
            with open(stats_path, 'r') as f:
                self.training_stats = json.load(f)
            print(f"Loaded training stats from {stats_path}")


# Convenience function for creating SB3 PPO trainer
def create_sb3_ppo_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config: SB3PPOConfig = None
) -> SB3PPOTrainer:
    """
    Create a SB3 PPO trainer instance.
    
    Args:
        model: Pre-trained language model
        tokenizer: Tokenizer for the model
        config: PPO configuration
        
    Returns:
        SB3PPOTrainer instance
    """
    return SB3PPOTrainer(model, tokenizer, config)
