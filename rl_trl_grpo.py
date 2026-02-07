"""
TRL GRPO Implementation for Browser Navigation Agent.

This module provides a wrapper around TRL's GRPOTrainer algorithm
to work with the browser navigation task, making it configurable like
the custom RL algorithms implemented in rl_trainer.py.

Key Features:
- Group Relative Policy Optimization from Hugging Face TRL
- Same interface as other RL implementations (REINFORCE, PPO, custom GRPO)
- Custom reward function integration
- Memory efficient compared to PPO (no separate value network)
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field
from transformers import PreTrainedModel, PreTrainedTokenizer
import os
import json

from insta.configs.judge_config import BrowserJudgment

# TRL imports
try:
    from trl import GRPOTrainer, GRPOConfig
    from datasets import Dataset
    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False
    print("Warning: TRL not installed. Install with: pip install trl")


@dataclass
class TRLGRPOConfig:
    """Configuration for TRL GRPO Trainer."""
    
    # Learning parameters
    learning_rate: float = 1e-5
    num_train_epochs: int = 1  # Number of epochs per update_policy call
    
    # GRPO specific parameters
    num_generations: int = 4  # Number of completions to generate per prompt (group size)
    temperature: float = 1.0  # Sampling temperature for generation
    max_new_tokens: int = 512  # Maximum tokens to generate
    
    # Training parameters
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = False
    
    # Optimization
    max_grad_norm: float = 1.0
    optim: str = "adamw_torch"
    warmup_ratio: float = 0.1
    
    # GRPO algorithm parameters
    beta: float = 0.1  # KL penalty coefficient
    gamma: float = 0.99  # Discount factor
    
    # Reward weights (same as custom RL algorithms)
    success_weight: float = 0.5
    efficiency_weight: float = 0.3
    self_correction_weight: float = 0.2
    
    # Logging
    logging_steps: int = 1
    verbose: int = 1
    
    # Output directory
    output_dir: str = "./grpo_output"
    
    # Device
    device: str = "auto"


class RewardCalculator:
    """Calculates reward from BrowserJudgment scores (same as in rl_trainer.py)."""
    
    def __init__(self, config: TRLGRPOConfig = None):
        self.config = config or TRLGRPOConfig()
    
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


class TRLGRPOTrainer:
    """
    Wrapper for TRL's GRPOTrainer to work with browser navigation task.
    
    This class provides a similar interface to the custom RL algorithms
    in rl_trainer.py, using TRL's GRPOTrainer under the hood.
    
    Key differences from custom GRPO:
    - Uses TRL's battle-tested implementation
    - Generates multiple completions per prompt (true GRPO)
    - More memory efficient
    - Better suited for LLM fine-tuning
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: TRLGRPOConfig = None
    ):
        if not TRL_AVAILABLE:
            raise ImportError(
                "TRL is not installed. "
                "Install it with: pip install trl"
            )
        
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or TRLGRPOConfig()
        self.reward_calculator = RewardCalculator(self.config)
        
        # Store judgment(s) for reward function
        self.current_judgments: Optional[List[BrowserJudgment]] = None
        
        # Training statistics
        self.training_stats = {
            "total_updates": 0,
            "avg_reward": 0.0,
            "avg_loss": 0.0,
            "total_episodes": 0,
        }
        
        # We'll initialize the TRL trainer lazily when we have data
        self.grpo_trainer: Optional[GRPOTrainer] = None
        
        print("Initialized TRLGRPOTrainer with Hugging Face TRL")
    
    def _create_reward_function(self) -> Callable:
        """
        Create a reward function compatible with TRL's GRPOTrainer.
        
        The reward function must accept:
        - prompts: List of prompts
        - completions: List of generated completions
        - **kwargs: Additional dataset columns
        
        And return a list of rewards (one per completion).
        """
        def reward_function(
            prompts: List[str],
            completions: List[str],
            **kwargs
        ) -> List[float]:
            """
            Custom reward function for browser navigation task.
            
            Maps each completion to its corresponding judgment from the
            current_judgments list (populated during update_policy).
            """
            rewards = []
            
            # For each completion, find its reward
            for i, completion in enumerate(completions):
                if self.current_judgments is not None and i < len(self.current_judgments):
                    # Use the actual judgment reward for this specific completion
                    reward = self.reward_calculator.compute_reward(self.current_judgments[i])
                else:
                    # Fallback: use first judgment or heuristic
                    if self.current_judgments and len(self.current_judgments) > 0:
                        reward = self.reward_calculator.compute_reward(self.current_judgments[0])
                    else:
                        # Last resort heuristic (shouldn't happen in normal flow)
                        reward = max(0.0, 1.0 - len(completion.split()) / 100.0)
                
                rewards.append(reward)
            
            return rewards
        
        return reward_function
    
    def _create_dataset_from_trajectory(
        self,
        trajectory_prompts: List[str],
        trajectory_responses: List[str]
    ) -> Dataset:
        """
        Create a HuggingFace Dataset from trajectory data.
        
        TRL's GRPOTrainer expects a dataset with a "prompt" column.
        We'll use the trajectory prompts as training data.
        """
        # For GRPO, we only need prompts - it will generate completions
        dataset_dict = {
            "prompt": trajectory_prompts,
            # Optionally include reference responses for context
            "reference_response": trajectory_responses,
        }
        
        return Dataset.from_dict(dataset_dict)
    
    def _initialize_trainer(self, train_dataset: Dataset):
        """Initialize the TRL GRPOTrainer with the dataset."""
        
        # Create generation_kwargs for text generation parameters
        generation_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
            "do_sample": True,  # Required for temperature sampling
        }
        
        # Create GRPOConfig
        grpo_config = GRPOConfig(
            output_dir=self.config.output_dir,
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            gradient_checkpointing=self.config.gradient_checkpointing,
            logging_steps=self.config.logging_steps,
            max_grad_norm=self.config.max_grad_norm,
            optim=self.config.optim,
            warmup_ratio=self.config.warmup_ratio,
            # GRPO specific
            num_generations=self.config.num_generations,
            generation_kwargs=generation_kwargs,  # Pass generation params here
            beta=self.config.beta,
            # Training steps - needed for small datasets
            max_steps=100,  # Set a reasonable number of steps for small datasets
            # Disable some features we don't need
            save_strategy="no",
            report_to="none",
        )
        
        # Create reward function
        reward_func = self._create_reward_function()
        
        # Initialize GRPOTrainer
        # Note: TRL's GRPOTrainer infers tokenizer from the model automatically
        # and uses 'args' instead of 'config' for the configuration
        self.grpo_trainer = GRPOTrainer(
            model=self.model,
            args=grpo_config,  # TRL uses 'args' not 'config'
            reward_funcs=reward_func,
            train_dataset=train_dataset,
        )
    
    def update_policy(
        self,
        trajectory_prompts: List[str],
        trajectory_responses: List[str],
        judgment: BrowserJudgment = None,
        judgments: List[BrowserJudgment] = None,  # NEW: for multi-execution
    ) -> Dict[str, float]:
        """
        Update policy using trajectory data.
        
        This is the PRIMARY interface for training - it receives completed
        trajectories from external browser interactions.
        
        Supports two modes:
        1. Single execution (backward compatible):
           - judgment: Single BrowserJudgment for the entire trajectory
           
        2. Multi-execution (for TRL GRPO):
           - judgments: List of BrowserJudgment, one per response
           - Enables true GRPO with multiple attempts per prompt
        
        Args:
            trajectory_prompts: List of prompts in the trajectory
            trajectory_responses: List of responses in the trajectory
            judgment: Single browser judgment (for backward compatibility)
            judgments: List of judgments (one per response, for multi-execution)
            
        Returns:
            Dictionary with training metrics
        """
        # Determine mode and validate inputs
        if judgments is not None:
            # Multi-execution mode
            if len(judgments) != len(trajectory_responses):
                raise ValueError(
                    f"Number of judgments ({len(judgments)}) must match "
                    f"number of responses ({len(trajectory_responses)})"
                )
            use_multi_execution = True
            # Compute average reward across all attempts
            rewards = [self.reward_calculator.compute_reward(j) for j in judgments]
            trajectory_reward = sum(rewards) / len(rewards)
        elif judgment is not None:
            # Single execution mode (backward compatible)
            use_multi_execution = False
            trajectory_reward = self.reward_calculator.compute_reward(judgment)
            # Create a list with single judgment for consistent handling
            judgments = [judgment]
            rewards = [trajectory_reward]
        else:
            raise ValueError("Either 'judgment' or 'judgments' must be provided")
        
        # Store judgments for reward function
        self.current_judgments = judgments
        
        try:
            # Create dataset from trajectory
            train_dataset = self._create_dataset_from_trajectory(
                trajectory_prompts,
                trajectory_responses
            )
            
            # Initialize or reinitialize trainer with new data
            self._initialize_trainer(train_dataset)
            
            # Train for one epoch
            train_output = self.grpo_trainer.train()
            
            # Extract metrics
            loss = train_output.training_loss if hasattr(train_output, 'training_loss') else 0.0
            
            # Update statistics
            self.training_stats["total_updates"] += 1
            self.training_stats["total_episodes"] += 1
            self.training_stats["avg_reward"] = (
                0.9 * self.training_stats["avg_reward"] + 0.1 * trajectory_reward
            )
            self.training_stats["avg_loss"] = (
                0.9 * self.training_stats["avg_loss"] + 0.1 * loss
            )
            
            # Build return dict
            result = {
                "algorithm": "trl_grpo",
                "trajectory_reward": trajectory_reward,
                "total_updates": self.training_stats["total_updates"],
                "avg_reward": self.training_stats["avg_reward"],
                "avg_loss": self.training_stats["avg_loss"],
                "num_steps": len(trajectory_prompts),
                "multi_execution": use_multi_execution,
            }
            
            # Add judgment scores (use first judgment for single mode)
            primary_judgment = judgments[0]
            result.update({
                "success": primary_judgment.success if primary_judgment.success is not None else 0.0,
                "efficiency": primary_judgment.efficiency if primary_judgment.efficiency is not None else 0.0,
                "self_correction": primary_judgment.self_correction if primary_judgment.self_correction is not None else 0.0,
            })
            
            # Add multi-execution specific metrics
            if use_multi_execution:
                result["num_attempts"] = len(judgments)
                result["min_reward"] = min(rewards)
                result["max_reward"] = max(rewards)
                result["reward_std"] = float(np.std(rewards)) if len(rewards) > 1 else 0.0
            
            return result
            
        except Exception as e:
            print(f"Error during TRL GRPO update: {e}")
            import traceback
            traceback.print_exc()
            return {
                "algorithm": "trl_grpo",
                "trajectory_reward": trajectory_reward,
                "error": str(e),
                "num_steps": len(trajectory_prompts),
                "multi_execution": use_multi_execution,
            }
        finally:
            # Clear judgment reference
            self.current_judgments = None
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        os.makedirs(path, exist_ok=True)
        
        # Save the model and tokenizer
        self.model.save_pretrained(os.path.join(path, "model"))
        self.tokenizer.save_pretrained(os.path.join(path, "tokenizer"))
        
        # Save training statistics
        with open(os.path.join(path, "training_stats.json"), 'w') as f:
            json.dump(self.training_stats, f, indent=2)
        
        # Save config
        config_dict = {
            "learning_rate": self.config.learning_rate,
            "num_generations": self.config.num_generations,
            "temperature": self.config.temperature,
            "max_new_tokens": self.config.max_new_tokens,
            "beta": self.config.beta,
            "success_weight": self.config.success_weight,
            "efficiency_weight": self.config.efficiency_weight,
            "self_correction_weight": self.config.self_correction_weight,
        }
        with open(os.path.join(path, "grpo_config.json"), 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        # Load training statistics
        stats_path = os.path.join(path, "training_stats.json")
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                self.training_stats = json.load(f)
            print(f"Loaded training stats from {stats_path}")
        
        # Load config
        config_path = os.path.join(path, "grpo_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            # Update config with loaded values
            for key, value in config_dict.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
            print(f"Loaded GRPO config from {config_path}")
        
        print(f"Checkpoint loaded from {path}")


# Convenience function for creating TRL GRPO trainer
def create_trl_grpo_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config: TRLGRPOConfig = None
) -> TRLGRPOTrainer:
    """
    Create a TRL GRPO trainer instance.
    
    Args:
        model: Pre-trained language model
        tokenizer: Tokenizer for the model
        config: GRPO configuration
        
    Returns:
        TRLGRPOTrainer instance
    """
    return TRLGRPOTrainer(model, tokenizer, config)


# Preset configurations for different use cases
def get_grpo_preset(preset_name: str) -> TRLGRPOConfig:
    """
    Get preset GRPO configurations.
    
    Args:
        preset_name: Name of the preset ("default", "fast", "quality", "memory_efficient")
        
    Returns:
        TRLGRPOConfig with preset values
    """
    if preset_name == "default":
        return TRLGRPOConfig()
    
    elif preset_name == "fast":
        # Faster training with fewer generations
        return TRLGRPOConfig(
            learning_rate=3e-5,
            num_generations=2,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            max_new_tokens=256,
        )
    
    elif preset_name == "quality":
        # Higher quality with more generations
        return TRLGRPOConfig(
            learning_rate=1e-5,
            num_generations=8,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            max_new_tokens=512,
            temperature=0.8,
        )
    
    elif preset_name == "memory_efficient":
        # Use gradient checkpointing for lower memory usage
        return TRLGRPOConfig(
            learning_rate=1e-5,
            num_generations=4,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            gradient_checkpointing=True,
            max_new_tokens=256,
        )
    
    else:
        raise ValueError(
            f"Unknown preset: {preset_name}. "
            f"Available presets: default, fast, quality, memory_efficient"
        )
