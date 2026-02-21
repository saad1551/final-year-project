"""
On-Policy Reinforcement Learning Trainer for Browser Navigation Agent.

This module implements multiple RL algorithms:
- REINFORCE: Classic policy gradient with baseline
- PPO: Proximal Policy Optimization with clipped objective
- GRPO: Group Relative Policy Optimization

All algorithms use judgment scores from the judge LLM as rewards.
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from transformers import PreTrainedModel, PreTrainedTokenizer

from insta.configs.judge_config import BrowserJudgment


# Debug logging configuration
DEBUG_RL = False  # Set to True or use --debug flag to enable debug logging

def debug_log(message: str, level: str = "INFO"):
    """Print debug messages if DEBUG_RL is enabled."""
    if DEBUG_RL:
        print(f"[RL-{level}] {message}")


class RLAlgorithm(Enum):
    """Supported RL algorithms."""
    REINFORCE = "reinforce"
    PPO = "ppo"
    GRPO = "grpo"


@dataclass
class RLConfig:
    """Configuration for RL training."""
    
    # Algorithm selection
    algorithm: str = "ppo"  # "reinforce", "ppo", or "grpo"
    
    # Common hyperparameters
    learning_rate: float = 1e-5
    gamma: float = 0.99
    max_grad_norm: float = 1.0
    
    # Reward weights
    success_weight: float = 0.7
    efficiency_weight: float = 0.2
    self_correction_weight: float = 0.1
    
    # REINFORCE specific
    baseline_momentum: float = 0.99
    entropy_coef: float = 0.03
    
    # PPO specific
    ppo_epochs: int = 3
    ppo_clip_epsilon: float = 0.2
    ppo_value_clip: float = 0.2
    ppo_mini_batch_size: int = 4
    kl_penalty_coef: float = 0.01
    target_kl: float = 0.03  # Early stopping threshold (0.03 allows more learning per trajectory)
    
    # GRPO specific
    grpo_group_size: int = 4  # Number of responses per prompt for comparison
    grpo_temperature: float = 1.0
    grpo_beta: float = 0.1  # KL divergence coefficient
    
    def get_algorithm(self) -> RLAlgorithm:
        """Get the algorithm enum from string."""
        return RLAlgorithm(self.algorithm.lower())


class RewardCalculator:
    """Calculates reward from BrowserJudgment scores."""
    
    def __init__(self, config: RLConfig = None):
        self.config = config or RLConfig()
    
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
        
        debug_log(f"Reward Calculation: success={success:.3f}, efficiency={efficiency:.3f}, "
                  f"self_correction={self_correction:.3f} => reward={reward:.4f}")
        
        return reward


class BaseRLAlgorithm(ABC):
    """Abstract base class for RL algorithms."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: RLConfig
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.reward_calculator = RewardCalculator(config)
        
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable_params, lr=config.learning_rate)
        
        self.training_stats = {
            "total_updates": 0,
            "avg_reward": 0.0,
            "avg_loss": 0.0,
        }
    
    def compute_log_probs_and_entropy(
        self,
        prompt_texts: List[str],
        response_texts: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute log probabilities and entropy of responses given prompts with micro-batching.
        
        IMPORTANT: The caller must set model.train() or model.eval() before calling.
        - train mode: gradient checkpointing active, gradients tracked (for policy update)
        - eval mode: no gradient checkpointing, faster (for old log probs / reference)
        """
        import gc
        
        log_probs_list = []
        entropies_list = []
        
        # Process in micro-batches to avoid OOM on long trajectories
        micro_batch_size = 6  # 6 steps at a time (safe for 20GB VRAM with 4-bit 1.7B model)
        
        for batch_start in range(0, len(prompt_texts), micro_batch_size):
            batch_end = min(batch_start + micro_batch_size, len(prompt_texts))
            
            # Clear cache before each micro-batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            for idx in range(batch_start, batch_end):
                prompt = prompt_texts[idx]
                response = response_texts[idx]
                
                max_length = 8192  # Total budget for prompt + response (8K to avoid prompt truncation with history)
                
                # Tokenize response first to know its length (no special tokens)
                response_ids = self.tokenizer(
                    response,
                    return_tensors="pt",
                    add_special_tokens=False,
                    truncation=True,
                    max_length=512  # Cap response at 512 tokens
                )["input_ids"]
                response_len = response_ids.shape[-1]
                
                # Allocate remaining budget to prompt, truncating prompt if needed
                max_prompt_len = max(256, max_length - response_len)
                prompt_ids = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_prompt_len
                )["input_ids"]
                prompt_length = prompt_ids.shape[-1]
                
                # Concatenate prompt + response token IDs directly
                input_ids = torch.cat([prompt_ids, response_ids], dim=-1).to(self.model.device)
                attention_mask = torch.ones_like(input_ids)
                inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
                
                debug_log(f"Step {idx}: prompt_tokens={prompt_length}, response_tokens={response_len}, "
                          f"total_tokens={input_ids.shape[-1]}")
                
                # Warn if response is empty (would give zero gradient signal)
                if response_len == 0:
                    debug_log(f"WARNING: Step {idx} has 0 response tokens! Log prob will be 0.", level="ERROR")
                
                # Clean up intermediate tensors
                del prompt_ids, response_ids
                
                # NOTE: caller must set train/eval mode before calling this method
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                response_logits = logits[:, prompt_length-1:-1, :]
                response_targets = inputs["input_ids"][:, prompt_length:]
                
                log_probs = F.log_softmax(response_logits, dim=-1)
                
                token_log_probs = torch.gather(
                    log_probs,
                    dim=-1,
                    index=response_targets.unsqueeze(-1)
                ).squeeze(-1)
                
                # Use mean (not sum) to normalize by response length,
                # preventing longer responses from dominating the gradient signal
                sequence_log_prob = token_log_probs.mean(dim=-1)
                log_probs_list.append(sequence_log_prob)
                
                probs = F.softmax(response_logits, dim=-1)
                entropy = -(probs * log_probs).sum(dim=-1).mean()
                entropies_list.append(entropy)
                
                # Aggressive cleanup after each step
                del inputs, outputs, logits, response_logits, response_targets
                del log_probs, token_log_probs, probs
                
            # Clear cache after each micro-batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        stacked_log_probs = torch.stack(log_probs_list)
        stacked_entropies = torch.stack(entropies_list)
        
        # Clear intermediate lists to free memory
        log_probs_list.clear()
        entropies_list.clear()
        
        debug_log(f"Log Probs: shape={stacked_log_probs.shape}, "
                  f"values=[{', '.join([f'{x:.2f}' for x in stacked_log_probs.flatten()[:5].tolist()])}...]")
        debug_log(f"Entropies: shape={stacked_entropies.shape}, mean={stacked_entropies.mean().item():.4f}")
        
        return stacked_log_probs, stacked_entropies
    
    def compute_discounted_rewards(
        self,
        rewards: List[float],
        normalize: bool = True
    ) -> torch.Tensor:
        """Compute discounted cumulative rewards for each step."""
        discounted = []
        running_reward = 0.0
        
        for r in reversed(rewards):
            running_reward = r + self.config.gamma * running_reward
            discounted.insert(0, running_reward)
        
        discounted_tensor = torch.tensor(discounted, dtype=torch.float32)
        
        if normalize and len(discounted) > 1:
            std = discounted_tensor.std()
            if std > 1e-8:
                discounted_tensor = (discounted_tensor - discounted_tensor.mean()) / std
        
        return discounted_tensor.to(self.model.device)
    
    @abstractmethod
    def update_policy(
        self,
        trajectory_prompts: List[str],
        trajectory_responses: List[str],
        judgment: BrowserJudgment
    ) -> Dict[str, float]:
        """Perform policy update. Implemented by subclasses."""
        pass
    
    def _update_stats(self, loss: float, reward: float):
        """Update running training statistics."""
        self.training_stats["total_updates"] += 1
        self.training_stats["avg_reward"] = (
            0.9 * self.training_stats["avg_reward"] + 0.1 * reward
        )
        self.training_stats["avg_loss"] = (
            0.9 * self.training_stats["avg_loss"] + 0.1 * loss
        )


class REINFORCEAlgorithm(BaseRLAlgorithm):
    """
    REINFORCE algorithm with baseline.
    
    Classic policy gradient that uses the reward signal to weight
    the gradient of log probabilities.
    
    Loss = -E[log π(a|s) * (R - baseline)]
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: RLConfig
    ):
        super().__init__(model, tokenizer, config)
        self.baseline = 0.0
    
    def update_baseline(self, reward: float):
        """Update the running baseline with exponential moving average."""
        self.baseline = (
            self.config.baseline_momentum * self.baseline +
            (1 - self.config.baseline_momentum) * reward
        )
    
    def update_policy(
        self,
        trajectory_prompts: List[str],
        trajectory_responses: List[str],
        judgment: BrowserJudgment
    ) -> Dict[str, float]:
        """Perform REINFORCE update with baseline."""
        debug_log("="*50)
        debug_log("REINFORCE Update Starting")
        debug_log(f"Number of trajectory steps: {len(trajectory_prompts)}")
        
        # Step 1: Compute reward from judgment
        trajectory_reward = self.reward_calculator.compute_reward(judgment)
        debug_log(f"Step 1 - Trajectory Reward: {trajectory_reward:.4f}")
        
        # Step 2: Create per-step rewards (sparse terminal reward for proper credit assignment)
        num_steps = len(trajectory_prompts)
        step_rewards = [0.0] * (num_steps - 1) + [trajectory_reward]
        debug_log(f"Step 2 - Step Rewards: sparse terminal, final step={trajectory_reward:.4f}")
        
        # Step 3: Compute discounted rewards
        discounted_rewards = self.compute_discounted_rewards(step_rewards)
        debug_log(f"Step 3 - Discounted Rewards: shape={discounted_rewards.shape}, "
                  f"mean={discounted_rewards.mean().item():.4f}, std={discounted_rewards.std().item():.4f}")
        
        # Step 4: Compute advantages
        advantages = discounted_rewards - self.baseline
        debug_log(f"Step 4 - Advantages: mean={advantages.mean().item():.4f}, "
                  f"baseline={self.baseline:.4f}")
        
        # Step 5: Compute log probabilities
        debug_log("Step 5 - Computing log probabilities...")
        self.model.train()  # Ensure train mode for gradient checkpointing & gradient tracking
        log_probs, entropies = self.compute_log_probs_and_entropy(
            trajectory_prompts,
            trajectory_responses
        )
        debug_log(f"Step 5 - Log Probs computed: shape={log_probs.shape}")
        
        # Clear cache after forward passes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Step 6: Compute losses
        policy_loss = -(log_probs * advantages).mean()
        entropy_bonus = entropies.mean()
        total_loss = policy_loss - self.config.entropy_coef * entropy_bonus
        
        debug_log(f"Step 6 - Losses: policy_loss={policy_loss.item():.4f}, "
                  f"entropy={entropy_bonus.item():.4f}, total_loss={total_loss.item():.4f}")
        
        # Check for NaN/Inf
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            debug_log("WARNING: Loss is NaN or Inf!", level="ERROR")
            debug_log(f"  log_probs contains NaN: {torch.isnan(log_probs).any()}")
            debug_log(f"  advantages contains NaN: {torch.isnan(advantages).any()}")
        
        # Step 7: Backward pass
        debug_log("Step 7 - Running backward pass...")
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Check gradients
        total_grad_norm = 0.0
        num_params_with_grad = 0
        for p in self.model.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.data.norm(2).item() ** 2
                num_params_with_grad += 1
        total_grad_norm = total_grad_norm ** 0.5
        debug_log(f"Step 7 - Gradient norm before clipping: {total_grad_norm:.4f} "
                  f"({num_params_with_grad} params with gradients)")
        
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm
        )
        debug_log(f"Step 7 - Gradients clipped to max_norm={self.config.max_grad_norm}")
        
        # Step 8: Optimizer step
        debug_log("Step 8 - Applying optimizer step...")
        self.optimizer.step()
        debug_log("Step 8 - Optimizer step completed ✓")
        
        # Clear gradients and cache after update
        self.optimizer.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Step 9: Update baseline
        old_baseline = self.baseline
        self.update_baseline(trajectory_reward)
        debug_log(f"Step 9 - Baseline updated: {old_baseline:.4f} -> {self.baseline:.4f}")
        
        self._update_stats(total_loss.item(), trajectory_reward)
        debug_log(f"Training stats: total_updates={self.training_stats['total_updates']}, "
                  f"avg_reward={self.training_stats['avg_reward']:.4f}")
        debug_log("REINFORCE Update Complete")
        debug_log("="*50)
        
        return {
            "algorithm": "reinforce",
            "policy_loss": policy_loss.item(),
            "entropy": entropy_bonus.item(),
            "total_loss": total_loss.item(),
            "trajectory_reward": trajectory_reward,
            "baseline": self.baseline,
            "num_steps": num_steps,
            "grad_norm": total_grad_norm,
        }


class PPOAlgorithm(BaseRLAlgorithm):
    """
    Proximal Policy Optimization (PPO) with clipped objective.
    
    Uses a clipped surrogate objective to prevent large policy updates:
    
    L_CLIP = E[min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A)]
    
    where r(θ) = π_new(a|s) / π_old(a|s) is the probability ratio.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: RLConfig
    ):
        super().__init__(model, tokenizer, config)
        self.baseline = 0.0
    
    @torch.no_grad()
    def compute_old_log_probs(
        self,
        prompt_texts: List[str],
        response_texts: List[str]
    ) -> torch.Tensor:
        """Compute log probs under the old policy (before update).
        
        Uses eval mode to disable gradient checkpointing (faster, no warnings).
        """
        was_training = self.model.training
        self.model.eval()  # eval mode disables gradient checkpointing
        log_probs, _ = self.compute_log_probs_and_entropy(prompt_texts, response_texts)
        if was_training:
            self.model.train()  # restore train mode for subsequent calls
        return log_probs.detach()
    
    def update_policy(
        self,
        trajectory_prompts: List[str],
        trajectory_responses: List[str],
        judgment: BrowserJudgment
    ) -> Dict[str, float]:
        """Perform PPO update with clipped objective and memory optimizations."""
        import gc
        
        trajectory_reward = self.reward_calculator.compute_reward(judgment)
        num_steps = len(trajectory_prompts)
        
        # Handle empty trajectories gracefully
        if num_steps == 0:
            print("[PPO] Warning: Empty trajectory, skipping update")
            return {
                "algorithm": "ppo",
                "policy_loss": 0.0,
                "entropy": 0.0,
                "kl_divergence": 0.0,
                "total_loss": 0.0,
                "trajectory_reward": trajectory_reward,
                "baseline": self.baseline,
                "num_steps": 0,
                "ppo_epochs_run": 0,
            }
        
        # Sparse terminal reward: only the final step gets the reward
        step_rewards = [0.0] * (num_steps - 1) + [trajectory_reward]
        discounted_rewards = self.compute_discounted_rewards(step_rewards)
        advantages = discounted_rewards - self.baseline
        
        # Clear cache before computing old log probs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Compute old log probs before any updates
        old_log_probs = self.compute_old_log_probs(
            trajectory_prompts, 
            trajectory_responses
        )
        
        total_policy_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        last_grad_norm = 0.0
        num_epochs_run = 0
        
        # Scale PPO epochs based on trajectory length for memory efficiency
        effective_ppo_epochs = (
            min(4, self.config.ppo_epochs)
            if num_steps > 15
            else min(6, self.config.ppo_epochs)
        )
        
        # Multiple PPO epochs
        for epoch in range(effective_ppo_epochs):
            # Aggressive memory cleanup before each epoch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            self.model.train()  # Ensure train mode for gradient checkpointing & gradient tracking
            new_log_probs, entropies = self.compute_log_probs_and_entropy(
                trajectory_prompts,
                trajectory_responses
            )
            
            # Compute probability ratio
            log_ratio = new_log_probs - old_log_probs
            ratio = torch.exp(log_ratio)
            
            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio,
                1 - self.config.ppo_clip_epsilon,
                1 + self.config.ppo_clip_epsilon
            ) * advantages
            
            policy_loss = -torch.min(surr1, surr2).mean()
            entropy_bonus = entropies.mean()
            
            # KL divergence for early stopping
            approx_kl = ((ratio - 1) - log_ratio).mean()
            
            if approx_kl > self.config.target_kl:
                print(f"[PPO] Early stopping at epoch {epoch} due to reaching target KL")
                # Clean up before breaking
                del new_log_probs, entropies, log_ratio, ratio, surr1, surr2
                break
            
            # Standard PPO: clipping only, no KL penalty (clipping already constrains updates)
            total_loss = (
                policy_loss 
                - self.config.entropy_coef * entropy_bonus
            )
            
            # Check for NaN
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"[PPO] Warning: NaN/Inf loss detected at epoch {epoch}, skipping")
                del new_log_probs, entropies, log_ratio, ratio, surr1, surr2
                continue
            
            self.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            
            # Verify gradients are flowing through LoRA parameters
            total_grad_norm = 0.0
            num_params_with_grad = 0
            for p in self.model.parameters():
                if p.requires_grad and p.grad is not None:
                    total_grad_norm += p.grad.data.norm(2).item() ** 2
                    num_params_with_grad += 1
            total_grad_norm = total_grad_norm ** 0.5
            
            if num_params_with_grad == 0:
                print(f"[PPO] WARNING: No parameters received gradients at epoch {epoch}!")
                print(f"[PPO] This means LoRA weights are NOT being updated.")
            else:
                debug_log(f"PPO epoch {epoch}: grad_norm={total_grad_norm:.6f}, "
                          f"params_with_grad={num_params_with_grad}")
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_entropy += entropy_bonus.item()
            total_kl += approx_kl.item()
            last_grad_norm = total_grad_norm
            num_epochs_run += 1
            
            # Free memory immediately after each epoch
            del new_log_probs, entropies, log_ratio, ratio, surr1, surr2, policy_loss, entropy_bonus, approx_kl, total_loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Final cleanup
        del old_log_probs, advantages, discounted_rewards
        self.optimizer.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Update baseline
        self.baseline = (
            self.config.baseline_momentum * self.baseline +
            (1 - self.config.baseline_momentum) * trajectory_reward
        )
        
        if num_epochs_run == 0:
            num_epochs_run = 1  # Avoid division by zero
        avg_policy_loss = total_policy_loss / num_epochs_run
        self._update_stats(avg_policy_loss, trajectory_reward)
        
        return {
            "algorithm": "ppo",
            "policy_loss": avg_policy_loss,
            "entropy": total_entropy / num_epochs_run,
            "kl_divergence": total_kl / num_epochs_run,
            "total_loss": avg_policy_loss,
            "trajectory_reward": trajectory_reward,
            "baseline": self.baseline,
            "num_steps": num_steps,
            "ppo_epochs_run": num_epochs_run,
            "grad_norm": last_grad_norm if num_epochs_run > 0 else 0.0,
        }


class GRPOAlgorithm(BaseRLAlgorithm):
    """
    Group Relative Policy Optimization (GRPO).
    
    GRPO compares multiple responses for the same prompt and uses
    relative rewards within the group to compute advantages.
    
    This implementation adapts GRPO for single-response trajectories
    by using step-level comparisons within the trajectory.
    
    Key idea: Instead of absolute rewards, use relative performance
    within a group to reduce variance and improve learning signal.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: RLConfig
    ):
        super().__init__(model, tokenizer, config)
        self.reward_history: List[float] = []
        self.reference_log_probs: Optional[torch.Tensor] = None
    
    def compute_group_advantages(
        self,
        rewards: List[float],
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Compute advantages using group-relative normalization.
        
        Uses softmax-normalized rewards within the group.
        """
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        
        # Group normalization using softmax
        normalized_rewards = F.softmax(rewards_tensor / temperature, dim=0)
        
        # Center the advantages
        advantages = normalized_rewards - normalized_rewards.mean()
        
        return advantages.to(self.model.device)
    
    @torch.no_grad()
    def store_reference_log_probs(
        self,
        prompt_texts: List[str],
        response_texts: List[str]
    ):
        """Store reference log probs for KL penalty computation."""
        was_training = self.model.training
        self.model.eval()  # eval mode disables gradient checkpointing
        self.reference_log_probs, _ = self.compute_log_probs_and_entropy(
            prompt_texts, response_texts
        )
        self.reference_log_probs = self.reference_log_probs.detach()
        if was_training:
            self.model.train()
    
    def update_policy(
        self,
        trajectory_prompts: List[str],
        trajectory_responses: List[str],
        judgment: BrowserJudgment
    ) -> Dict[str, float]:
        """Perform GRPO update using relative rewards."""
        trajectory_reward = self.reward_calculator.compute_reward(judgment)
        
        # Store current reward for group comparison
        self.reward_history.append(trajectory_reward)
        
        # Keep only recent rewards for group comparison
        max_history = self.config.grpo_group_size * 2
        if len(self.reward_history) > max_history:
            self.reward_history = self.reward_history[-max_history:]
        
        # Sparse terminal reward: only the final step gets the reward
        num_steps = len(trajectory_prompts)
        step_rewards = [0.0] * (num_steps - 1) + [trajectory_reward]
        
        # Use group-relative advantages
        advantages = self.compute_group_advantages(
            step_rewards,
            temperature=self.config.grpo_temperature
        )
        
        # Store reference for KL computation (first time)
        if self.reference_log_probs is None:
            self.store_reference_log_probs(trajectory_prompts, trajectory_responses)
        
        # Compute current log probs
        self.model.train()  # Ensure train mode for gradient checkpointing & gradient tracking
        log_probs, entropies = self.compute_log_probs_and_entropy(
            trajectory_prompts,
            trajectory_responses
        )
        
        # Policy gradient loss with GRPO-style advantages
        policy_loss = -(log_probs * advantages).mean()
        
        # KL penalty against reference policy
        if self.reference_log_probs is not None and len(self.reference_log_probs) == len(log_probs):
            kl_div = (self.reference_log_probs - log_probs).mean()
        else:
            kl_div = torch.tensor(0.0, device=self.model.device)
        
        entropy_bonus = entropies.mean()
        
        total_loss = (
            policy_loss 
            - self.config.entropy_coef * entropy_bonus
            + self.config.grpo_beta * kl_div
        )
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm
        )
        self.optimizer.step()
        
        # Update reference log probs periodically
        if self.training_stats["total_updates"] % self.config.grpo_group_size == 0:
            self.store_reference_log_probs(trajectory_prompts, trajectory_responses)
        
        self._update_stats(total_loss.item(), trajectory_reward)
        
        # Compute relative rank in recent history
        if len(self.reward_history) > 1:
            sorted_rewards = sorted(self.reward_history, reverse=True)
            rank = sorted_rewards.index(trajectory_reward) + 1
            relative_rank = rank / len(self.reward_history)
        else:
            relative_rank = 0.5
        
        return {
            "algorithm": "grpo",
            "policy_loss": policy_loss.item(),
            "entropy": entropy_bonus.item(),
            "kl_divergence": kl_div.item(),
            "total_loss": total_loss.item(),
            "trajectory_reward": trajectory_reward,
            "relative_rank": relative_rank,
            "num_steps": num_steps,
            "group_size": len(self.reward_history),
        }


class OnPolicyTrainer:
    """
    Factory class for on-policy RL training.
    
    Supports switching between REINFORCE, PPO, and GRPO algorithms
    based on configuration.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: RLConfig = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or RLConfig()
        
        # Create the appropriate algorithm
        self.algorithm = self._create_algorithm()
        
        print(f"Initialized OnPolicyTrainer with {self.config.algorithm.upper()} algorithm")
    
    def _create_algorithm(self) -> BaseRLAlgorithm:
        """Create the algorithm based on config."""
        algo_type = self.config.get_algorithm()
        
        if algo_type == RLAlgorithm.REINFORCE:
            return REINFORCEAlgorithm(self.model, self.tokenizer, self.config)
        elif algo_type == RLAlgorithm.PPO:
            return PPOAlgorithm(self.model, self.tokenizer, self.config)
        elif algo_type == RLAlgorithm.GRPO:
            return GRPOAlgorithm(self.model, self.tokenizer, self.config)
        else:
            raise ValueError(f"Unknown algorithm: {self.config.algorithm}")
    
    @property
    def training_stats(self) -> Dict:
        """Get training statistics from the algorithm."""
        return self.algorithm.training_stats
    
    def update_policy(
        self,
        trajectory_prompts: List[str],
        trajectory_responses: List[str],
        judgment: BrowserJudgment
    ) -> Dict[str, float]:
        """Delegate to the selected algorithm."""
        return self.algorithm.update_policy(
            trajectory_prompts,
            trajectory_responses,
            judgment
        )
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        import os
        os.makedirs(path, exist_ok=True)
        
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        # Save algorithm-specific state
        state = {
            "optimizer_state_dict": self.algorithm.optimizer.state_dict(),
            "training_stats": self.algorithm.training_stats,
            "config": self.config,
            "algorithm_type": self.config.algorithm,
        }
        
        # Add algorithm-specific attributes
        if isinstance(self.algorithm, REINFORCEAlgorithm):
            state["baseline"] = self.algorithm.baseline
        elif isinstance(self.algorithm, PPOAlgorithm):
            state["baseline"] = self.algorithm.baseline
        elif isinstance(self.algorithm, GRPOAlgorithm):
            state["reward_history"] = self.algorithm.reward_history
        
        torch.save(state, f"{path}/trainer_state.pt")
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load trainer state from checkpoint."""
        state = torch.load(f"{path}/trainer_state.pt")
        
        self.algorithm.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.algorithm.training_stats = state["training_stats"]
        
        # Load algorithm-specific attributes
        if isinstance(self.algorithm, REINFORCEAlgorithm) and "baseline" in state:
            self.algorithm.baseline = state["baseline"]
        elif isinstance(self.algorithm, PPOAlgorithm) and "baseline" in state:
            self.algorithm.baseline = state["baseline"]
        elif isinstance(self.algorithm, GRPOAlgorithm) and "reward_history" in state:
            self.algorithm.reward_history = state["reward_history"]
        
        print(f"Trainer state loaded from {path}")


def compute_reward_from_judgment(
    judgment: BrowserJudgment,
    success_weight: float = 0.7,
    efficiency_weight: float = 0.2,
    self_correction_weight: float = 0.1
) -> float:
    """
    Standalone reward function compatible with rl/reward_func.py interface.
    """
    if judgment is None:
        return 0.0
    
    success = judgment.success if judgment.success is not None else 0.0
    efficiency = judgment.efficiency if judgment.efficiency is not None else 0.0
    self_correction = judgment.self_correction if judgment.self_correction is not None else 0.0
    
    reward = (
        success_weight * success +
        efficiency_weight * efficiency +
        self_correction_weight * self_correction
    )
    
    return reward


# Convenience function to create trainer with specific algorithm
def create_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    algorithm: str = "reinforce",
    **kwargs
) -> OnPolicyTrainer:
    """
    Create an OnPolicyTrainer with the specified algorithm.
    
    Args:
        model: The language model to train.
        tokenizer: The tokenizer for the model.
        algorithm: One of "reinforce", "ppo", or "grpo".
        **kwargs: Additional config parameters.
    
    Returns:
        Configured OnPolicyTrainer instance.
    
    Example:
        trainer = create_trainer(model, tokenizer, algorithm="ppo", learning_rate=1e-5)
    """
    config = RLConfig(algorithm=algorithm, **kwargs)
    return OnPolicyTrainer(model, tokenizer, config)
