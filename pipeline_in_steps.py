import json
import os
import time
import argparse
import sys
from datetime import datetime

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

import rl_trainer
from client import BrowserClient
from markdown import get_markdown_tree, render_markdown_tree
from insta.agent_prompts.base_agent_prompt import BaseAgentPrompt, AGENT_PATTERN
from configs.browser_config import BrowserObservation, NodeMetadata, BrowserConfig
from insta.configs.agent_config import BrowserAction
from utils import safe_call, BrowserStatus
from judge_integration import judge_trajectory, print_judgment
from rl_trainer import OnPolicyTrainer, RLConfig, compute_reward_from_judgment
from rl_sb3_ppo import SB3PPOTrainer
from rl_sb3_config_examples import get_config as get_sb3_config
from observability import ObservabilityLogger


MODEL_NAME = "btrabucco/Insta-Qwen3-1.7B-SFT"
BROWSER_SERVER_URL = "http://localhost:3000"
MAX_TRAJECTORY_STEPS = 15  # Reduced from 30 to fit in 8GB VRAM with PPO
MAX_HISTORY_STEPS = 2
SCREENSHOT_OUTPUT_DIR = "visualization_output"
PAGE_LOAD_WAIT_SECONDS = 5
OBSERVATION_RETRY_ATTEMPTS = 5
OBSERVATION_RETRY_DELAY_SECONDS = 3

# Debug configuration
DEBUG_PIPELINE = False  # Controlled by --debug flag

def pipeline_debug(message: str, level: str = "INFO"):
    """Print debug messages for pipeline if DEBUG_PIPELINE is enabled."""
    if DEBUG_PIPELINE:
        print(f"[PIPELINE-{level}] {message}")


def save_screenshot(observation: BrowserObservation, step: int, prefix: str = "step"):
    if observation is None or observation.screenshot is None:
        return
    
    os.makedirs(SCREENSHOT_OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{step:03d}_{timestamp}.png"
    filepath = os.path.join(SCREENSHOT_OUTPUT_DIR, filename)
    
    try:
        observation.screenshot.save(filepath)
        print(f"Screenshot saved: {filepath}")
    except Exception as e:
        print(f"Failed to save screenshot: {e}")


def extract_first_json_object(text: str) -> dict:
    text = text.strip()
    brace_count = 0
    start_idx = None

    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx is not None:
                candidate = text[start_idx:i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    start_idx = None

    raise ValueError("No valid JSON object found in the input.")


def convert_metadata_to_node_objects(metadata_dict):
    if not metadata_dict:
        return {}
    
    return {
        node_id: NodeMetadata(**meta) if isinstance(meta, dict) else meta
        for node_id, meta in metadata_dict.items()
    }


def convert_html_to_markdown(observation_data):
    try:
        raw_html = observation_data.get('raw_html', '')
        metadata = observation_data.get('metadata', {})
        
        if not raw_html:
            return "No HTML content found."
        
        node_metadata = convert_metadata_to_node_objects(metadata)
        print(f"Node metadata converted, length: {len(node_metadata)}")
        
        markdown_nodes = safe_call(
            get_markdown_tree,
            raw_html,
            node_metadata,
            catch_errors=True,
            log_errors=True,
            max_errors=1
        )
        
        if markdown_nodes is BrowserStatus.ERROR:
            return "Failed to get markdown tree."

        markdown_text = safe_call(
            render_markdown_tree,
            markdown_nodes,
            catch_errors=True,
            log_errors=True,
            max_errors=1
        )

        if markdown_text is BrowserStatus.ERROR:
            return "Failed to render markdown tree."
            
        return " ".join(markdown_text)
        
    except Exception as e:
        print(f"Error in markdown conversion: {e}")
        return f"Error in markdown conversion: {e}"


# def load_language_model():
#     quantization_config = BitsAndBytesConfig(load_in_8bit=True)
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_NAME,
#         quantization_config=quantization_config,
#         device_map="auto"
#     )
#     return tokenizer, model

def load_language_model():
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    
    # Switch to 4-bit for 8GB GPU headroom
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
        dtype=torch.float16,
    )
    
    # Prepare model for k-bit training (required for LoRA with quantization)
    model = prepare_model_for_kbit_training(model)
    model.enable_input_require_grads() 

    
    # Configure LoRA for efficient fine-tuning
    lora_config = LoraConfig(
        r=16,  # LoRA rank
        lora_alpha=32,  # LoRA alpha scaling
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Apply LoRA adapters
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters info
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    # Enable gradient checkpointing to save memory during RL update
    model.gradient_checkpointing_enable()
    model.config.use_cache = False  # Must be False for gradient checkpointing
    
    return tokenizer, model


def initialize_browser_session(start_url):
    browser_config = BrowserConfig(
        playwright_url=BROWSER_SERVER_URL,
        screen_width=1920,
        screen_height=1080
    )
    client = BrowserClient(browser_config)
    
    status = client.start()
    if status == BrowserStatus.ERROR:
        raise Exception("Failed to start browser session.")
    print(f"Browser session started with ID: {client.session_id}")
    
    status = client.goto(start_url)
    if status == BrowserStatus.ERROR:
        raise Exception(f"Failed to navigate to {start_url}.")
    print(f"Navigated to {start_url}")
    
    print("Waiting for page to load...")
    time.sleep(PAGE_LOAD_WAIT_SECONDS)
    
    return client


def get_observation_with_retry(client, max_wait_seconds=60):
    """Get observation with retry logic and timeout."""
    observation = client.observation()
    if isinstance(observation, BrowserObservation):
        return observation
    
    total_wait = 0
    for attempt in range(OBSERVATION_RETRY_ATTEMPTS):
        print(f"  Observation attempt {attempt + 1}/{OBSERVATION_RETRY_ATTEMPTS}...")
        time.sleep(OBSERVATION_RETRY_DELAY_SECONDS)
        total_wait += OBSERVATION_RETRY_DELAY_SECONDS
        
        if total_wait > max_wait_seconds:
            print(f"  Timeout: Waited {total_wait}s for observation, giving up.")
            raise TimeoutError(f"Failed to get observation after {total_wait}s")
        
        observation = client.observation()
        if isinstance(observation, BrowserObservation):
            return observation
        
        print(f"  Observation failed, status: {observation}")
    
    # If we get here, all retries failed
    raise Exception(f"Failed to get observation after {OBSERVATION_RETRY_ATTEMPTS} attempts")


def format_history_for_prompt(history):
    if not history:
        return ""
    
    history_lines = ["You have already taken the following steps:"]
    for i, (obs, act) in enumerate(history):
        history_lines.append(f"--- Step {i+1} ---")
        history_lines.append(f"Observation:\n{obs}")
        history_lines.append(f"Action:\n```json\n{act}\n```")
    history_lines.append("--- Current Step ---")
    
    return "\n".join(history_lines)


def generate_action_from_observation(tokenizer, model, task_instruction, current_observation, markdown_content, history):
    import gc
    
    agent_prompt = BaseAgentPrompt()
    history_str = format_history_for_prompt(history)
    
    prompt_with_history = f"{history_str}You are at {current_observation.current_url} observing the viewport:\n\n{markdown_content}"
    
    user_prompt = agent_prompt.user_prompt_template.format(
        instruction=task_instruction,
        current_url=current_observation.current_url,
        observation=prompt_with_history
    )
    
    messages = [
        {"role": "system", "content": agent_prompt.system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Clear GPU cache before generation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Switch to eval mode for generation (critical for stable output)
    was_training = model.training
    model.eval()
    
    # Temporarily disable gradient checkpointing for inference
    try:
        # Generate with explicit settings for stability
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=8192).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
            )
        
        generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Cleanup
        del inputs, outputs, generated_tokens
        
    finally:
        # Restore training mode if it was previously training
        if was_training:
            model.train()
    
    # Clear cache after generation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return response_text, agent_prompt, prompt_text


def parse_action_from_response(response_text, agent_prompt):
    matches = list(AGENT_PATTERN.finditer(response_text))
    
    for match in matches:
        try:
            candidate_json = match.group("json")
            candidate_action = agent_prompt.parse_action(f"```json\n{candidate_json}\n```")
            if isinstance(candidate_action, BrowserAction):
                return candidate_action, candidate_json
        except (ValueError, json.JSONDecodeError):
            continue
    
    return None, None


def is_stop_action(json_text):
    try:
        action_dict = extract_first_json_object(json_text)
        action_key = action_dict.get("action_key")
        print(f"Extracted action_key: {action_key}")
        return action_key in ["stop", "exit"]
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Warning: Could not parse action_key from JSON: {e}")
        return False


def run_trajectory(task_data: dict, model, tokenizer, trainer: OnPolicyTrainer = None, enable_rl_update: bool = True, rl_config: RLConfig = None):
    """
    Run a trajectory for the given task and optionally perform on-policy RL update.
    
    Args:
        task_data: Dictionary containing 'website' and 'instruction' keys.
        trainer: Optional OnPolicyTrainer instance. If None and enable_rl_update is True,
                 a new trainer will be created.
        enable_rl_update: Whether to perform RL updates after trajectory completion.
        rl_config: Optional RLConfig for trainer initialization. If None, uses defaults.
        
    Returns:
        Dictionary containing trajectory results, judgment, and RL update stats.
    """
    print("--- Initialization: Policy Setup ---")
    print(f"Using model and tokenizer for trajectory execution")
    
    if enable_rl_update and trainer is None:
        print("Initializing On-Policy RL Trainer...")
        if rl_config is None:
            rl_config = RLConfig()
        
        # Check if using SB3 PPO
        if rl_config.algorithm == "sb3_ppo":
            # Use SB3 PPO trainer
            sb3_preset = getattr(rl_config, 'sb3_preset', 'default')
            sb3_config = get_sb3_config(sb3_preset)
            # Override with learning_rate if specified
            sb3_config.learning_rate = rl_config.learning_rate
            trainer = SB3PPOTrainer(model, tokenizer, sb3_config)
            print(f"RL Trainer initialized with SB3 PPO ({sb3_preset} preset).")
        else:
            # Use custom algorithms
            trainer = OnPolicyTrainer(model, tokenizer, rl_config)
            print(f"RL Trainer initialized with {rl_config.algorithm.upper()} algorithm.")
    
    print("\n--- Step 1: Start Environment and Retrieve Initial State ---")
    
    start_url = task_data['website']
    if not start_url.startswith('http'):
        start_url = 'https://' + start_url
    
    trajectory_observations = []
    trajectory_actions = []
    trajectory_markdown_observations = []
    trajectory_action_jsons = []
    trajectory_prompts = []
    trajectory_responses = []
    history = []
    client = None

    try:
        client = initialize_browser_session(start_url)
        current_observation = get_observation_with_retry(client)
        print("Initial observation received.")
        save_screenshot(current_observation, 0, "initial")
        
        for step in range(MAX_TRAJECTORY_STEPS):
            print(f"\n--- Step {step + 2} ---")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("Converting HTML to Markdown...")
            markdown_content = convert_html_to_markdown(current_observation.__dict__)
            if not markdown_content:
                print("Markdown conversion failed. Stopping.")
                break
            print("Markdown content generated.")
            
            trajectory_observations.append(current_observation)
            trajectory_markdown_observations.append(markdown_content)
            save_screenshot(current_observation, step + 1, "observation")

            print("Predicting Action from State...")
            response_text, agent_prompt, prompt_text = generate_action_from_observation(
                tokenizer, model, task_data['instruction'], 
                current_observation, markdown_content, history
            )
            
            trajectory_prompts.append(prompt_text)
            trajectory_responses.append(response_text)
            
            pipeline_debug(f"Step {step+1} - Prompt length: {len(prompt_text)} chars")
            pipeline_debug(f"Step {step+1} - Response length: {len(response_text)} chars")
            pipeline_debug(f"Step {step+1} - Response preview: {response_text[:100]}...")
            
            predicted_action, json_text = parse_action_from_response(response_text, agent_prompt)
            
            if not predicted_action:
                print("LLM response did not contain a valid JSON block. Stopping.")
                print("Full response:", response_text)
                break

            print(f"LLM generated action (JSON):\n{json_text}")
            print(f"Parsed Action function calls: {predicted_action.function_calls}")
            trajectory_actions.append(predicted_action)
            trajectory_action_jsons.append(json_text)

            history.append((markdown_content, json_text))
            if len(history) > MAX_HISTORY_STEPS:
                history = history[-MAX_HISTORY_STEPS:]

            if is_stop_action(json_text):
                print("Stop action received. Ending trajectory.")
                break

            print("Executing Action...")
            status = client.action(predicted_action.function_calls)
            if status == BrowserStatus.ERROR:
                print("Failed to execute action. Stopping.")
                break
            print("Action executed successfully.")
            
            current_observation = get_observation_with_retry(client)
            print("Received next observation.")
            save_screenshot(current_observation, step + 2, "after_action")

        print("\n--- Judging Trajectory ---")
        judgment = judge_trajectory(
            instruction=task_data['instruction'],
            observations=trajectory_markdown_observations,
            actions=trajectory_action_jsons,
            criteria=task_data.get('criteria', ''),
            steps=task_data.get('steps', '')
        )
        print("Judgment received:")
        print_judgment(judgment)
        
        # Clear GPU cache before RL update to free memory from trajectory generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"GPU memory cleared. Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        
        rl_update_stats = None
        if enable_rl_update and trainer is not None and len(trajectory_prompts) > 0:
            print("\n--- Performing On-Policy RL Update ---")

            torch.cuda.empty_cache()
            
            pipeline_debug("="*50)
            pipeline_debug("RL Update Input Summary:")
            pipeline_debug(f"  Number of prompts: {len(trajectory_prompts)}")
            pipeline_debug(f"  Number of responses: {len(trajectory_responses)}")
            pipeline_debug(f"  Judgment: success={judgment.success}, efficiency={judgment.efficiency}, self_correction={judgment.self_correction}")
            for i, (p, r) in enumerate(zip(trajectory_prompts, trajectory_responses)):
                pipeline_debug(f"  Step {i+1}: prompt={len(p)} chars, response={len(r)} chars")
            pipeline_debug("="*50)
            
            reward = compute_reward_from_judgment(judgment)
            print(f"Computed reward from judgment: {reward:.4f}")
            
            import gc

            # ... inside run_trajectory before trainer.update_policy ...
            torch.cuda.empty_cache()
            gc.collect() 
            
            rl_update_stats = trainer.update_policy(
                trajectory_prompts=trajectory_prompts,
                trajectory_responses=trajectory_responses,
                judgment=judgment
            )
            
            print(f"Policy Loss: {rl_update_stats['policy_loss']:.4f}")
            print(f"Entropy: {rl_update_stats['entropy']:.4f}")
            print(f"Total Loss: {rl_update_stats['total_loss']:.4f}")
            print(f"Baseline: {rl_update_stats['baseline']:.4f}")
            print(f"Total Updates: {trainer.training_stats['total_updates']}")
            
            if 'grad_norm' in rl_update_stats:
                pipeline_debug(f"Gradient Norm: {rl_update_stats['grad_norm']:.4f}")

        return {
            "task_instruction": task_data['instruction'],
            "website": start_url,
            "trajectory_observations": trajectory_observations,
            "trajectory_actions": trajectory_actions,
            "trajectory_markdown_observations": trajectory_markdown_observations,
            "trajectory_action_jsons": trajectory_action_jsons,
            "trajectory_prompts": trajectory_prompts,
            "trajectory_responses": trajectory_responses,
            "judgment": judgment,
            "rl_update_stats": rl_update_stats,
            "trainer": trainer
        }
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    finally:
        if client and client.session_id:
            client.close()
            print("\nBrowser session closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run browser navigation with on-policy RL training")
    parser.add_argument("--num_trajectories", type=int, default=1, 
                        help="Number of trajectories to run for training")
    parser.add_argument("--algorithm", type=str, default="sb3_ppo",
                        choices=["reinforce", "ppo", "grpo", "sb3_ppo"],
                        help="RL algorithm to use (reinforce, ppo, grpo, or sb3_ppo)")
    parser.add_argument("--enable_rl", action="store_true", default=True,
                        help="Enable on-policy RL updates")
    parser.add_argument("--disable_rl", action="store_true", default=False,
                        help="Disable on-policy RL updates")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save checkpoint every N trajectories")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate for RL updates")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Starting index in the dataset")
    
    # PPO-specific arguments
    parser.add_argument("--ppo_epochs", type=int, default=6,
                        help="Number of PPO epochs per update")
    parser.add_argument("--ppo_clip_epsilon", type=float, default=0.2,
                        help="PPO clipping epsilon")
    
    # GRPO-specific arguments
    parser.add_argument("--grpo_group_size", type=int, default=4,
                        help="GRPO group size for comparison")
    parser.add_argument("--grpo_beta", type=float, default=0.1,
                        help="GRPO KL divergence coefficient")
    
    # SB3 PPO-specific arguments
    parser.add_argument("--sb3_preset", type=str, default="aggressive",
                        choices=["default", "low_memory", "aggressive", "conservative", "exploration"],
                        help="SB3 PPO configuration preset (only used with sb3_ppo)")
    
    # Debug options
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Enable detailed RL debug logging")
    
    args = parser.parse_args()
    
    enable_rl_update = args.enable_rl and not args.disable_rl
    
    tokenizer, model = load_language_model()
    
    # Set debug flag in rl_trainer module
    rl_trainer.DEBUG_RL = args.debug
    
    # Set debug flag for pipeline (use module reference to avoid global issue)
    current_module = sys.modules[__name__]
    current_module.DEBUG_PIPELINE = args.debug
    
    if args.debug:
        print("[DEBUG] Debug mode enabled for both pipeline and RL trainer")
    
    # Create RL config with command-line arguments
    rl_config = RLConfig(
        algorithm=args.algorithm,
        learning_rate=args.learning_rate,
        ppo_epochs=args.ppo_epochs,
        ppo_clip_epsilon=args.ppo_clip_epsilon,
        grpo_group_size=args.grpo_group_size,
        grpo_beta=args.grpo_beta,
    )
    
    # Add SB3 preset if using sb3_ppo
    if args.algorithm == "sb3_ppo":
        rl_config.sb3_preset = args.sb3_preset
    
    df = pd.read_csv("data/insta-150k-test.csv")
    
    trainer = None
    total_rewards = []
    
    # Initialize trainer from checkpoint if resuming
    if args.resume_from and enable_rl_update:
        print(f"\n=== Resuming from Checkpoint ===")
        print(f"Checkpoint Path: {args.resume_from}")
        
        # Check if using SB3 PPO
        if args.algorithm == "sb3_ppo":
            sb3_preset = args.sb3_preset
            sb3_config = get_sb3_config(sb3_preset)
            sb3_config.learning_rate = args.learning_rate
            trainer = SB3PPOTrainer(model, tokenizer, sb3_config)
            print(f"Initialized SB3 PPO Trainer ({sb3_preset} preset)")
        else:
            trainer = OnPolicyTrainer(model, tokenizer, rl_config)
            print(f"Initialized {args.algorithm.upper()} Trainer")
        
        # Load checkpoint
        trainer.load_checkpoint(args.resume_from)
        print(f"✓ Checkpoint loaded successfully")
        print(f"  Total updates from checkpoint: {trainer.training_stats.get('total_updates', 0)}")
        print()
    
    print(f"=== Starting RL Training Loop ===")
    print(f"Algorithm: {args.algorithm.upper()}")
    print(f"Trajectories: {args.num_trajectories}")
    print(f"RL Updates Enabled: {enable_rl_update}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Checkpoint Dir: {args.checkpoint_dir}")
    if args.resume_from:
        print(f"Resuming From: {args.resume_from}")
    print()
    
    # Initialize observability logger
    obs_logger = ObservabilityLogger()
    print(f"Observability logging enabled: {obs_logger.log_dir}")
    print()
    
    for i in range(args.num_trajectories):
        task_idx = args.start_idx + i
        if task_idx >= len(df):
            print(f"Reached end of dataset at index {task_idx}")
            break
            
        task_row = df.iloc[task_idx].to_dict()
        
        print(f"\n{'='*60}")
        print(f"TRAJECTORY {i+1}/{args.num_trajectories} (Dataset Index: {task_idx})")
        print(f"{'='*60}")
        
        trajectory_result = run_trajectory(
            task_row,
            model,
            tokenizer,
            trainer=trainer, 
            enable_rl_update=enable_rl_update,
            rl_config=rl_config
        )
        
        if trajectory_result:
            print(f"\n--- Trajectory {i+1} Finished ---")
            print(f"Instruction: {trajectory_result['task_instruction']}")
            print(f"Total Steps: {len(trajectory_result['trajectory_actions'])}")
            
            trainer = trajectory_result.get('trainer')
            
            # Log trajectory to observability system
            obs_logger.log_trajectory(
                trajectory_id=i + 1,
                dataset_index=task_idx,
                task_instruction=trajectory_result['task_instruction'],
                website=trajectory_result['website'],
                observations=trajectory_result['trajectory_markdown_observations'],
                action_jsons=trajectory_result['trajectory_action_jsons'],
                judgment=trajectory_result['judgment'],
                rl_stats=trajectory_result.get('rl_update_stats'),
                trainer_stats=trainer.training_stats if trainer else None,
                algorithm=args.algorithm,
                learning_rate=args.learning_rate
            )
            
            if trajectory_result.get('rl_update_stats'):
                reward = trajectory_result['rl_update_stats']['trajectory_reward']
                total_rewards.append(reward)
                print(f"Trajectory Reward: {reward:.4f}")
                print(f"Average Reward (last 10): {sum(total_rewards[-10:])/len(total_rewards[-10:]):.4f}")
            
            if trainer and (i + 1) % args.save_every == 0:
                os.makedirs(args.checkpoint_dir, exist_ok=True)
                checkpoint_path = f"{args.checkpoint_dir}/checkpoint_trajectory_{i+1}"
                trainer.save_checkpoint(checkpoint_path)
                print(f"Checkpoint saved at trajectory {i+1}")
        else:
            print(f"Trajectory {i+1} generation failed.")
    
    if trainer:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        final_checkpoint = f"{args.checkpoint_dir}/final_checkpoint"
        trainer.save_checkpoint(final_checkpoint)
        print(f"\nFinal checkpoint saved to {final_checkpoint}")
    
    print(f"\n=== Training Complete ===")
    print(f"Total Trajectories Completed: {len(total_rewards)}")
    if total_rewards:
        print(f"Average Reward: {sum(total_rewards)/len(total_rewards):.4f}")
        print(f"Max Reward: {max(total_rewards):.4f}")
        print(f"Min Reward: {min(total_rewards):.4f}")
    
    # Print observability summary
    obs_logger.print_summary()