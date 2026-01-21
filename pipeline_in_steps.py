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


MODEL_NAME = "btrabucco/Insta-Qwen3-1.7B-SFT"
BROWSER_SERVER_URL = "http://localhost:3000"
MAX_TRAJECTORY_STEPS = 30
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


def load_language_model():
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto"
    )
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


def get_observation_with_retry(client):
    observation = client.observation()
    if isinstance(observation, BrowserObservation):
        return observation
    
    for _ in range(OBSERVATION_RETRY_ATTEMPTS):
        time.sleep(OBSERVATION_RETRY_DELAY_SECONDS)
        observation = client.observation()
        if isinstance(observation, BrowserObservation):
            return observation
    
    return observation


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
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    
    outputs = model.generate(**inputs, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id)
    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
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


def run_trajectory(task_data: dict, trainer: OnPolicyTrainer = None, enable_rl_update: bool = True, rl_config: RLConfig = None):
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
    try:
        tokenizer, model = load_language_model()
        print("Policy and tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading policy/tokenizer: {e}")
        return None
    
    if enable_rl_update and trainer is None:
        print("Initializing On-Policy RL Trainer...")
        if rl_config is None:
            rl_config = RLConfig()
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
            actions=trajectory_action_jsons
        )
        print("Judgment received:")
        print_judgment(judgment)
        
        rl_update_stats = None
        if enable_rl_update and trainer is not None and len(trajectory_prompts) > 0:
            print("\n--- Performing On-Policy RL Update ---")
            
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
            "trajectory_observations": trajectory_observations,
            "trajectory_actions": trajectory_actions,
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
    parser.add_argument("--algorithm", type=str, default="reinforce",
                        choices=["reinforce", "ppo", "grpo"],
                        help="RL algorithm to use (reinforce, ppo, or grpo)")
    parser.add_argument("--enable_rl", action="store_true", default=True,
                        help="Enable on-policy RL updates")
    parser.add_argument("--disable_rl", action="store_true", default=False,
                        help="Disable on-policy RL updates")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--save_every", type=int, default=5,
                        help="Save checkpoint every N trajectories")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate for RL updates")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Starting index in the dataset")
    
    # PPO-specific arguments
    parser.add_argument("--ppo_epochs", type=int, default=4,
                        help="Number of PPO epochs per update")
    parser.add_argument("--ppo_clip_epsilon", type=float, default=0.2,
                        help="PPO clipping epsilon")
    
    # GRPO-specific arguments
    parser.add_argument("--grpo_group_size", type=int, default=4,
                        help="GRPO group size for comparison")
    parser.add_argument("--grpo_beta", type=float, default=0.1,
                        help="GRPO KL divergence coefficient")
    
    # Debug options
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Enable detailed RL debug logging")
    
    args = parser.parse_args()
    
    enable_rl_update = args.enable_rl and not args.disable_rl
    
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
    
    df = pd.read_csv("data/insta-150k-test.csv")
    
    trainer = None
    total_rewards = []
    
    print(f"=== Starting RL Training Loop ===")
    print(f"Algorithm: {args.algorithm.upper()}")
    print(f"Trajectories: {args.num_trajectories}")
    print(f"RL Updates Enabled: {enable_rl_update}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Checkpoint Dir: {args.checkpoint_dir}")
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
            trainer=trainer, 
            enable_rl_update=enable_rl_update,
            rl_config=rl_config
        )
        
        if trajectory_result:
            print(f"\n--- Trajectory {i+1} Finished ---")
            print(f"Instruction: {trajectory_result['task_instruction']}")
            print(f"Total Steps: {len(trajectory_result['trajectory_actions'])}")
            
            trainer = trajectory_result.get('trainer')
            
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