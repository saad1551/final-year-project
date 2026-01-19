import json
import os
import time
from datetime import datetime

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from client import BrowserClient
from markdown import get_markdown_tree, render_markdown_tree
from insta.agent_prompts.base_agent_prompt import BaseAgentPrompt, AGENT_PATTERN
from configs.browser_config import BrowserObservation, NodeMetadata, BrowserConfig
from insta.configs.agent_config import BrowserAction
from utils import safe_call, BrowserStatus
from judge_integration import judge_trajectory, print_judgment


MODEL_NAME = "btrabucco/Insta-Qwen3-1.7B-SFT"
BROWSER_SERVER_URL = "http://localhost:3000"
MAX_TRAJECTORY_STEPS = 30
MAX_HISTORY_STEPS = 2
SCREENSHOT_OUTPUT_DIR = "visualization_output"
PAGE_LOAD_WAIT_SECONDS = 5
OBSERVATION_RETRY_ATTEMPTS = 5
OBSERVATION_RETRY_DELAY_SECONDS = 3


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
    
    return response_text, agent_prompt


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


def run_trajectory(task_data: dict):
    print("--- Initialization: Policy Setup ---")
    try:
        tokenizer, model = load_language_model()
        print("Policy and tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading policy/tokenizer: {e}")
        return None
    
    print("\n--- Step 1: Start Environment and Retrieve Initial State ---")
    
    start_url = task_data['website']
    if not start_url.startswith('http'):
        start_url = 'https://' + start_url
    
    trajectory_observations = []
    trajectory_actions = []
    trajectory_markdown_observations = []
    trajectory_action_jsons = []
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
            response_text, agent_prompt = generate_action_from_observation(
                tokenizer, model, task_data['instruction'], 
                current_observation, markdown_content, history
            )
            
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

        return {
            "task_instruction": task_data['instruction'],
            "trajectory_observations": trajectory_observations,
            "trajectory_actions": trajectory_actions,
            "judgment": judgment
        }
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    finally:
        if client and client.session_id:
            client.close()
            print("\nBrowser session closed.")


if __name__ == "__main__":
    df = pd.read_csv("data/insta-150k-test.csv")
    task_row = df.iloc[14].to_dict()
    
    trajectory_result = run_trajectory(task_row)
    
    if trajectory_result:
        print("\n--- Trajectory Finished ---")
        print(f"Instruction: {trajectory_result['task_instruction']}")
        print(f"Total Steps: {len(trajectory_result['trajectory_actions'])}")
    else:
        print("Trajectory generation failed.")