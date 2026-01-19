import sys
import json
import argparse
import base64
import io
import os
import pandas as pd
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from client import BrowserClient
from markdown import get_markdown_tree, render_markdown_tree
from insta.agent_prompts.base_agent_prompt import BaseAgentPrompt, AGENT_PATTERN
from configs.browser_config import BrowserObservation, NodeMetadata, BrowserConfig
from insta.configs.agent_config import BrowserAction
from utils import safe_call, BrowserStatus
import time
import torch


MAX_STEPS = 30

MAX_HISTORY_STEPS = 2

CACHE_DIR = "/media/tukl/ee279b7d-bb8a-4a20-8bf9-90b2c542efcc/Saad/final-year-project/hf_cache"

VISUALIZATION_DIR = "visualization_output"

def save_screenshot(observation: BrowserObservation, step: int, prefix: str = "step"):
    """
    Save a screenshot from the browser observation for visualization.
    
    Args:
        observation: BrowserObservation containing the screenshot
        step: Step number in the trajectory
        prefix: Prefix for the filename
    """
    if observation is None or observation.screenshot is None:
        return
    
    # Create visualization directory if it doesn't exist
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)
    
    # Generate filename with timestamp and step number
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{step:03d}_{timestamp}.png"
    filepath = os.path.join(VISUALIZATION_DIR, filename)
    
    try:
        observation.screenshot.save(filepath)
        print(f"Screenshot saved: {filepath}")
    except Exception as e:
        print(f"Failed to save screenshot: {e}")

def extract_last_json_object(json_text: str) -> dict:
    """
    Safely extract the LAST complete JSON object from a string.
    Handles:
    - Multiple JSON objects
    - Extra text before/after JSON
    - Nested objects
    - LLM log-style outputs
    """
    json_text = json_text.strip()

    last_json_str = None
    brace_count = 0
    start_idx = None

    for i in range(len(json_text)):
        if json_text[i] == '{':
            if brace_count == 0:
                start_idx = i  # potential new JSON start
            brace_count += 1

        elif json_text[i] == '}':
            brace_count -= 1

            if brace_count == 0 and start_idx is not None:
                # Found a complete JSON object
                candidate = json_text[start_idx:i + 1]
                try:
                    json.loads(candidate)  # validate it
                    last_json_str = candidate  # ✅ keep overwriting → LAST one wins
                except json.JSONDecodeError:
                    pass  # ignore invalid JSON blocks

    if last_json_str is not None:
        return json.loads(last_json_str)

    raise ValueError("No valid JSON object found in the input.")

def image_to_base64(image):
    """Convert PIL Image to base64 string for JSON serialization."""
    if image is None:
        return None
    
    buffer = io.BytesIO()
    # image is a PIL Image object
    image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str


def convert_metadata_dict_to_objects(metadata_dict):
    """Convert metadata dict to NodeMetadata objects."""
    if not metadata_dict:
        return {}
    
    converted = {}
    for node_id, metadata in metadata_dict.items():
        if isinstance(metadata, dict):
            # Convert dict to NodeMetadata object
            converted[node_id] = NodeMetadata(**metadata)
        else:
            # Already a NodeMetadata object
            converted[node_id] = metadata
    
    return converted


def convert_to_markdown(observation_data):
    """
    Convert HTML observation to markdown format.
    
    Args:
        observation_data: Dictionary containing raw_html, metadata, etc.
        
    Returns:
        str: Markdown representation of the page
    """
    try:
        # Extract the raw HTML and metadata
        raw_html = observation_data.get('raw_html', '')
        metadata = observation_data.get('metadata', {})
        
        # print(f"Raw HTML length: {len(raw_html)}")
        # print(f"Metadata type: {type(metadata)}, keys: {list(metadata.keys()) if isinstance(metadata, dict) else 'not dict'}")
        
        if not raw_html:
            return "No HTML content found."
        
        # Convert metadata dict to NodeMetadata objects if needed
        node_metadata = convert_metadata_dict_to_objects(metadata)
        
        print(f"Node metadata converted, length: {len(node_metadata)}")
        
        # Get markdown tree
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

        # Render markdown tree
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


# --- Configuration ---
MODEL_NAME = "btrabucco/Insta-Qwen3-1.7B-SFT"
BROWSER_SERVER_URL = "http://localhost:3000"


# --- Data Simulation: First Row of Insta Dataset ---
# Simulating: df.iloc[0]

df = pd.read_csv("data/insta-150k-test.csv")

first_row = df.iloc[13].to_dict()


def run_trajectory(task_data: dict):
    """
    Performs Initialization, and then loops through the agent's decision process.
    """
    print("--- Initialization: Policy Setup ---")
    try:
        # Load tokenizer and model from Hugging Face
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quantization_config,
            device_map="auto"
        )
        
        # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
        # model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR).to(device)

        print("Policy and tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading policy/tokenizer: {e}")
        return None
        
    print("\n--- Step 1: Start Environment and Retrieve Initial State (s_1) ---")
    
    start_url = task_data['website']
    if not start_url.startswith('http'):
        start_url = 'https://' + start_url
    
    # Initialize the browser client
    browser_config = BrowserConfig(playwright_url=BROWSER_SERVER_URL, screen_width=1920, screen_height=1080)
    client = BrowserClient(browser_config)
    
    trajectory_observations = []
    trajectory_actions = []
    history = []

    try:
        # Start a new session
        status = client.start()
        if status == BrowserStatus.ERROR:
            raise Exception("Failed to start browser session.")
        print(f"Browser session started with ID: {client.session_id}")
        
        # Navigate to the URL
        status = client.goto(start_url)
        if status == BrowserStatus.ERROR:
            raise Exception(f"Failed to navigate to {start_url}.")
        print(f"Navigated to {start_url}")
        
        # Give the page time to load
        print("Waiting for page to load...")
        time.sleep(5)
        
        # Get initial observation
        current_observation = client.observation()
        if not isinstance(current_observation, BrowserObservation):
            # raise Exception(f"Failed to get observation: {current_observation}")
            for i in range(5):
                time.sleep(3)
                current_observation = client.observation()
                if isinstance(current_observation, BrowserObservation):
                    break
        
        print("Initial observation received.")
        save_screenshot(current_observation, 0, "initial")
        
        for step in range(MAX_STEPS):
            print(f"\n--- Step {step + 2} ---")
            
            if device == "cuda":
                torch.cuda.empty_cache()

            # --- Convert HTML observation to Markdown ---
            print("Converting HTML to Markdown...")
            markdown_content = convert_to_markdown(current_observation.__dict__)
            if not markdown_content:
                print("Markdown conversion failed. Stopping.")
                break
            print("Markdown content generated.")
            trajectory_observations.append(current_observation)
            save_screenshot(current_observation, step + 1, "observation")

            # --- Predict Action with LLM ---
            print("Predicting Action from State...")
            agent_prompt = BaseAgentPrompt()

            # Format history for the prompt
            history_str = ""
            if history:
                history_str += "You have already taken the following steps:\n"
                for i, (obs, act) in enumerate(history):
                    history_str += f"--- Step {i+1} ---\n"
                    history_str += f"Observation:\n{obs}\n"
                    history_str += f"Action:\n```json\n{act}\n```\n"
                history_str += "--- Current Step ---\n"

            prompt_with_history = f"{history_str}You are at {current_observation.current_url} observing the viewport:\n\n{markdown_content}"

            user_prompt = agent_prompt.user_prompt_template.format(
                instruction=task_data['instruction'],
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
            
            # Extract all json blocks and parse them, keeping only the last successfully parsed action
            matches = list(AGENT_PATTERN.finditer(response_text))
            predicted_action = None
            json_text = None
            
            if matches:
                # Try parsing each JSON block from last to first, use the last one that successfully parses
                for match in matches:
                    try:
                        candidate_json = match.group("json")
                        candidate_action = agent_prompt.parse_action(f"```json\n{candidate_json}\n```")
                        if isinstance(candidate_action, BrowserAction):
                            predicted_action = candidate_action
                            json_text = candidate_json
                            break
                    except (ValueError, json.JSONDecodeError) as e:
                        continue
            
            if not predicted_action:
                print("LLM response did not contain a valid JSON block that could be parsed. Stopping.")
                print("Full response:", response_text)
                break

            print(f"LLM generated action (JSON):\n{json_text}")
            print(f"Parsed Action function calls: {predicted_action.function_calls}")
            trajectory_actions.append(predicted_action)

            # Update history with the observation and the action taken
            history.append((markdown_content, json_text))

            if len(history) > MAX_HISTORY_STEPS:
                history = history[-MAX_HISTORY_STEPS:]

            # --- Check for Stop Action ---
            try:
                action_dict = extract_last_json_object(json_text)
                action_key = action_dict.get("action_key")
                print(f"Extracted action_key: {action_key}")
                if action_key in ["stop", "exit"]:
                    print("Stop action received. Ending trajectory.")
                    break
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Warning: Could not parse action_key from JSON: {e}")
                # Continue with execution if we can't parse the action_key

            # --- Execute Action ---
            print("Executing Action...")
            status = client.action(predicted_action.function_calls)
            if status == BrowserStatus.ERROR:
                print("Failed to execute action. Stopping.")
                break
            print("Action executed successfully.")
            
            # Get the new observation for the next loop iteration
            current_observation = client.observation()
            if not isinstance(current_observation, BrowserObservation):
                for i in range(5):
                    time.sleep(3)
                    current_observation = client.observation()
                    if isinstance(current_observation, BrowserObservation):
                        break
            print("Received next observation.")
            save_screenshot(current_observation, step + 2, "after_action")

        return {
            "task_instruction": task_data['instruction'],
            "trajectory_observations": trajectory_observations,
            "trajectory_actions": trajectory_actions
        }
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    finally:
        # Clean up the session
        if client.session_id:
            client.close()
            print("\nBrowser session closed.")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    trajectory_result = run_trajectory(first_row)
    
    if trajectory_result:
        print("\n--- Trajectory Finished ---")
        print(f"Instruction: {trajectory_result['task_instruction']}")
        print(f"Total Steps: {len(trajectory_result['trajectory_actions'])}")
        # You can add more detailed printing of the trajectory here if needed
    else:
        print("Trajectory generation failed.")