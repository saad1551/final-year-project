import sys
import json
import argparse
import base64
import io
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from client import BrowserClient
from markdown import get_markdown_tree, render_markdown_tree
from insta.agent_prompts.base_agent_prompt import BaseAgentPrompt, AGENT_PATTERN
from configs.browser_config import BrowserObservation, NodeMetadata, BrowserConfig
from insta.configs.agent_config import BrowserAction
from utils import safe_call, BrowserStatus
import time

MAX_STEPS = 30

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

first_row = df.iloc[3].to_dict()


def setup_and_convert_initial_state(task_data: dict):
    """
    Performs Initialization, Step 1, and Step 2 of the RL process.
    """
    print("--- Initialization: Policy Setup ---")
    try:
        # Load tokenizer and model from Hugging Face
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        print("Policy and tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading policy/tokenizer: {e}")
        return None
        
    print("\n--- Step 1: Start Environment and Retrieve Initial State (s_1) ---")
    
    start_url = 'https://' + task_data['website']
    
    # Initialize the browser client
    browser_config = BrowserConfig(playwright_url=BROWSER_SERVER_URL, screen_width=1920, screen_height=1080)
    client = BrowserClient(browser_config)
    
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
        initial_observation = client.observation()
        if not isinstance(initial_observation, BrowserObservation):
            raise Exception(f"Failed to get observation: {initial_observation}")
        
        print("Initial observation received.")
        
        # --- Step 2: Convert HTML observation to Markdown ---
        print("\n--- Step 2: Convert HTML to Markdown (s_1 -> m_1) ---")
        
        markdown_content = convert_to_markdown(initial_observation.__dict__)
        
        if not markdown_content:
            raise Exception("Markdown conversion failed.")
        
        print("Markdown content generated successfully.")

        # --- Step 3: Predict Action with LLM ---
        print("\n--- Step 3: Predict Action a_1 from State m_1 ---")

        agent_prompt = BaseAgentPrompt()

        # Prepare prompt for LLM
        user_prompt = agent_prompt.user_prompt_template.format(
            instruction=task_data['instruction'],
            current_url=initial_observation.current_url,
            observation=markdown_content
        )
        
        # Following the chat template of the model
        messages = [
            {"role": "system", "content": agent_prompt.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = tokenizer(prompt_text, return_tensors="pt")

        # Generate action
        outputs = model.generate(**inputs, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id)
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the json part from the response
        match = AGENT_PATTERN.search(response_text)
        if match:
            json_text = match.group("json")
            print("LLM generated action (JSON):")
            print(json_text)
            
            # Parse the action
            predicted_action = agent_prompt.parse_action(f"```json\n{json_text}\n```")
            if isinstance(predicted_action, BrowserAction):
                print("\nParsed Function Calls:")
                for func_call in predicted_action.function_calls:
                    print(f"- {func_call.dotpath}({func_call.args})")
            else:
                print("Failed to parse LLM response.")
                predicted_action = None
        else:
            print("LLM response did not contain a valid JSON block.")
            print("Full response:", response_text)
            predicted_action = None

        # --- Step 4: Execute Action ---
        if predicted_action:
            print("\n--- Step 4: Execute Action a_1 and get State s_2 ---")
            status = client.action(predicted_action.function_calls)
            if status == BrowserStatus.ERROR:
                print("Failed to execute action.")
            else:
                print("Action executed successfully.")
                
                # Get the new observation
                next_observation = client.observation()
                if not isinstance(next_observation, BrowserObservation):
                    print(f"Failed to get next observation: {next_observation}")
                else:
                    print("Received next observation.")
                    # You could now convert this to markdown and loop
                    # next_markdown = convert_to_markdown(next_observation.__dict__)
                    # print("Next state (markdown) generated.")

        return {
            "task_instruction": task_data['instruction'],
            "initial_markdown": markdown_content,
            "predicted_action": predicted_action,
            "model": model,
            "tokenizer": tokenizer
        }
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    finally:
        # Clean up the session
        if client.session_id:
            client.close()
            print("\nBrowser session closed.")


def run_trajectory(task_data: dict):
    """
    Performs Initialization, and then loops through the agent's decision process.
    """
    print("--- Initialization: Policy Setup ---")
    try:
        # Load tokenizer and model from Hugging Face
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
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
            raise Exception(f"Failed to get observation: {current_observation}")
        
        print("Initial observation received.")
        
        for step in range(MAX_STEPS):
            print(f"\n--- Step {step + 2} ---")

            # --- Convert HTML observation to Markdown ---
            print("Converting HTML to Markdown...")
            markdown_content = convert_to_markdown(current_observation.__dict__)
            if not markdown_content:
                print("Markdown conversion failed. Stopping.")
                break
            print("Markdown content generated.")
            trajectory_observations.append(current_observation)

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
            inputs = tokenizer(prompt_text, return_tensors="pt")

            outputs = model.generate(**inputs, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id)
            response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            match = AGENT_PATTERN.search(response_text)
            if not match:
                print("LLM response did not contain a valid JSON block. Stopping.")
                print("Full response:", response_text)
                break

            json_text = match.group("json")
            # Strip whitespace and handle potential extra content
            json_text = json_text.strip()
            print(f"LLM generated action (JSON):\n{json_text}")
            
            predicted_action = agent_prompt.parse_action(f"```json\n{json_text}\n```")
            trajectory_actions.append(predicted_action)

            if not isinstance(predicted_action, BrowserAction):
                print("Failed to parse LLM response into a valid action. Stopping.")
                break

            # Update history with the observation and the action taken
            history.append((markdown_content, json_text))

            # --- Check for Stop Action ---
            # Use the matched_response from the parsed action, which is guaranteed to be valid JSON
            try:
                action_dict = json.loads(predicted_action.matched_response)
                action_key = action_dict.get("action_key")
            except (json.JSONDecodeError, AttributeError) as e:
                print(f"Warning: Could not extract action_key from parsed action: {e}")
                action_key = None
            
            if action_key and action_key in ["stop", "exit"]:
                print("Stop action received. Ending trajectory.")
                break

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
                print(f"Failed to get next observation: {current_observation}. Stopping.")
                break
            print("Received next observation.")

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
    trajectory_result = run_trajectory(first_row)
    
    if trajectory_result:
        print("\n--- Trajectory Finished ---")
        print(f"Instruction: {trajectory_result['task_instruction']}")
        print(f"Total Steps: {len(trajectory_result['trajectory_actions'])}")
        # You can add more detailed printing of the trajectory here if needed
    else:
        print("Trajectory generation failed.")