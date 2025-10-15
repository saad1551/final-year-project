import sys
import json
import argparse
import base64
import io
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from client import BrowserClient
from insta.markdown.build import get_markdown_tree
from insta.markdown.render import render_markdown_tree
from insta.agent_prompts.base_agent_prompt import BaseAgentPrompt, AGENT_PATTERN
from configs.browser_config import BrowserObservation, NodeMetadata, BrowserConfig
from configs.agent_config import BrowserAction
from utils import safe_call, BrowserStatus

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
        
        if not raw_html:
            return "No HTML content found."
        
        # Convert metadata dict to NodeMetadata objects if needed
        node_metadata = convert_metadata_dict_to_objects(metadata)
        
        # Get markdown tree
        markdown_nodes = safe_call(
            get_markdown_tree,
            raw_html,
            node_metadata,
            catch_errors=True,
            log_errors=False,
            max_errors=1
        )
        
        if markdown_nodes is BrowserStatus.ERROR:
            return "Failed to get markdown tree."

        # Render markdown tree
        markdown_text = safe_call(
            render_markdown_tree,
            markdown_nodes,
            catch_errors=True,
            log_errors=False,
            max_errors=1
        )

        if markdown_text is BrowserStatus.ERROR:
            return "Failed to render markdown tree."
            
        return markdown_text
        
    except Exception as e:
        print(f"Error in markdown conversion: {e}")
        return None


# --- Configuration ---
MODEL_NAME = "btrabucco/Insta-Qwen3-1.7B-SFT"
BROWSER_SERVER_URL = "http://localhost:3000"


# --- Data Simulation: First Row of Insta Dataset ---
# Simulating: df.iloc[0]

df = pd.read_csv("data/insta-150k-test.csv")

first_row = df.iloc[0].to_dict()


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
    
    start_url = task_data['website']
    
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
        
        # Get observation
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


if __name__ == "__main__":
    result = setup_and_convert_initial_state(first_row)
    
    if result:
        print("\n--- Pipeline Finished ---")
        print(f"Instruction: {result['task_instruction']}")
        # print(f"Initial Markdown:\n{result['initial_markdown']}")
        if result['predicted_action']:
            print(f"Predicted Action object: {result['predicted_action']}")
        else:
            print("No action was predicted.")