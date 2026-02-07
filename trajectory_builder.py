import sys
import json
import argparse
import io
import time
from datetime import datetime
import torch
import base64

from transformers import AutoTokenizer, AutoModelForCausalLM
# Assuming client, markdown, configs, utils are available in the project structure
from client import BrowserClient
from markdown import get_markdown_tree, render_markdown_tree
from configs.browser_config import BrowserObservation, NodeMetadata, BrowserConfig
from utils import safe_call, BrowserStatus

# --- 1. Replicate/Include HTML Conversion Functions (From your original code) ---

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
    # Simplified version, assuming NodeMetadata is importable
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

def convert_to_markdown(observation_data: dict) -> str:
    """
    Convert HTML observation to markdown format.
    
    Args:
        observation_data: Dictionary containing raw_html, metadata, etc.
        
    Returns:
        str: Markdown representation of the page
    """
    try:
        raw_html = observation_data.get('raw_html', '')
        metadata = observation_data.get('metadata', {})
        
        if not raw_html:
            return "No HTML content available"
        
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
            return "Failed to parse HTML into markdown structure"
        
        # Render markdown tree to text
        markdown_outputs = safe_call(
            render_markdown_tree,
            markdown_nodes,
            catch_errors=True,
            log_errors=False,
            max_errors=1
        )
        
        if markdown_outputs is BrowserStatus.ERROR:
            return "Failed to render markdown tree"
        
        markdown_text = " ".join(markdown_outputs)
        return markdown_text
        
    except Exception as e:
        return f"Error converting to markdown: {str(e)}"

# --- 2. Agent and Judge LLM Interfaces (Placeholders) ---

class AgentLLM:
    """
    The Policy: Insta-Qwen3-1.7B-SFT model.
    Handles prompt engineering and action prediction.
    """
    def __init__(self, model_name: str):
        print(f"Loading Agent Policy: {model_name}")
        # Initialize the SFT model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Assuming the model should run on CUDA if available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if self.device == 'cuda' else None
        ).to(self.device)
        self.model.eval() # Set to evaluation mode initially
        print("Agent Policy Loaded.")

    def predict_action(self, instruction: str, history: list, observation: str) -> dict:
        """
        Receives instruction, history, and markdown observation.
        Generates the next action a_t as a JSON object (function call).
        
        In a real web agent, this involves complex prompt engineering 
        (e.g., chat-template, JSON mode generation).
        """
        # --- PROMPT CONSTRUCTION (Crucial for Action Prediction) ---
        
        # A simplified system prompt for the web agent
        system_prompt = (
            "You are an expert web browsing agent. Your goal is to follow the instruction "
            "by predicting the next valid Playwright-style action in JSON format. "
            "The web page is provided as a simplified Markdown observation."
        )

        # Combine history, observation, and instruction into the model prompt
        prompt = (
            f"{system_prompt}\n\n"
            f"--- HISTORY ---\n{history}\n\n"
            f"--- WEB PAGE OBSERVATION (Markdown) ---\n{observation}\n\n"
            f"--- INSTRUCTION ---\n{instruction}\n\n"
            "PREDICT NEXT ACTION (JSON only):"
        )
        
        # --- GENERATION ---
        
        # Simple generation call - RL frameworks use more complex generation
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Placeholder for JSON-constrained generation (e.g., using a grammar or dedicated JSON decoder)
        # This is the point where we instruct the model to output a JSON Playwright action.
        output_tokens = self.model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            # Add JSON generation constraints here for robust output
        )
        
        # Decode and try to parse the action
        generated_text = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        # A simple placeholder to extract the JSON-like part
        try:
            # Look for the last JSON block the model generated
            json_start = generated_text.rfind('{')
            json_end = generated_text.rfind('}') + 1
            action_json_str = generated_text[json_start:json_end]
            
            # Action must conform to a schema like:
            # {"function_name": "click", "args": {"selector": "...", "text": "..."}}
            action = json.loads(action_json_str)
            print(f"✅ Predicted Action: {action}")
            return action
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"❌ Failed to parse valid JSON action: {e}. Defaulting to NOOP.")
            return {"function_name": "noop", "args": {}}

class JudgeLLM:
    """
    The Outcome-supervised Reward Model (ORM): Gemini 2.5 Flash.
    Calculates the final sparse binary reward (r_T).
    """
    def __init__(self, api_key: str):
        # NOTE: Real implementation requires importing the Gemini SDK
        # from google import genai
        # self.client = genai.Client(api_key=api_key)
        self.api_key = api_key
        print("Judge LLM (Gemini 2.5 Flash) Initialized.")

    def determine_reward(self, trajectory: list, instruction: str, criteria: list) -> int:
        """
        Evaluates the full trajectory against the task instruction and criteria.
        Returns r_T ∈ {0, 1}.
        """
        # In a real setup, this would construct a prompt with:
        # 1. The original instruction
        # 2. The success criteria
        # 3. The full list of (Action, State) pairs from the trajectory
        # 4. The final state's HTML/Screenshot
        
        # And then call the Gemini API to ask: "Did the agent satisfy ALL criteria? Respond with 1 or 0."
        
        print("\n--- JUDGE LLM EVALUATION (Placeholder) ---")
        print(f"Instruction: {instruction}")
        print(f"Criteria: {criteria}")
        # print(f"Trajectory Length: {len(trajectory)} steps")
        
        # --- API CALL SIMULATION ---
        time.sleep(1) # Simulate network delay
        
        # For a runnable example, we'll return a random reward.
        # In real RL, you would get a hard 1 or 0 based on the Judge's classification.
        reward = 1 if len(trajectory) > 3 and "search" in instruction.lower() else 0
        
        print(f"Judge Output: Final Reward r_T = {reward}")
        return reward

# --- 3. RL Trainer and Trajectory Collection Loop ---

def run_rl_trajectory(
    client: BrowserClient, 
    agent: AgentLLM, 
    judge: JudgeLLM, 
    task: dict, 
    max_steps: int = 15
) -> dict:
    """
    Executes a single trajectory for RL training (Steps 1-6).
    
    Args:
        client: The BrowserClient instance.
        agent: The AgentLLM instance (policy).
        judge: The JudgeLLM instance (reward model).
        task: A dictionary from the Insta Dataset (website, instruction, criteria).
        max_steps: Maximum number of actions allowed.
        
    Returns:
        dict: The collected trajectory data (states, actions, reward).
    """
    instruction = task['instruction']
    start_url = task['website']
    criteria = task['criteria']
    
    trajectory = []
    action_history = []
    
    print(f"\n--- STARTING RL TRAJECTORY ---")
    print(f"Task: {instruction}")
    
    try:
        # Step 1: Start Environment and Navigate
        client.start()
        client.goto(f"https://{start_url}")
        
        # Loop Check variables
        is_done = False
        step = 0
        
        while not is_done and step < max_steps:
            print(f"\n--- STEP {step+1} ---")
            
            # Step 1/4: Retrieve current state (s_t)
            observation_result = client.observation()
            if not isinstance(observation_result, BrowserObservation):
                print(f"❌ Failed to get observation: {observation_result}")
                break
            observation = observation_result
            
            # Step 2: Pre-processing (Convert HTML to Markdown)
            observation_dict = {
                'raw_html': observation.raw_html,
                'metadata': observation.metadata,
                'current_url': observation.current_url,
                'screenshot': observation.screenshot, # not used in convert_to_markdown, but stored
            }
            markdown_obs = convert_to_markdown(observation_dict)
            
            # Step 3: Agent LLM Predicts Action (a_t)
            action = agent.predict_action(instruction, action_history, markdown_obs)
            
            # Store the full step data
            trajectory.append({
                'step': step,
                'state_markdown': markdown_obs,
                'state_raw': observation_dict,
                'action': action,
                'current_url': observation.current_url
            })
            
            # Step 4: Execution (Playwright)
            action_history.append(action)
            
            # Execute the action (Implementation depends on BrowserClient's action API)
            if action['function_name'] == 'stop':
                is_done = True
                print("Agent predicted STOP action.")
                continue
                
            # Placeholder for client execution
            execution_result = client.execute_action(action) 
            if execution_result != BrowserStatus.SUCCESS:
                print(f"⚠️ Action execution failed: {execution_result}")
                # Decide if agent should continue or stop on failure
                # break
            
            step += 1
            
        # Step 6: Reward Determination (Evaluate trajectory T)
        final_reward = judge.determine_reward(trajectory, instruction, criteria)
        
        # Finalize the trajectory data
        full_trajectory = {
            'task': task,
            'trajectory': trajectory,
            'final_reward': final_reward,
            'success': final_reward == 1,
            'timestamp': datetime.now().isoformat()
        }
        
        return full_trajectory
        
    except Exception as e:
        print(f"❌ Trajectory run failed: {e}")
        return {'task': task, 'trajectory': [], 'final_reward': 0, 'success': False, 'error': str(e)}
        
    finally:
        client.close()


def placeholder_rl_update(agent: AgentLLM, trajectory_data: dict):
    """
    Step 7: Policy Update (Placeholder for M-GRPO or PPO/KL-constrained update).
    """
    print("\n--- STEP 7: RL POLICY UPDATE (Placeholder) ---")
    if trajectory_data.get('final_reward', 0) == 1:
        print("✅ Trajectory was successful. Applying positive gradient/KL constraint update.")
        # In a real setup, this would call a trainer object (e.g., from TRL or a custom GRPO implementation)
        # trainer.step(trajectory_data)
        # The update uses the reward, the policy's log-probabilities for the actions taken,
        # and a baseline/value estimate (if using an actor-critic method).
    else:
        print("❌ Trajectory failed. Applying negative gradient/KL constraint update or sampling for ORM.")
        
    # Example of a simple dummy update counter
    agent.model.rl_update_count = getattr(agent.model, 'rl_update_count', 0) + 1
    print(f"Total RL Updates: {agent.model.rl_update_count}")


# --- 4. Main RL Training Loop ---

if __name__ == "__main__":
    
    # 1. Configuration and Initialization
    MODEL_NAME = "btrabucco/Insta-Qwen3-1.7B-SFT"
    BROWSER_SERVER_URL = "http://localhost:3000"
    JUDGE_API_KEY = "YOUR_GEMINI_API_KEY" # Placeholder
    
    # Dummy Insta Dataset (in reality, you'd load this from a file)
    INSTA_DATASET = [
        {
            'website': 'railway.gov.tw', 
            'instruction': 'Find the earliest train from Taipei to Kaohsiung on 2025.05.20 and state its departure time and fare for a standard adult ticket.', 
            'steps': ['...'], # Not used in inference/RL loop, but useful for debugging
            'criteria': ['Identifies the earliest train listed for the specified route and date.', 'Extracts and states the departure time of the earliest train.', 'Extracts and states the standard adult fare for the earliest train.']
        },
        {
            'website': 'gameandfishmag.com',
            'instruction': 'Find an article on the website that discusses spring bass fishing in Wisconsin and identify at least two specific lakes or rivers mentioned in that article.',
            'steps': ['...'],
            'criteria': ['Identifies an article specifically about spring bass fishing in Wisconsin.', 'Lists at least two distinct lakes or rivers from the article that are recommended for spring bass fishing.']
        }
    ]
    
    # Initialize components
    browser_config = BrowserConfig(playwright_url=BROWSER_SERVER_URL, screen_width=1920, screen_height=1080)
    client = BrowserClient(browser_config)
    agent_llm = AgentLLM(MODEL_NAME)
    judge_llm = JudgeLLM(JUDGE_API_KEY)
    
    print("\n--- STARTING RL TRAINING EPOCH ---")
    
    # 2. RL Training Epoch (Iterate through tasks)
    TRAINING_ITERATIONS = 5 # Number of tasks to run per epoch
    
    for i, task in enumerate(INSTA_DATASET[:TRAINING_ITERATIONS]):
        
        # Run Trajectory (Steps 1-6)
        trajectory_data = run_rl_trajectory(client, agent_llm, judge_llm, task)
        
        # Policy Update (Step 7)
        placeholder_rl_update(agent_llm, trajectory_data)
        
        print(f"\n--- ITERATION {i+1} COMPLETE ---")
        time.sleep(2) # Pause between tasks
        
    print("\n--- RL TRAINING EPOCH FINISHED ---")