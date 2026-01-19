import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional


LLM_API_KEY = "YOUR_API_KEY_HERE"
LLM_BASE_URL = "YOUR_LLM_URL_HERE"
LLM_MODEL = "YOUR_MODEL_NAME_HERE"

DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.5
DEFAULT_TOP_P = 1.0


@dataclass
class JudgeScores:
    success: float
    efficiency: float
    self_correction: float
    raw_response: Optional[str] = None
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "success": self.success,
            "efficiency": self.efficiency,
            "self_correction": self.self_correction
        }


@dataclass
class JudgeLLMConfig:
    api_key: str = LLM_API_KEY
    base_url: str = LLM_BASE_URL
    model: str = LLM_MODEL
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    
    def to_client_kwargs(self) -> Dict:
        return {
            "api_key": self.api_key,
            "base_url": self.base_url
        }
    
    def to_generation_kwargs(self) -> Dict:
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p
        }


JUDGE_SYSTEM_PROMPT = """You are helping me evaluate a language model agent that interacts with and navigates live webpages. I will share a task provided to the agent, and a sequence of webpages and actions produced by the agent.

## The Action Format

The agent produces actions as JSON in a fenced code block:

```json
{
    "action_key": str,
    "action_kwargs": dict,
    "target_element_id": int
}
```

Actions have the following components:

- `action_key`: The name of the selected action.
- `action_kwargs`: A dictionary of arguments for the action.
- `target_element_id`: An optional id for the element to call the action on.

## Evaluation Instructions

Based on the agent's trajectory, you are helping me determine if the agent's task has been completed successfully. 

You will provide scores as JSON in a fenced code block:

```json
{
    "success": float,
    "efficiency": float,
    "self_correction": float
}
```

### Score Definitions

- `success`: Your confidence the agent's task has been completed successfully.
    - range: 0.0 (not possible) to 1.0 (absolutely certain).

- `efficiency`: Your confidence the agent has taken the most efficient path to complete the task.
    - range: 0.0 (not possible) to 1.0 (absolutely certain).

- `self_correction`: Your confidence the agent has demonstrated self-corrective behaviors during its completion of the task. These behaviors include backtracking to a more promising state, replanning when new information is discovered, and recognizing its own mistakes.
    - range: 0.0 (not possible) to 1.0 (absolutely certain).

Write a 300 word analysis that establishes rigorous success criteria for the task, and determines which criteria the agent has satisfied. After your response, provide your scores as JSON in a fenced code block."""


JUDGE_USER_PROMPT_TEMPLATE = """## Evaluate The Following Task

{instruction}

Here is the agent's trajectory:

{trajectory_summary}

## Evaluation Instructions

Based on the agent's trajectory, you are helping me determine if the agent's task has been completed successfully. 

You will provide scores as JSON in a fenced code block:

```json
{{
    "success": float,
    "efficiency": float,
    "self_correction": float
}}
```

### Score Definitions

- `success`: Your confidence the agent's task has been completed successfully.
    - range: 0.0 (not possible) to 1.0 (absolutely certain).

- `efficiency`: Your confidence the agent has taken the most efficient path to complete the task.
    - range: 0.0 (not possible) to 1.0 (absolutely certain).

- `self_correction`: Your confidence the agent has demonstrated self-corrective behaviors during its completion of the task. These behaviors include backtracking to a more promising state, replanning when new information is discovered, and recognizing its own mistakes.
    - range: 0.0 (not possible) to 1.0 (absolutely certain).

Write a 300 word analysis that establishes rigorous success criteria for the task, and determines which criteria the agent has satisfied. After your response, provide your scores as JSON in a fenced code block."""


JUDGE_JSON_PATTERN = re.compile(
    r"```json\n(?P<json>.*?)\n```",
    re.DOTALL
)


def configure_judge_llm(
    api_key: str,
    base_url: str,
    model: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P
) -> JudgeLLMConfig:
    return JudgeLLMConfig(
        api_key=api_key,
        base_url=base_url,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p
    )


def create_openai_client(config: JudgeLLMConfig):
    import openai
    return openai.OpenAI(**config.to_client_kwargs())


def format_trajectory_for_judge(
    observations: List[str],
    actions: List[str],
    last_observations: int = 5,
    last_actions: int = 5
) -> str:
    outputs = []
    
    for step, (observation, action) in enumerate(zip(observations, actions)):
        time_left = len(actions) - step - 1
        
        if time_left < last_observations:
            step_label = "Previous" if time_left > 0 else "Last"
            outputs.append(f"## {step_label} Webpage:\n\n{observation}")
        
        if time_left < last_actions:
            step_label = "Previous" if time_left > 0 else "Last"
            outputs.append(f"## {step_label} Action:\n\n{action}")
    
    return "\n\n".join(outputs)


def build_judge_prompt(
    instruction: str,
    trajectory_summary: str
) -> List[Dict[str, str]]:
    user_prompt = JUDGE_USER_PROMPT_TEMPLATE.format(
        instruction=instruction,
        trajectory_summary=trajectory_summary
    )
    
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]


def parse_judge_response(response: str) -> JudgeScores:
    match = JUDGE_JSON_PATTERN.search(response)
    
    if match is None or "json" not in match.groupdict():
        raise ValueError("Failed to extract JSON from judge response")
    
    matched_json = match.group("json")
    
    try:
        response_dict = json.loads(matched_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON from judge response: {e}")
    
    required_keys = ["success", "efficiency", "self_correction"]
    for key in required_keys:
        if key not in response_dict:
            raise ValueError(f"Missing required key '{key}' in judge response")
    
    success = response_dict["success"]
    efficiency = response_dict["efficiency"]
    self_correction = response_dict["self_correction"]
    
    for name, value in [("success", success), ("efficiency", efficiency), ("self_correction", self_correction)]:
        if not isinstance(value, (int, float)):
            raise ValueError(f"Score '{name}' must be a number, got {type(value).__name__}")
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"Score '{name}' must be between 0.0 and 1.0, got {value}")
    
    return JudgeScores(
        success=float(success),
        efficiency=float(efficiency),
        self_correction=float(self_correction),
        raw_response=response
    )


def query_judge_llm(
    client,
    config: JudgeLLMConfig,
    messages: List[Dict[str, str]]
) -> str:
    response = client.chat.completions.create(
        messages=messages,
        **config.to_generation_kwargs()
    )
    return response.choices[0].message.content


def evaluate_agent_trajectory(
    config: JudgeLLMConfig,
    instruction: str,
    observations: List[str],
    actions: List[str],
    last_observations: int = 5,
    last_actions: int = 5
) -> JudgeScores:
    client = create_openai_client(config)
    
    trajectory_summary = format_trajectory_for_judge(
        observations=observations,
        actions=actions,
        last_observations=last_observations,
        last_actions=last_actions
    )
    
    messages = build_judge_prompt(
        instruction=instruction,
        trajectory_summary=trajectory_summary
    )
    
    raw_response = query_judge_llm(
        client=client,
        config=config,
        messages=messages
    )
    
    scores = parse_judge_response(raw_response)
    return scores


def evaluate_with_final_response(
    config: JudgeLLMConfig,
    instruction: str,
    final_response: str
) -> JudgeScores:
    return evaluate_agent_trajectory(
        config=config,
        instruction=instruction,
        observations=["Agent's Final State"],
        actions=[final_response],
        last_observations=1,
        last_actions=1
    )


def get_default_config() -> JudgeLLMConfig:
    return JudgeLLMConfig()


def scores_to_dict(scores: JudgeScores) -> Dict[str, float]:
    return scores.to_dict()


def validate_scores(scores_dict: Dict[str, float]) -> bool:
    required_keys = ["success", "efficiency", "self_correction"]
    
    for key in required_keys:
        if key not in scores_dict:
            return False
        value = scores_dict[key]
        if not isinstance(value, (int, float)):
            return False
        if not 0.0 <= float(value) <= 1.0:
            return False
    
    return True
