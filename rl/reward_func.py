from insta import (
    ACTION_PARSERS,
    BaseActionParser,
    BrowserStatus,
)

from insta.utils import safe_call

import json


SIMPLIFIED_JSON_PARSER: BaseActionParser = (
    ACTION_PARSERS["simplified_json"]()
)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info = None
) -> float:
    """Reward function for training LLM agents to operate a browser,
    and complete a desired web navigation task.

    Arguments:

    model_output: str
        The model output to evaluate.

    ground_truth: str
        The ground truth output to compare against.

    Returns:

    reward: float
        The reward value for GRPO update.
    
    """
    
    ground_truth = json.loads(
        ground_truth
    )

    action = safe_call(
        SIMPLIFIED_JSON_PARSER.parse_action,
        response = solution_str,
        catch_errors = True,
        max_errors = 1,
        log_errors = False
    )

    if action is BrowserStatus.ERROR:

        return 0.0

    action = json.loads(
        action.matched_response
    )

    has_required_keys = (
        "action_key" in action and
        "target_element_id" in action and
        "action_kwargs" in action
    )

    if not has_required_keys:

        return 0.0

    reward = 0.1

    action_key_match = (
        ground_truth["action_key"] == 
        action["action_key"]
    )

    if action_key_match:

        reward += 0.3

    target_element_match = (
        ground_truth["target_element_id"] == 
        action["target_element_id"]
    )

    if target_element_match:

        reward += 0.3

    ground_truth_kwargs_str = json.dumps(
        ground_truth["action_kwargs"]
    )

    action_kwargs_str = json.dumps(
        action["action_kwargs"]
    )

    action_kwargs_match = (
        ground_truth_kwargs_str ==
        action_kwargs_str
    )

    if action_kwargs_match:

        reward += 0.3

    return reward


def compute_judgment_reward(
    judgment_dict: dict,
    success_weight: float = 0.5,
    efficiency_weight: float = 0.3,
    self_correction_weight: float = 0.2
) -> float:
    """Compute reward from judge LLM scores.
    
    This function computes a weighted reward from the three judgment scores:
    - success: Whether the task was completed successfully (0-1)
    - efficiency: How efficiently the task was completed (0-1)  
    - self_correction: Ability to recover from mistakes (0-1)

    Arguments:

    judgment_dict: dict
        Dictionary containing 'success', 'efficiency', and 'self_correction' keys.
        
    success_weight: float
        Weight for success score (default: 0.5)
        
    efficiency_weight: float
        Weight for efficiency score (default: 0.3)
        
    self_correction_weight: float
        Weight for self-correction score (default: 0.2)

    Returns:

    reward: float
        Combined reward value between 0 and 1.
    
    """
    if judgment_dict is None:
        return 0.0
    
    success = judgment_dict.get('success', 0.0) or 0.0
    efficiency = judgment_dict.get('efficiency', 0.0) or 0.0
    self_correction = judgment_dict.get('self_correction', 0.0) or 0.0
    
    reward = (
        success_weight * success +
        efficiency_weight * efficiency +
        self_correction_weight * self_correction
    )
    
    return reward