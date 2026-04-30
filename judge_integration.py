import os
from insta.judge import BrowserJudge
from insta.configs.judge_config import JudgeConfig, BrowserJudgment
from typing import List


# Gemini API Configuration.
# Read from JUDGE_API_KEY env var; the literal below is a development fallback
# only and should be overridden in production / on the training VM.
JUDGE_API_KEY = os.environ.get(
    "JUDGE_API_KEY", "AIzaSyCpMDhbwWz12rsPGErG2v3oqTQjNg3N8C8"
)
JUDGE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
JUDGE_MODEL = "gemini-2.5-flash"

JUDGE_MAX_TOKENS = 4096
JUDGE_TEMPERATURE = 0.5
JUDGE_TOP_P = 1.0

JUDGE_LAST_ACTIONS = 50
JUDGE_LAST_OBS = 50


def get_judge_config() -> JudgeConfig:
    return JudgeConfig(
        client_type="openai",
        client_kwargs={
            "api_key": JUDGE_API_KEY,
            "base_url": JUDGE_BASE_URL,
        },
        generation_kwargs={
            "model": JUDGE_MODEL,
            "max_tokens": JUDGE_MAX_TOKENS,
            "temperature": JUDGE_TEMPERATURE,
            "top_p": JUDGE_TOP_P,
        },
        last_actions=JUDGE_LAST_ACTIONS,
        last_obs=JUDGE_LAST_OBS,
    )


_judge_instance = None


def get_judge() -> BrowserJudge:
    global _judge_instance
    if _judge_instance is None:
        config = get_judge_config()
        _judge_instance = BrowserJudge(config)
    return _judge_instance


def judge_trajectory(
    instruction: str,
    observations: List[str],
    actions: List[str],
    criteria: str = "",
    steps: str = ""
) -> BrowserJudgment:
    judge = get_judge()
    judgment = judge(
        observations=observations,
        actions=actions,
        instruction=instruction,
        criteria=criteria,
        steps=steps
    )
    return judgment


def print_judgment(judgment: BrowserJudgment) -> None:
    print(f"  Success: {judgment.success}")
    print(f"  Efficiency: {judgment.efficiency}")
    print(f"  Self-correction: {judgment.self_correction}")
