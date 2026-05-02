"""
Gemini judge integration.

Two paths:
  - Vertex AI (preferred when JUDGE_VERTEX_PROJECT is set or JUDGE_USE_VERTEX=1):
    Uses google-genai with vertexai=True, drawing on the GenAI/Vertex credit
    pool and the VM's attached service-account credentials. No API key needed
    when the VM has cloud-platform scope.
  - AI Studio (fallback): uses google-genai with a Gemini API key.

Public API mirrors what the rest of the codebase imports:
  judge_trajectory(instruction, observations, actions, criteria, steps)
      -> BrowserJudgment
  print_judgment(judgment) -> None

Implementation notes:
  - We import the InSTA judgment dataclass and the verbose judge prompt
    directly. We do NOT import insta.judge — it pulls in vllm, which we
    don't need on the training VM.
  - Trajectory summary formatting matches insta.judge.BrowserJudge.get_user_prompt
    (last-N actions/observations, with "Previous"/"Last"/"Next" labels).
"""

from __future__ import annotations

import os
import re
import time
from typing import List, Optional

from google import genai
from google.genai.types import GenerateContentConfig, HttpOptions

from insta.configs.judge_config import BrowserJudgment
from insta.judge_prompts import JUDGE_PROMPTS


# ── Config ─────────────────────────────────────────────────────────────────────
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gemini-2.5-flash")
JUDGE_TEMPERATURE = float(os.environ.get("JUDGE_TEMPERATURE", "0.5"))
JUDGE_MAX_TOKENS = int(os.environ.get("JUDGE_MAX_TOKENS", "4096"))
JUDGE_TIMEOUT = int(os.environ.get("JUDGE_TIMEOUT", "120"))  # seconds
JUDGE_PROMPT_NAME = os.environ.get("JUDGE_PROMPT", "verbose")  # 'base' or 'verbose'

JUDGE_LAST_ACTIONS = int(os.environ.get("JUDGE_LAST_ACTIONS", "50"))
JUDGE_LAST_OBS = int(os.environ.get("JUDGE_LAST_OBS", "50"))
# Per-observation cap (characters, not tokens — Gemini handles large contexts
# but we still want to keep individual observations bounded for prompt clarity)
JUDGE_MAX_OBS_CHARS = int(os.environ.get("JUDGE_MAX_OBS_CHARS", "8000"))

JUDGE_RETRIES = int(os.environ.get("JUDGE_RETRIES", "3"))

# Vertex auth: set JUDGE_VERTEX_PROJECT to your GCP project (or JUDGE_USE_VERTEX=1
# to auto-detect from gcloud). Location defaults to us-central1.
JUDGE_VERTEX_PROJECT = os.environ.get("JUDGE_VERTEX_PROJECT", "")
JUDGE_VERTEX_LOCATION = os.environ.get("JUDGE_VERTEX_LOCATION", "us-central1")
JUDGE_USE_VERTEX = os.environ.get("JUDGE_USE_VERTEX", "").lower() in ("1", "true", "yes")

# AI Studio fallback path. Read from env only — no hardcoded fallback.
# Used only when Vertex auth (JUDGE_USE_VERTEX / JUDGE_VERTEX_PROJECT) is
# not configured. Get a key at https://aistudio.google.com/apikey.
JUDGE_API_KEY = os.environ.get("JUDGE_API_KEY")


_NULL_JUDGMENT = BrowserJudgment(
    success=None,
    efficiency=None,
    self_correction=None,
    response=None,
    matched_response=None,
)


# ── Client (lazy, cached) ──────────────────────────────────────────────────────
_gemini_client: Optional[genai.Client] = None
_judge_prompt = None


def _resolve_vertex_project() -> Optional[str]:
    """Resolve the Vertex project from env or gcloud config."""
    if JUDGE_VERTEX_PROJECT:
        return JUDGE_VERTEX_PROJECT
    if JUDGE_USE_VERTEX:
        # Try gcloud
        try:
            import subprocess
            out = subprocess.check_output(
                ["gcloud", "config", "get-value", "project"],
                stderr=subprocess.DEVNULL, timeout=5
            ).decode().strip()
            if out and out not in ("(unset)", ""):
                return out
        except Exception:
            pass
        # Try metadata server (works on any GCE VM)
        try:
            import urllib.request
            req = urllib.request.Request(
                "http://metadata.google.internal/computeMetadata/v1/project/project-id",
                headers={"Metadata-Flavor": "Google"},
            )
            return urllib.request.urlopen(req, timeout=2).read().decode().strip()
        except Exception:
            pass
    return None


def _get_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        project = _resolve_vertex_project()
        if project:
            print(f"[judge] Using Vertex AI  project={project}  location={JUDGE_VERTEX_LOCATION}  model={JUDGE_MODEL}")
            _gemini_client = genai.Client(
                vertexai=True,
                project=project,
                location=JUDGE_VERTEX_LOCATION,
            )
        else:
            print(f"[judge] Using AI Studio API key  model={JUDGE_MODEL}")
            _gemini_client = genai.Client(api_key=JUDGE_API_KEY)
    return _gemini_client


def _get_judge_prompt():
    global _judge_prompt
    if _judge_prompt is None:
        _judge_prompt = JUDGE_PROMPTS[JUDGE_PROMPT_NAME]()
    return _judge_prompt


# ── Prompt construction (mirrors insta.judge.BrowserJudge.get_user_prompt) ─────
def _truncate_obs(obs: str) -> str:
    """Soft char-cap on a single observation. Keeps head + tail with marker."""
    if len(obs) <= JUDGE_MAX_OBS_CHARS:
        return obs
    head = JUDGE_MAX_OBS_CHARS // 2
    tail = JUDGE_MAX_OBS_CHARS - head
    return obs[:head] + f"\n\n... [{len(obs) - JUDGE_MAX_OBS_CHARS} chars truncated] ...\n\n" + obs[-tail:]


def _format_trajectory_summary(observations: List[str], actions: List[str]) -> str:
    """Concatenate the last-N observations and actions, matching insta's labels."""
    parts = []
    n = len(actions)
    for step, (obs, act) in enumerate(zip(observations, actions)):
        time_left = n - step - 1
        if time_left < JUDGE_LAST_OBS:
            label = "Previous" if time_left > 0 else "Last"
            parts.append(f"## {label} Webpage:\n\n{_truncate_obs(obs or '')}")
        if time_left < JUDGE_LAST_ACTIONS:
            label = "Previous" if time_left > 0 else "Next"
            parts.append(f"## {label} Action:\n\n{act or ''}")
    return "\n\n".join(parts)


def _build_user_prompt(
    instruction: str,
    observations: List[str],
    actions: List[str],
    criteria: str,
    steps: str,
) -> str:
    summary = _format_trajectory_summary(observations, actions)
    parts = [f"Determine if the agent has completed the task:\n\n{instruction}"]
    if steps:
        parts.append(f"\n\nExpected steps:\n{steps}")
    if criteria:
        parts.append(f"\n\nSuccess criteria:\n{criteria}")
    parts.append(f"\n\nHere is the agent's trajectory:\n\n{summary}")
    return "".join(parts)


# ── Main entry point ───────────────────────────────────────────────────────────
def judge_trajectory(
    instruction: str,
    observations: List[str],
    actions: List[str],
    criteria: str = "",
    steps: str = "",
) -> BrowserJudgment:
    """Send the trajectory to the Gemini judge and return a BrowserJudgment.

    Uses Vertex AI when JUDGE_VERTEX_PROJECT (or JUDGE_USE_VERTEX=1) is set,
    falling back to the AI Studio API key path otherwise.
    """
    client = _get_client()
    judge_prompt = _get_judge_prompt()

    user_prompt = _build_user_prompt(instruction, observations, actions, criteria, steps)
    system_instruction = judge_prompt.system_prompt

    config = GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=JUDGE_TEMPERATURE,
        max_output_tokens=JUDGE_MAX_TOKENS,
        http_options=HttpOptions(timeout=JUDGE_TIMEOUT * 1000),
    )

    last_err: Optional[Exception] = None
    for attempt in range(JUDGE_RETRIES):
        try:
            response = client.models.generate_content(
                model=JUDGE_MODEL,
                contents=user_prompt,
                config=config,
            )
            raw = (response.text or "").strip()
            print(f"\n[JUDGE DEBUG] Raw LLM Response:\n{raw}\n")
            return judge_prompt.parse_judgment(response=raw)
        except Exception as e:
            last_err = e
            msg = str(e)
            is_rate_limit = "429" in msg or "RESOURCE_EXHAUSTED" in msg
            if attempt < JUDGE_RETRIES - 1:
                if is_rate_limit:
                    delay = _parse_retry_delay(msg) or 60.0
                    delay += 5.0
                    print(f"[judge] Rate limited; sleeping {delay:.0f}s before retry "
                          f"(attempt {attempt + 1}/{JUDGE_RETRIES})")
                    time.sleep(delay)
                else:
                    backoff = 2 ** attempt
                    print(f"[judge] Error '{e}'; retrying in {backoff}s "
                          f"(attempt {attempt + 1}/{JUDGE_RETRIES})")
                    time.sleep(backoff)
            else:
                print(f"[judge] All retries exhausted; returning NULL_JUDGMENT. Last error: {e}")
    return _NULL_JUDGMENT


def _parse_retry_delay(msg: str) -> Optional[float]:
    """Pull a 'retry in Ns' hint out of a Gemini error string, if present."""
    m = re.search(r"retry[^0-9]*([0-9]+(?:\.[0-9]+)?)\s*(?:s|sec|second)", msg, re.IGNORECASE)
    return float(m.group(1)) if m else None


def print_judgment(judgment: BrowserJudgment) -> None:
    print(f"  Success         : {judgment.success}")
    print(f"  Efficiency      : {judgment.efficiency}")
    print(f"  Self-correction : {judgment.self_correction}")


# ── Compatibility shim ─────────────────────────────────────────────────────────
# Some callers may import these legacy names. Keep them defined so imports don't
# break, but they no longer do anything (we don't construct an OpenAI client).
def get_judge_config():
    return None


def get_judge():
    return None
