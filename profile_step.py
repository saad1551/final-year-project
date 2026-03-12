"""
profile_step.py — CPU-phase step timing profiler

Measures wall-clock time for the three CPU-bound phases in each trajectory step:
  1. Observation  — HTTP POST to Playwright server + base64 decode + PIL construction
                    (includes the configured 0.5 s observation sleep in client.py)
  2. Markdown     — lxml HTML clean/parse → recursive DOM walk → markdown render
  3. Action exec  — HTTP POST to Playwright server to execute the browser action

Model inference is intentionally excluded (GPU-dependent).
A fixed dummy action (page.mouse.wheel scroll) is used every step so no model is
required.  Functions that would otherwise be imported from pipeline_in_steps are
copied here directly to avoid pulling in the heavy torch/transformers/rl imports
that live at module level in that file.

Usage:
    python profile_step.py [--url URL] [--steps N] [--no-sleep]

    --url       Starting URL for the browser session  (default: https://www.google.com)
    --steps     Number of steps to profile            (default: 10)
    --no-sleep  Zero out the configured observation sleep so you measure pure
                HTTP + processing latency instead of wall-clock cost
"""

import argparse
import time
import statistics
import sys

from client import BrowserClient
from configs.browser_config import (
    BrowserConfig,
    BrowserObservation,
    FunctionCall,
    NodeMetadata,
    DEFAULT_BROWSER_CONFIG,
)
from markdown import get_markdown_tree, render_markdown_tree
from utils import safe_call, BrowserStatus

# ── constants (mirrored from pipeline_in_steps.py) ───────────────────────────
PAGE_LOAD_WAIT_SECONDS = 5
OBSERVATION_RETRY_ATTEMPTS = 5
OBSERVATION_RETRY_DELAY_SECONDS = 3


# ── helpers (inlined from pipeline_in_steps.py to avoid heavy torch imports) ─


def convert_metadata_to_node_objects(metadata_dict):
    if not metadata_dict:
        return {}
    return {
        node_id: NodeMetadata(**meta) if isinstance(meta, dict) else meta
        for node_id, meta in metadata_dict.items()
    }


def convert_html_to_markdown(observation_data):
    try:
        raw_html = observation_data.get("raw_html", "")
        metadata = observation_data.get("metadata", {})

        if not raw_html:
            print("Markdown conversion failed: no HTML content.")
            return None

        node_metadata = convert_metadata_to_node_objects(metadata)

        markdown_nodes = safe_call(
            get_markdown_tree,
            raw_html,
            node_metadata,
            catch_errors=True,
            log_errors=True,
            max_errors=1,
        )

        if markdown_nodes is BrowserStatus.ERROR:
            print("Markdown conversion failed: could not get markdown tree.")
            return None

        markdown_text = safe_call(
            render_markdown_tree,
            markdown_nodes,
            catch_errors=True,
            log_errors=True,
            max_errors=1,
        )

        if markdown_text is BrowserStatus.ERROR:
            print("Markdown conversion failed: could not render markdown tree.")
            return None

        result = " ".join(markdown_text)
        if not result.strip():
            print("Markdown conversion failed: rendered content is empty.")
            return None

        return result

    except Exception as e:
        print(f"Error in markdown conversion: {e}")
        return None


def get_observation_with_retry(client):
    """Get observation with retry logic (mirrored from pipeline_in_steps.py)."""
    observation = client.observation()
    if isinstance(observation, BrowserObservation):
        return observation

    for attempt in range(OBSERVATION_RETRY_ATTEMPTS):
        print(f"  Observation attempt {attempt + 1}/{OBSERVATION_RETRY_ATTEMPTS}...")
        time.sleep(OBSERVATION_RETRY_DELAY_SECONDS)
        observation = client.observation()
        if isinstance(observation, BrowserObservation):
            return observation
        print(f"  Observation failed, status: {observation}")

    raise Exception(
        f"Failed to get observation after {OBSERVATION_RETRY_ATTEMPTS} attempts"
    )


# ── constants ─────────────────────────────────────────────────────────────────
DEFAULT_START_URL = "https://www.google.com"
DEFAULT_N_STEPS = 10

# Dummy action: scroll down 300 px — exercises the full client.action() /
# Playwright server round-trip without requiring navigation or a model.
# Uses the same dotpath + args format as base_agent_prompt.py (positional string).
DUMMY_ACTION = [
    FunctionCall(
        dotpath="page.mouse.wheel",
        args="0,300",
    )
]

SEPARATOR = "─" * 60


# ── helpers ───────────────────────────────────────────────────────────────────


def build_browser_config(playwright_url: str, zero_obs_sleep: bool) -> BrowserConfig:
    delays = dict(DEFAULT_BROWSER_CONFIG.delays or {})
    if zero_obs_sleep:
        delays["observation"] = 0
    return BrowserConfig(
        playwright_url=playwright_url,
        screen_width=DEFAULT_BROWSER_CONFIG.screen_width,
        screen_height=DEFAULT_BROWSER_CONFIG.screen_height,
        restrict_viewport=DEFAULT_BROWSER_CONFIG.restrict_viewport,
        require_visible=DEFAULT_BROWSER_CONFIG.require_visible,
        require_frontmost=DEFAULT_BROWSER_CONFIG.require_frontmost,
        catch_errors=DEFAULT_BROWSER_CONFIG.catch_errors,
        log_errors=DEFAULT_BROWSER_CONFIG.log_errors,
        max_errors=DEFAULT_BROWSER_CONFIG.max_errors,
        delays=delays,
    )


def initialize_session(start_url: str, browser_config: BrowserConfig) -> BrowserClient:
    client = BrowserClient(browser_config)

    status = client.start()
    if status == BrowserStatus.ERROR:
        print("[ERROR] Failed to start browser session.")
        sys.exit(1)
    print(f"Browser session started  (id: {client.session_id})")

    status = client.goto(start_url)
    if status == BrowserStatus.ERROR:
        print(f"[ERROR] Failed to navigate to {start_url}.")
        sys.exit(1)
    print(f"Navigated to {start_url}")

    print(f"Waiting {PAGE_LOAD_WAIT_SECONDS}s for page to load...")
    time.sleep(PAGE_LOAD_WAIT_SECONDS)
    return client


def print_step_timing(step: int, t_obs: float, t_md: float, t_exec: float) -> None:
    total = t_obs + t_md + t_exec
    print(
        f"[TIMING] Step {step:>2d}  "
        f"obs={t_obs:.3f}s  "
        f"md={t_md:.3f}s  "
        f"exec={t_exec:.3f}s  "
        f"total={total:.3f}s"
    )


def print_summary(timings: dict, obs_sleep: float) -> None:
    n = len(timings["obs"])
    if n == 0:
        print("No steps completed — nothing to summarise.")
        return

    phases = [
        ("obs", "Observation  ", f"(incl. {obs_sleep:.1f}s configured sleep)"),
        ("md", "Markdown     ", ""),
        ("exec", "Action exec  ", ""),
    ]

    combined = [
        timings["obs"][i] + timings["md"][i] + timings["exec"][i] for i in range(n)
    ]

    print(f"\n{SEPARATOR}")
    print(f"  PROFILING SUMMARY  ({n} step{'s' if n != 1 else ''})")
    print(SEPARATOR)
    print(f"  {'Phase':<16}  {'mean':>7}  {'min':>7}  {'max':>7}  {'total':>8}  note")
    print(f"  {'-' * 14}  {'-' * 7}  {'-' * 7}  {'-' * 7}  {'-' * 8}  ----")

    for key, label, note in phases:
        vals = timings[key]
        print(
            f"  {label}  "
            f"{statistics.mean(vals):>6.3f}s  "
            f"{min(vals):>6.3f}s  "
            f"{max(vals):>6.3f}s  "
            f"{sum(vals):>7.3f}s  "
            f"{note}"
        )

    print(f"  {'-' * 14}  {'-' * 7}  {'-' * 7}  {'-' * 7}  {'-' * 8}")
    print(
        f"  {'CPU total':<16}  "
        f"{statistics.mean(combined):>6.3f}s  "
        f"{min(combined):>6.3f}s  "
        f"{max(combined):>6.3f}s  "
        f"{sum(combined):>7.3f}s  "
        f"(excl. inference)"
    )
    print(SEPARATOR)


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile CPU-bound step phases (observation, markdown, action exec)."
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_START_URL,
        help=f"Starting URL for the browser session (default: {DEFAULT_START_URL})",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_N_STEPS,
        help=f"Number of steps to profile (default: {DEFAULT_N_STEPS})",
    )
    parser.add_argument(
        "--no-sleep",
        action="store_true",
        help="Zero out the configured observation sleep to measure pure HTTP + processing latency",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_BROWSER_CONFIG.playwright_port,
        help=f"Port the Playwright server is listening on (default: {DEFAULT_BROWSER_CONFIG.playwright_port})",
    )
    args = parser.parse_args()

    obs_sleep = (
        0.0
        if args.no_sleep
        else (DEFAULT_BROWSER_CONFIG.delays or {}).get("observation", 0.0)
    )

    print(SEPARATOR)
    print("  Step CPU Profiler")
    print(SEPARATOR)
    print(f"  URL        : {args.url}")
    print(f"  Steps      : {args.steps}")
    print(f"  Port       : {args.port}")
    print(
        f"  Obs sleep  : {'disabled (--no-sleep)' if args.no_sleep else f'{obs_sleep}s'}"
    )
    print(f"  Inference  : skipped (dummy scroll action used)")
    print(SEPARATOR)

    playwright_url = "http://localhost:{port}"
    browser_config = build_browser_config(playwright_url, zero_obs_sleep=args.no_sleep)
    browser_config.playwright_port = args.port
    client = initialize_session(args.url, browser_config)

    timings: dict[str, list[float]] = {"obs": [], "md": [], "exec": []}
    completed_steps = 0

    # ── initial observation (not timed — page is already fully loaded) ────────
    print("\nFetching initial observation (untimed, warmup)...")
    try:
        current_observation = get_observation_with_retry(client)
    except Exception as e:
        print(f"[ERROR] Could not get initial observation: {e}")
        client.close()
        sys.exit(1)
    print("Initial observation received.\n")

    # ── step loop ─────────────────────────────────────────────────────────────
    for step in range(1, args.steps + 1):
        print(f"--- Step {step} ---")

        # 1. Markdown conversion
        t0 = time.perf_counter()
        markdown_content = convert_html_to_markdown(current_observation.__dict__)
        t_md = time.perf_counter() - t0

        if not markdown_content:
            print(
                f"  [WARN] Markdown conversion returned empty/None at step {step}. Stopping."
            )
            break

        # 2. Action execution (dummy scroll — no model needed)
        t0 = time.perf_counter()
        status = client.action(DUMMY_ACTION)
        t_exec = time.perf_counter() - t0

        if status == BrowserStatus.ERROR:
            print(f"  [WARN] Action execution failed at step {step}. Stopping.")
            break

        # 3. Observation (after action — this is the meaningful timing target)
        t0 = time.perf_counter()
        try:
            current_observation = get_observation_with_retry(client)
        except Exception as e:
            print(f"  [WARN] Observation failed at step {step}: {e}. Stopping.")
            break
        t_obs = time.perf_counter() - t0

        # Record & report
        timings["obs"].append(t_obs)
        timings["md"].append(t_md)
        timings["exec"].append(t_exec)
        completed_steps += 1

        print_step_timing(step, t_obs, t_md, t_exec)

    # ── cleanup & summary ─────────────────────────────────────────────────────
    try:
        client.close()
    except Exception:
        pass

    print_summary(timings, obs_sleep)


if __name__ == "__main__":
    main()
