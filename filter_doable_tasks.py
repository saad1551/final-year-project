"""
Filter doable tasks from the training dataset using Gemini with URL context + Google Search.

Randomly samples tasks, checks if they're still completable in 2026, and saves
doable tasks to a new CSV until 2000 unique tasks are collected.

Usage:
    python filter_doable_tasks.py
    python filter_doable_tasks.py --target 2000 --output data/doable_tasks.csv
    python filter_doable_tasks.py --target 2000 --output data/doable_tasks.csv --workers 3
"""

import argparse
import os
import time
import random
import logging
from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

INPUT_CSV = "data/insta-150k-train.csv"
DEFAULT_OUTPUT_CSV = "data/doable_tasks.csv"
DEFAULT_TARGET = 2000
MODEL_ID = "gemini-3.1-pro-preview"

# Rate limiting: seconds to wait between API calls
REQUEST_DELAY = 2.0
MAX_RETRIES = 3
RETRY_BACKOFF = 5  # seconds, doubles each retry


class DoabilityResult(BaseModel):
    doable: bool = Field(
        description=(
            "True if the task can be completed by a web agent in 2026 — "
            "the website is live, the required information/action still exists, "
            "and the task is not locked to a past date or expired event."
        )
    )
    reason: str = Field(
        description="One sentence explaining why the task is or is not doable."
    )


SYSTEM_PROMPT = """You are evaluating whether a web navigation task is still completable by an automated agent in 2026.

Use the URL context tool to visit the website and the Google Search tool to verify if the required information or action is still available.

A task is DOABLE if ALL of these are true:
- The website is accessible and functional
- The specific information, product, or action required by the task still exists on the site
- The task is not tied to a past date, expired event, or discontinued feature

A task is NOT DOABLE if ANY of these are true:
- The website is down or redirects to an error/parked domain
- The specific information no longer exists (e.g., a product discontinued, a listing removed)
- The task requires completing an action for a past date (e.g., booking a flight for a 2023 date)
- The task references a specific temporary event that has since ended

Be decisive. If the website loads and the general type of action is still possible (even if exact values differ), mark it as doable."""


def build_user_prompt(website: str, instruction: str) -> str:
    url = website if website.startswith("http") else f"https://{website}"
    return (
        f"Website: {url}\n"
        f"Task: {instruction}\n\n"
        f"Visit the website using the URL context tool and determine if this task "
        f"is still completable by a web agent in 2026."
    )


def check_task_doability(
    client: genai.Client,
    website: str,
    instruction: str,
) -> Optional[DoabilityResult]:
    url = website if website.startswith("h ttp") else f"https://{website}"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=build_user_prompt(url, instruction),
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    tools=[
                        types.Tool(google_search=types.GoogleSearch()),
                        types.Tool(url_context=types.UrlContext()),
                    ],
                    response_mime_type="application/json",
                    response_json_schema=DoabilityResult.model_json_schema(),
                    temperature=0.1,
                ),
            )

            result = DoabilityResult.model_validate_json(response.text)
            return result

        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "quota" in err_str or "rate" in err_str:
                wait = RETRY_BACKOFF * (2 ** (attempt - 1))
                log.warning(f"Rate limited (attempt {attempt}/{MAX_RETRIES}). Waiting {wait}s...")
                time.sleep(wait)
            elif attempt < MAX_RETRIES:
                log.warning(f"Error on attempt {attempt}/{MAX_RETRIES}: {e}. Retrying in {RETRY_BACKOFF}s...")
                time.sleep(RETRY_BACKOFF)
            else:
                log.error(f"Failed after {MAX_RETRIES} attempts: {e}")
                return None

    return None


def load_existing_output(output_path: str) -> tuple[pd.DataFrame, set]:
    """Load already-collected tasks, return (df, set of (website, instruction) tuples)."""
    if os.path.exists(output_path):
        df = pd.read_csv(output_path)
        seen = set(zip(df["website"], df["instruction"]))
        log.info(f"Resuming: {len(df)} tasks already collected in {output_path}")
        return df, seen
    return pd.DataFrame(), set()


def main():
    parser = argparse.ArgumentParser(description="Filter doable tasks using Gemini")
    parser.add_argument("--input", default=INPUT_CSV, help="Path to training CSV")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_CSV, help="Path to output CSV")
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET, help="Number of doable tasks to collect")
    parser.add_argument("--delay", type=float, default=REQUEST_DELAY, help="Seconds between API calls")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (omit for a different random order each run)")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable.\n"
            "  export GEMINI_API_KEY=your_key_here"
        )

    client = genai.Client(api_key=api_key)

    log.info(f"Loading dataset from {args.input}...")
    df_full = pd.read_csv(args.input)
    log.info(f"Loaded {len(df_full):,} total tasks")

    # Shuffle with a random seed each run so different tasks are sampled across runs
    rng_seed = args.seed if args.seed is not None else random.randint(0, 2**31)
    log.info(f"Using random seed: {rng_seed}")
    df_shuffled = df_full.sample(frac=1, random_state=rng_seed).reset_index(drop=True)

    existing_df, seen_keys = load_existing_output(args.output)
    collected_rows = existing_df.to_dict("records") if not existing_df.empty else []

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    checked = 0
    doable_count = len(collected_rows)
    skipped_duplicates = 0

    log.info(f"Target: {args.target} doable tasks. Starting from {doable_count} already collected.")
    log.info("-" * 60)

    for _, row in df_shuffled.iterrows():
        if doable_count >= args.target:
            break

        website = str(row["website"]).strip()
        instruction = str(row["instruction"]).strip()

        # Skip already-seen tasks (for resume)
        key = (website, instruction)
        if key in seen_keys:
            skipped_duplicates += 1
            continue

        checked += 1
        log.info(
            f"[{doable_count}/{args.target} doable | checked {checked}] "
            f"{website[:40]} — {instruction[:60]}..."
        )

        result = check_task_doability(client, website, instruction)

        if result is None:
            log.warning("  → API call failed, skipping this task")
            time.sleep(args.delay)
            continue

        status = "✓ DOABLE" if result.doable else "✗ SKIP"
        log.info(f"  → {status}: {result.reason}")

        if result.doable:
            task_row = row.to_dict()
            collected_rows.append(task_row)
            seen_keys.add(key)
            doable_count += 1

            # Write incrementally so progress is never lost
            pd.DataFrame(collected_rows).to_csv(args.output, index=False)

        time.sleep(args.delay)

    log.info("-" * 60)
    log.info(f"Done. Collected {doable_count} doable tasks after checking {checked} candidates.")
    log.info(f"Output saved to: {args.output}")

    if doable_count < args.target:
        log.warning(
            f"Only collected {doable_count}/{args.target} tasks — "
            f"exhausted dataset or hit API errors. Re-run to continue."
        )


if __name__ == "__main__":
    main()
