"""
sample_feasible_tasks.py

Iterates through the test dataset (shuffled with a fixed seed) and calls Gemini
(via url_context) to classify each task's feasibility.  Keeps going until
exactly TARGET_COUNT tasks are found that Gemini classifies as FEASIBLE with
confidence >= MIN_CONFIDENCE.

The resulting tasks are saved to a CSV file with the same columns as the
original test dataset, plus extra columns from the feasibility assessment:
  feasibility_classification, feasibility_confidence, feasibility_reasoning,
  http_status_code, http_reachable, http_error

Usage
-----
  # Default: collect 50 FEASIBLE tasks, confidence >= 0.95
  python sample_feasible_tasks.py

  # Custom target / confidence / seed
  python sample_feasible_tasks.py --target 50 --min_confidence 0.95 --seed 42

  # Resume a previous interrupted run
  python sample_feasible_tasks.py --resume_from feasibility_results/feasible_sample_<timestamp>.csv

  # Use Vertex AI instead of the free AI Studio key (no daily quota cap)
  python sample_feasible_tasks.py --vertex_project my-gcp-project-id
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

# Force unbuffered stdout so logs appear immediately when redirected
sys.stdout.reconfigure(line_buffering=True)

# Pakistan Standard Time (UTC+5)
PKT = timezone(timedelta(hours=5))


def log(msg: str = "") -> None:
    """Print a message prefixed with the current Pakistan time (AM/PM)."""
    ts = datetime.now(PKT).strftime("%I:%M:%S %p")
    # Preserve leading newlines before the timestamp
    leading = ""
    while msg.startswith("\n"):
        leading += "\n"
        msg = msg[1:]
    print(f"{leading}[{ts}] {msg}")


import pandas as pd
import requests
from google import genai
from google.genai.types import GenerateContentConfig, HttpOptions, Tool, UrlContext

# ── Gemini config ──────────────────────────────────────────────────────────────
# Read from env only — no hardcoded fallback. For Vertex AI auth, pass
# --vertex_project on the CLI; for AI Studio, set JUDGE_API_KEY.
JUDGE_API_KEY = os.environ.get("JUDGE_API_KEY")
JUDGE_MODEL = "gemini-2.5-flash"

# ── Defaults ───────────────────────────────────────────────────────────────────
TARGET_COUNT = 50
MIN_CONFIDENCE = 0.95
DEFAULT_DATASET = "data/insta-150k-test.csv"
DATASET_SPLITS = {
    "test": "data/insta-150k-test.csv",
    "train": "data/insta-150k-train.csv",
}
DEFAULT_OUTPUT_DIR = "feasibility_results"
DEFAULT_SEED = 42

# ── Feasibility labels ─────────────────────────────────────────────────────────
FEASIBLE = "FEASIBLE"
WEBSITE_DOWN = "WEBSITE_DOWN"
CONTENT_OUTDATED = "CONTENT_LIKELY_OUTDATED"
UNCERTAIN = "UNCERTAIN"

# ── HTTP probe ─────────────────────────────────────────────────────────────────
HTTP_TIMEOUT = 10
LLM_TIMEOUT = 120  # seconds for Gemini generate_content (includes url_context fetches)
HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# ── Gemini client (lazy, cached) ───────────────────────────────────────────────
_gemini_client: Optional[genai.Client] = None
_vertex_project: Optional[str] = None
_vertex_location: str = "us-central1"


def get_gemini_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        if _vertex_project:
            _gemini_client = genai.Client(
                vertexai=True,
                project=_vertex_project,
                location=_vertex_location,
            )
        else:
            _gemini_client = genai.Client(api_key=JUDGE_API_KEY)
    return _gemini_client


# ── Prompt ─────────────────────────────────────────────────────────────────────
FEASIBILITY_PROMPT = """\
Today is {today}. You are auditing whether a web-navigation task created roughly
one year ago is still completable.

Use the url_context tool to visit the website below and read its current content,
then decide whether the task can still be completed.

Website (starting URL): {website}
Task instruction      : {instruction}
Success criteria      : {criteria}
Steps (as written ~1 year ago): {steps}
HTTP probe result     : {http_status}

After reading the page, classify the task into exactly one of these categories:

FEASIBLE
  The website is operational and the specific information / page / functionality
  required by the task is still present.

WEBSITE_DOWN
  The website is unreachable, returns a server error (5xx), or the domain no
  longer resolves.

CONTENT_LIKELY_OUTDATED
  The site is up but the specific page, data, or information the task requires
  is no longer available — e.g. an expired event page, changed personnel
  listing, discontinued product, or time-specific data that is now stale.

UNCERTAIN
  The starting page loaded but doesn't have enough information to confirm
  feasibility without further navigation into the site.

Respond with a JSON object and nothing else:
{{
  "classification": "<one of the four labels above>",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<one or two sentences based on what you actually saw on the page>"
}}
"""


# ── Helpers ────────────────────────────────────────────────────────────────────


def probe_website(website: str) -> dict:
    url = website if website.startswith("http") else f"https://{website}"
    result = {
        "url": url,
        "status_code": None,
        "reachable": False,
        "error": None,
        "final_url": url,
    }
    try:
        resp = requests.get(
            url, timeout=HTTP_TIMEOUT, headers=HTTP_HEADERS, allow_redirects=True
        )
        result["status_code"] = resp.status_code
        result["reachable"] = resp.status_code < 500
        result["final_url"] = resp.url
    except requests.exceptions.SSLError:
        try:
            http_url = website if website.startswith("http") else f"http://{website}"
            resp = requests.get(
                http_url,
                timeout=HTTP_TIMEOUT,
                headers=HTTP_HEADERS,
                allow_redirects=True,
            )
            result["status_code"] = resp.status_code
            result["reachable"] = resp.status_code < 500
            result["final_url"] = resp.url
        except Exception as e:
            result["error"] = str(e)
    except requests.exceptions.ConnectionError:
        result["error"] = "connection_error"
    except requests.exceptions.Timeout:
        result["error"] = "timeout"
    except Exception as e:
        result["error"] = str(e)
    return result


def _parse_retry_delay(exc: Exception) -> Optional[float]:
    msg = str(exc)
    m = re.search(r"'retryDelay':\s*'([0-9.]+)s'", msg)
    if not m:
        m = re.search(r'"retryDelay":\s*"([0-9.]+)s"', msg)
    return float(m.group(1)) if m else None


def assess_feasibility_with_llm(task: dict, http_probe: dict, retries: int = 5) -> dict:
    today = datetime.now().strftime("%B %Y")
    url = (
        task["website"]
        if task["website"].startswith("http")
        else f"https://{task['website']}"
    )
    status_str = (
        f"HTTP {http_probe['status_code']}"
        if http_probe["status_code"] is not None
        else f"Unreachable ({http_probe.get('error', 'unknown error')})"
    )

    # Skip LLM call if website is definitively unreachable
    if not http_probe["reachable"] and http_probe["status_code"] is None:
        return {
            "classification": WEBSITE_DOWN,
            "confidence": 0.95,
            "reasoning": f"Website could not be reached: {http_probe.get('error', 'unknown error')}",
        }

    if http_probe.get("status_code") == 403:
        return {
            "classification": WEBSITE_DOWN,
            "confidence": 0.9,
            "reasoning": "HTTP probe returned 403 Forbidden — site is blocking automated access.",
        }

    prompt = FEASIBILITY_PROMPT.format(
        today=today,
        website=url,
        instruction=task["instruction"],
        criteria=task.get("criteria", "N/A"),
        steps=task.get("steps", "N/A"),
        http_status=status_str,
    )

    client = get_gemini_client()
    url_context_tool = Tool(url_context=UrlContext)

    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=JUDGE_MODEL,
                contents=prompt,
                config=GenerateContentConfig(
                    tools=[url_context_tool],
                    response_modalities=["TEXT"],
                    temperature=0.2,
                    http_options=HttpOptions(timeout=LLM_TIMEOUT * 1000),
                ),
            )
            raw = response.text.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            parsed = json.loads(raw)
            assert parsed.get("classification") in (
                FEASIBLE,
                WEBSITE_DOWN,
                CONTENT_OUTDATED,
                UNCERTAIN,
            )
            return parsed
        except Exception as e:
            is_rate_limit = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
            if attempt < retries - 1:
                if is_rate_limit:
                    delay = _parse_retry_delay(e)
                    if delay is None:
                        delay = 60.0
                    delay += 5.0
                    log(
                        f"  Rate limited (429). Waiting {delay:.0f}s before retry "
                        f"(attempt {attempt + 1}/{retries})..."
                    )
                    time.sleep(delay)
                else:
                    time.sleep(2**attempt)
            else:
                return {
                    "classification": UNCERTAIN,
                    "confidence": 0.0,
                    "reasoning": f"LLM assessment failed: {e}",
                }
    return {
        "classification": UNCERTAIN,
        "confidence": 0.0,
        "reasoning": "No attempts made",
    }


# ── Core sampling loop ─────────────────────────────────────────────────────────


def collect_feasible_tasks(
    df: pd.DataFrame,
    target: int,
    min_confidence: float,
    seed: int,
    output_dir: str,
    resume_from: Optional[str] = None,
    rejected_from: Optional[str] = None,
) -> pd.DataFrame:
    """
    Shuffle the dataset and iterate row-by-row, calling Gemini for each task.
    Stops as soon as `target` tasks with classification=FEASIBLE and
    confidence >= `min_confidence` have been found.

    Saves a running CSV of accepted tasks after each new acceptance so progress
    is never lost.  Supports resuming from a prior output CSV.
    Also tracks rejected tasks in a separate CSV to avoid re-checking.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Shuffle the full dataset once
    shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # ── Load prior accepted tasks if resuming ──────────────────────────────────
    accepted_rows: list[dict] = []
    seen_websites: set[str] = set()

    if resume_from and os.path.isfile(resume_from):
        prior_df = pd.read_csv(resume_from)
        accepted_rows = prior_df.to_dict(orient="records")
        seen_websites = set(prior_df["website"].tolist())
        log(
            f"Resuming from '{resume_from}': "
            f"{len(accepted_rows)} tasks already collected, "
            f"{target - len(accepted_rows)} remaining."
        )

    # ── Load prior rejected tasks if resuming ───────────────────────────────────
    rejected_rows: list[dict] = []
    if rejected_from and os.path.isfile(rejected_from):
        rejected_df = pd.read_csv(rejected_from)
        rejected_rows = rejected_df.to_dict(orient="records")
        seen_websites.update(rejected_df["website"].tolist())
        log(
            f"Loaded {len(rejected_rows)} previously rejected tasks from '{rejected_from}'"
        )

    # ── Determine output path (reuse resume_from path or create new) ───────────
    if resume_from and os.path.isfile(resume_from):
        out_path = resume_from
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(output_dir, f"feasible_sample_{ts}.csv")

    # Rejected tasks output path
    rejected_out_path = out_path.replace("feasible_sample", "rejected_sample")

    total_checked = 0

    for _, row in shuffled.iterrows():
        if len(accepted_rows) >= target:
            break

        # Skip websites already accepted (handles resume)
        if row["website"] in seen_websites:
            continue

        total_checked += 1
        log(
            f"\n[Accepted: {len(accepted_rows)}/{target} | Checked: {total_checked}] "
            f"{row['website']}"
        )
        log(f"  Instruction: {str(row['instruction'])[:90]}...")

        # Step 1 – HTTP probe
        probe = probe_website(row["website"])
        log(
            f"  HTTP: status={probe['status_code']}  reachable={probe['reachable']}"
            + (f"  error={probe['error']}" if probe["error"] else "")
        )

        # Step 2 – LLM assessment
        log("  Asking Gemini for feasibility classification...")
        assessment = assess_feasibility_with_llm(row.to_dict(), probe)
        cls = assessment.get("classification", UNCERTAIN)
        conf = assessment.get("confidence", 0.0)
        reason = assessment.get("reasoning", "")
        log(f"  Classification : {cls}  (confidence={conf:.2f})")
        log(f"  Reasoning      : {reason}")

        # Accept if FEASIBLE with sufficient confidence
        if cls == FEASIBLE and conf >= min_confidence:
            record = row.to_dict()
            record["feasibility_classification"] = cls
            record["feasibility_confidence"] = conf
            record["feasibility_reasoning"] = reason
            record["http_status_code"] = probe["status_code"]
            record["http_reachable"] = probe["reachable"]
            record["http_error"] = probe["error"]
            accepted_rows.append(record)
            seen_websites.add(row["website"])
            log(f"  ✓ Accepted  ({len(accepted_rows)}/{target})")

            # Save incrementally after every accepted task
            pd.DataFrame(accepted_rows).to_csv(out_path, index=False)
        else:
            # Save rejected task
            rejected_record = row.to_dict()
            rejected_record["feasibility_classification"] = cls
            rejected_record["feasibility_confidence"] = conf
            rejected_record["feasibility_reasoning"] = reason
            rejected_record["http_status_code"] = probe["status_code"]
            rejected_record["http_reachable"] = probe["reachable"]
            rejected_record["http_error"] = probe["error"]
            rejected_rows.append(rejected_record)
            log(f"  ✗ Skipped  (cls={cls}, conf={conf:.2f})")
            pd.DataFrame(rejected_rows).to_csv(rejected_out_path, index=False)

        # Small delay to stay within rate limits
        time.sleep(2.0)

    result_df = pd.DataFrame(accepted_rows)

    log(f"\n{'=' * 65}")
    log(f"DONE — collected {len(result_df)} feasible tasks")
    log(f"  Total tasks checked : {total_checked}")
    log(
        f"  Acceptance rate     : {len(result_df) / max(total_checked, 1) * 100:.1f}%"
    )
    log(f"  Output saved to     : {out_path}")
    log(f"{'=' * 65}")

    return result_df


# ── CLI ────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sample tasks from the test dataset that Gemini classifies as "
            "FEASIBLE with confidence >= min_confidence."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--split",
        choices=["test", "train"],
        default="test",
        help="Which dataset split to sample from: 'test' or 'train' (default: test)",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help=(
            "Explicit path to a CSV file. Overrides --split if provided. "
            f"Defaults to '{DATASET_SPLITS['test']}' when --split=test."
        ),
    )
    parser.add_argument(
        "--target",
        type=int,
        default=TARGET_COUNT,
        help=f"Number of feasible tasks to collect (default: {TARGET_COUNT})",
    )
    parser.add_argument(
        "--min_confidence",
        type=float,
        default=MIN_CONFIDENCE,
        help=f"Minimum Gemini confidence to accept a task (default: {MIN_CONFIDENCE})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for shuffling the dataset (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to write output CSV (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--resume_from",
        default=None,
        metavar="PATH",
        help=(
            "Path to a prior feasible_sample CSV. Already-accepted tasks are "
            "loaded and their websites are skipped. Useful after a quota reset."
        ),
    )
    parser.add_argument(
        "--rejected_from",
        default=None,
        metavar="PATH",
        help=(
            "Path to a prior rejected_sample CSV. Already-checked (rejected) tasks "
            "are loaded and their websites are skipped."
        ),
    )
    parser.add_argument(
        "--vertex_project",
        default=None,
        metavar="PROJECT_ID",
        help=(
            "Google Cloud project ID to use Vertex AI instead of AI Studio. "
            "Requires GOOGLE_APPLICATION_CREDENTIALS or gcloud ADC. "
            "Bypasses the free-tier daily quota."
        ),
    )
    parser.add_argument(
        "--vertex_location",
        default="us-central1",
        metavar="REGION",
        help="Vertex AI region (default: us-central1).",
    )
    args = parser.parse_args()

    if args.vertex_project:
        global _vertex_project, _vertex_location
        _vertex_project = args.vertex_project
        _vertex_location = args.vertex_location
        log(
            f"Using Vertex AI  project={_vertex_project}  location={_vertex_location}"
        )
    else:
        log("Using AI Studio API key auth (free tier)")

    # Resolve dataset path: explicit --dataset overrides --split
    dataset_path = args.dataset if args.dataset else DATASET_SPLITS[args.split]

    log(
        f"\nLoading dataset: {dataset_path}  (split={args.split if not args.dataset else 'custom'})"
    )
    df = pd.read_csv(dataset_path)
    log(f"Dataset loaded: {len(df)} rows")
    log(
        f"Target: {args.target} FEASIBLE tasks with confidence >= {args.min_confidence}"
    )

    collect_feasible_tasks(
        df,
        target=args.target,
        min_confidence=args.min_confidence,
        seed=args.seed,
        output_dir=args.output_dir,
        resume_from=args.resume_from,
        rejected_from=args.rejected_from,
    )


if __name__ == "__main__":
    main()
