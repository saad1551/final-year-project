"""
check_task_feasibility.py

Validates the hypothesis that low evaluation success rates are partly caused by
outdated tasks — websites that have changed, pages that no longer exist, or
information that is no longer available.

Two modes:
  1. Fresh feasibility check (default)
     Samples the same tasks as the evaluation run (seed=42, sample_size=50),
     does an HTTP reachability probe for each website, then asks Gemini to
     classify whether the task is still completable today.

  2. Eval results analysis (--eval_results)
     Parses the JSON output of evaluate_checkpoint.py and inspects the page
     observations recorded during the actual run.  Failed tasks are scanned
     for "page not found / content removed" signals, so you can see what
     fraction of failures were caused by stale content vs. genuine agent errors.

Usage examples
--------------
  # Fresh check on the same 50 tasks used during evaluation (AI Studio key):
  python check_task_feasibility.py --sample_size 50 --seed 42

  # Same check via Vertex AI (no daily quota cap; needs GCP credentials):
  export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
  python check_task_feasibility.py --sample_size 50 --seed 42 \\
      --vertex_project my-gcp-project-id

  # Resume a previous run that was interrupted by quota limits:
  python check_task_feasibility.py --sample_size 50 --seed 42 \\
      --resume_from feasibility_results/feasibility_check_<timestamp>.json \\
      --vertex_project my-gcp-project-id

  # Analyse results you already have (download the JSON from RunPod first):
  python check_task_feasibility.py \\
      --eval_results eval_results/checkpoint_results_*.json \\
      --eval_results eval_results/base_results_*.json

  # Both at once:
  python check_task_feasibility.py --sample_size 50 --seed 42 \\
      --eval_results eval_results/checkpoint_results_*.json
"""

import argparse
import json
import os
import random
import re
import time
from datetime import datetime
from typing import Optional

import pandas as pd
import requests
from google import genai
from google.genai.types import Tool, GenerateContentConfig, UrlContext

# ── Gemini config ─────────────────────────────────────────────────────────────
# Read from env only — no hardcoded fallback.
# For Gemini AI Studio (free tier, API key auth), get a key at
# https://aistudio.google.com/apikey and set JUDGE_API_KEY.
# gemini-2.0-flash: 1500 requests/day on the AI Studio free tier.
# On Vertex AI there is no daily cap — pay-per-use only.
import os as _os
JUDGE_API_KEY = _os.environ.get("JUDGE_API_KEY")
if not JUDGE_API_KEY:
    raise RuntimeError(
        "JUDGE_API_KEY environment variable is not set. "
        "Get a key at https://aistudio.google.com/apikey and `export JUDGE_API_KEY=...`."
    )
JUDGE_MODEL = "gemini-2.0-flash"

# ── HTTP probe ─────────────────────────────────────────────────────────────────
HTTP_TIMEOUT = 10  # seconds
HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# ── Feasibility categories ─────────────────────────────────────────────────────
FEASIBLE = "FEASIBLE"
WEBSITE_DOWN = "WEBSITE_DOWN"
CONTENT_OUTDATED = "CONTENT_LIKELY_OUTDATED"
UNCERTAIN = "UNCERTAIN"

# Keywords in page observations that suggest a broken / missing page
STALE_CONTENT_PATTERNS = [
    r"\b404\b",
    r"page\s+(not|no longer)\s+found",
    r"page\s+doesn['\u2019]t\s+exist",
    r"content\s+(has\s+been\s+)?removed",
    r"this\s+page\s+(has\s+been\s+)?moved",
    r"no\s+longer\s+available",
    r"site\s+(can['\u2019]t\s+be\s+)?reached",
    r"err_name_not_resolved",
    r"err_connection_refused",
    r"domain\s+not\s+found",
    r"this\s+domain\s+is\s+for\s+sale",
    r"under\s+construction",
    r"coming\s+soon",
    r"account\s+suspended",
    r"access\s+denied",
    r"forbidden",
]
_STALE_RE = re.compile("|".join(STALE_CONTENT_PATTERNS), re.IGNORECASE)


# ══════════════════════════════════════════════════════════════════════════════
# Stratified task sampling (mirrors evaluate_checkpoint.py logic exactly)
# ══════════════════════════════════════════════════════════════════════════════


def _instruction_bucket(instruction: str) -> str:
    """Assign an instruction to a coarse action-type bucket based on its first verb."""
    text = instruction.lower().strip()
    patterns = [
        ("search", r"\b(search|look for)\b"),
        ("count", r"\b(how many|count|total number)\b"),
        ("fill_submit", r"\b(fill|submit|enter|type)\b"),
        ("navigate", r"\b(go to|navigate|open|click|visit)\b"),
        ("find_locate", r"\b(find|locate|identify|determine|what is|what are)\b"),
    ]
    for bucket, pattern in patterns:
        if re.search(pattern, text):
            return bucket
    return "other"


def sample_tasks(df: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    """Stratified random sample — identical logic to evaluate_checkpoint.py."""
    random.seed(seed)
    df = df.copy().reset_index(drop=True)
    df["_bucket"] = df["instruction"].apply(_instruction_bucket)

    bucket_counts = df["_bucket"].value_counts()
    total = len(df)
    raw_allocs = {
        b: (count / total) * sample_size for b, count in bucket_counts.items()
    }
    floored = {b: int(v) for b, v in raw_allocs.items()}
    remainder = sample_size - sum(floored.values())
    frac_parts = sorted(raw_allocs.items(), key=lambda x: -(x[1] - int(x[1])))
    for i, (b, _) in enumerate(frac_parts):
        if i < remainder:
            floored[b] += 1

    sampled_frames = []
    for bucket, n in floored.items():
        if n == 0:
            continue
        bucket_df = df[df["_bucket"] == bucket]
        n_draw = min(n, len(bucket_df))
        sampled_frames.append(bucket_df.sample(n=n_draw, random_state=seed))

    sample = (
        pd.concat(sampled_frames)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )
    sample = sample.drop(columns=["_bucket"])

    if len(sample) < sample_size:
        remaining = df[~df.index.isin(sample.index)].drop(columns=["_bucket"])
        shortfall = sample_size - len(sample)
        extra = remaining.sample(n=min(shortfall, len(remaining)), random_state=seed)
        sample = pd.concat([sample, extra]).reset_index(drop=True)

    print(f"Sampled {len(sample)} tasks (seed={seed})")
    return sample


# ══════════════════════════════════════════════════════════════════════════════
# HTTP probe
# ══════════════════════════════════════════════════════════════════════════════


def probe_website(website: str) -> dict:
    """
    Do a simple HTTP GET on the website root and return a status dict.
    Returns:
        {
          "url": str,
          "status_code": int | None,
          "reachable": bool,
          "error": str | None,
          "final_url": str,       # after redirects
        }
    """
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
        # Try plain HTTP as fallback
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
    except requests.exceptions.ConnectionError as e:
        result["error"] = "connection_error"
    except requests.exceptions.Timeout:
        result["error"] = "timeout"
    except Exception as e:
        result["error"] = str(e)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# LLM feasibility assessment  (uses Gemini url_context to read the live page)
# ══════════════════════════════════════════════════════════════════════════════

_gemini_client: Optional[genai.Client] = None

# Set by main() when --vertex_project is supplied
_vertex_project: Optional[str] = None
_vertex_location: str = "us-central1"


def get_gemini_client() -> genai.Client:
    """
    Return a cached Gemini client.

    - If _vertex_project is set, authenticates via Vertex AI (uses
      GOOGLE_APPLICATION_CREDENTIALS or gcloud ADC — no API key needed).
    - Otherwise falls back to AI Studio API key auth.
    """
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


def _parse_retry_delay(exc: Exception) -> Optional[float]:
    """
    Try to extract the suggested retryDelay (in seconds) from a Gemini 429
    exception message.  Returns None if the value cannot be parsed.
    """
    msg = str(exc)
    # The error detail looks like: 'retryDelay': '50s'  or  "retryDelay": "50.054s"
    m = re.search(r"'retryDelay':\s*'([0-9.]+)s'", msg)
    if not m:
        m = re.search(r'"retryDelay":\s*"([0-9.]+)s"', msg)
    if m:
        return float(m.group(1))
    return None


def assess_feasibility_with_llm(task: dict, http_probe: dict, retries: int = 5) -> dict:
    """
    Ask Gemini to visit the website via url_context and classify feasibility.
    Falls back gracefully if the url_context call fails.

    On 429 RESOURCE_EXHAUSTED the function reads the suggested retryDelay from
    the error response and sleeps that long before retrying (up to `retries`
    total attempts).
    """
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

    # If the website is already confirmed down via HTTP probe, skip the url_context
    # call and return immediately — there is nothing to visit.
    if not http_probe["reachable"] and http_probe["status_code"] is None:
        return {
            "classification": WEBSITE_DOWN,
            "confidence": 0.95,
            "reasoning": f"Website could not be reached: {http_probe.get('error', 'unknown error')}",
        }

    # HTTP 403 on the root page almost always means the site is blocking all
    # automated access (Cloudflare, bot protection, etc.).  Sending a url_context
    # request will also fail, so classify immediately.
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
                ),
            )
            raw = response.text.strip()
            # Strip markdown code fences if present
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
                        delay = 60.0  # conservative fallback
                    # Add a small buffer on top of the suggested delay
                    delay += 5.0
                    print(
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
    # Should not reach here, but satisfies type checker
    return {
        "classification": UNCERTAIN,
        "confidence": 0.0,
        "reasoning": "No attempts made",
    }


# ══════════════════════════════════════════════════════════════════════════════
# Fresh feasibility check
# ══════════════════════════════════════════════════════════════════════════════


def run_fresh_feasibility_check(
    df: pd.DataFrame,
    sample_size: int,
    seed: int,
    output_dir: str,
    resume_from: Optional[str] = None,
) -> list:
    """
    For each sampled task:
      1. HTTP probe the website
      2. Ask Gemini to classify feasibility
    Returns a list of result dicts.

    If `resume_from` is a path to a prior JSON output, tasks that were already
    successfully assessed (confidence > 0 or classification != UNCERTAIN) are
    loaded from that file and skipped.  This lets you continue after a quota
    reset without re-assessing the first ~11 tasks.
    """
    sample_df = sample_tasks(df, sample_size, seed)
    tasks = sample_df.to_dict(orient="records")

    # ── Load prior results if resuming ────────────────────────────────────────
    prior_results: dict[int, dict] = {}
    if resume_from and os.path.isfile(resume_from):
        with open(resume_from) as f:
            prior = json.load(f)
        for r in prior:
            idx = r.get("task_index")
            # Only reuse entries that had a real LLM assessment (not rate-limit failures)
            real_assessment = (
                r.get("confidence", 0.0) > 0
                or r.get("classification", UNCERTAIN) != UNCERTAIN
            )
            if idx is not None and real_assessment:
                prior_results[idx] = r
        print(
            f"Resuming from '{resume_from}': "
            f"{len(prior_results)} tasks already assessed, "
            f"{len(tasks) - len(prior_results)} remaining."
        )

    results = []
    for i, task in enumerate(tasks):
        # ── Reuse prior result if available ───────────────────────────────────
        if i in prior_results:
            r = prior_results[i]
            print(
                f"\n[{i + 1}/{len(tasks)}] {task['website']}  "
                f"[SKIP — already: {r['classification']}]"
            )
            results.append(r)
            continue

        print(f"\n[{i + 1}/{len(tasks)}] {task['website']}")
        print(f"  Instruction: {task['instruction'][:90]}...")

        # Step 1 – HTTP probe
        probe = probe_website(task["website"])
        print(
            f"  HTTP: status={probe['status_code']}  reachable={probe['reachable']}"
            + (f"  error={probe['error']}" if probe["error"] else "")
        )

        # Step 2 – LLM assessment
        print("  Asking Gemini for feasibility classification...")
        assessment = assess_feasibility_with_llm(task, probe)
        cls = assessment.get("classification", UNCERTAIN)
        conf = assessment.get("confidence", 0.0)
        reason = assessment.get("reasoning", "")
        print(f"  Classification : {cls}  (confidence={conf:.2f})")
        print(f"  Reasoning      : {reason}")

        results.append(
            {
                "task_index": i,
                "website": task["website"],
                "instruction": task["instruction"],
                "http_probe": probe,
                "classification": cls,
                "confidence": conf,
                "reasoning": reason,
            }
        )

        # Save incrementally after every task so progress is never lost
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tmp_path = os.path.join(output_dir, f"feasibility_check_{ts}.json")
        with open(tmp_path, "w") as f:
            json.dump(results, f, indent=2)

        # Small delay between tasks to stay well within rate limits
        time.sleep(2.0)

    # Print summary
    _print_feasibility_summary(results)

    # Final save
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"feasibility_check_{ts}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {path}")

    return results


def _print_feasibility_summary(results: list):
    total = len(results)
    counts = {FEASIBLE: 0, WEBSITE_DOWN: 0, CONTENT_OUTDATED: 0, UNCERTAIN: 0}
    for r in results:
        counts[r["classification"]] = counts.get(r["classification"], 0) + 1

    infeasible = counts[WEBSITE_DOWN] + counts[CONTENT_OUTDATED]
    infeasible_pct = infeasible / total * 100 if total else 0

    print(f"\n{'=' * 65}")
    print("FEASIBILITY CHECK SUMMARY")
    print(f"{'=' * 65}")
    print(f"Total tasks checked          : {total}")
    print(
        f"  FEASIBLE                   : {counts[FEASIBLE]:>3}  ({counts[FEASIBLE] / total * 100:.1f}%)"
    )
    print(
        f"  WEBSITE_DOWN               : {counts[WEBSITE_DOWN]:>3}  ({counts[WEBSITE_DOWN] / total * 100:.1f}%)"
    )
    print(
        f"  CONTENT_LIKELY_OUTDATED    : {counts[CONTENT_OUTDATED]:>3}  ({counts[CONTENT_OUTDATED] / total * 100:.1f}%)"
    )
    print(
        f"  UNCERTAIN                  : {counts[UNCERTAIN]:>3}  ({counts[UNCERTAIN] / total * 100:.1f}%)"
    )
    print(f"{'─' * 65}")
    print(
        f"  Total infeasible (down + outdated): {infeasible}/{total} ({infeasible_pct:.1f}%)"
    )
    print(f"{'=' * 65}")
    print("\nHYPOTHESIS INTERPRETATION")
    print(
        f"  If ≥ {infeasible_pct:.0f}% of tasks are infeasible, the base success-rate ceiling"
    )
    print(
        f"  (excluding infeasible tasks) would be ~{(total - infeasible) / total * 100:.0f}% of tasks."
    )
    print(f"  Observed base model success rate was 28% — if that is measured only")
    print(f"  against feasible tasks, the effective rate would be:")
    feasible_n = counts[FEASIBLE] + counts[UNCERTAIN]
    if feasible_n > 0:
        adjusted = 0.28 * total / feasible_n
        print(f"    28% × ({total} / {feasible_n} feasible) ≈ {adjusted * 100:.0f}%")
    print(f"{'=' * 65}")


# ══════════════════════════════════════════════════════════════════════════════
# Eval results analysis
# ══════════════════════════════════════════════════════════════════════════════


def classify_observation_staleness(observations: list) -> str:
    """
    Scan the first 1-3 page observations of a trajectory for stale-content
    signals.  Returns "stale_signals_found" or "no_stale_signals".
    """
    # Only look at the first few observations — that's where a missing page
    # would show up immediately.
    for obs in (observations or [])[:3]:
        if _STALE_RE.search(obs or ""):
            return "stale_signals_found"
    return "no_stale_signals"


def analyse_eval_results(paths: list) -> list:
    """
    Parse one or more eval results JSON files and classify each task as:
      - succeeded          (success score > 0.5)
      - failed_stale       (failed + stale-content signals in observations)
      - failed_agent_error (failed + no obvious stale signals)
      - crashed            (result was null / trajectory exception)
    """
    all_tasks = []
    for path in paths:
        with open(path) as f:
            data = json.load(f)

        label = os.path.basename(path)
        print(f"\nAnalysing: {label}  ({len(data)} entries)")

        for i, entry in enumerate(data):
            if entry is None:
                all_tasks.append(
                    {
                        "file": label,
                        "task_index": i,
                        "website": None,
                        "instruction": None,
                        "outcome": "crashed",
                        "success_score": None,
                        "stale_signals": None,
                        "stale_matches": [],
                    }
                )
                continue

            obs = entry.get("trajectory_markdown_observations", [])
            judgment = entry.get("judgment", {}) or {}
            success_score = judgment.get("success")
            succeeded = success_score is not None and success_score > 0.5
            stale = classify_observation_staleness(obs)

            if succeeded:
                outcome = "succeeded"
            elif stale == "stale_signals_found":
                outcome = "failed_stale"
            else:
                outcome = "failed_agent_error"

            # Collect the matching snippets for transparency
            matches = []
            for obs_text in (obs or [])[:3]:
                for m in _STALE_RE.finditer(obs_text or ""):
                    snippet = obs_text[max(0, m.start() - 40) : m.end() + 40]
                    matches.append(snippet.replace("\n", " ").strip())

            all_tasks.append(
                {
                    "file": label,
                    "task_index": i,
                    "website": entry.get("website"),
                    "instruction": entry.get("task_instruction"),
                    "outcome": outcome,
                    "success_score": success_score,
                    "stale_signals": stale,
                    "stale_matches": matches[:3],
                }
            )

    _print_eval_analysis_summary(all_tasks)
    return all_tasks


def _print_eval_analysis_summary(tasks: list):
    from collections import Counter

    files = sorted(set(t["file"] for t in tasks))

    for fname in files:
        subset = [t for t in tasks if t["file"] == fname]
        total = len(subset)
        counts = Counter(t["outcome"] for t in subset)

        succeeded = counts["succeeded"]
        failed_stale = counts["failed_stale"]
        failed_agent = counts["failed_agent_error"]
        crashed = counts["crashed"]
        total_failed = failed_stale + failed_agent + crashed

        print(f"\n{'=' * 65}")
        print(f"EVAL RESULTS ANALYSIS — {fname}")
        print(f"{'=' * 65}")
        print(f"Total tasks          : {total}")
        print(
            f"  Succeeded          : {succeeded:>3}  ({succeeded / total * 100:.1f}%)"
        )
        print(f"  Failed (crashed)   : {crashed:>3}  ({crashed / total * 100:.1f}%)")
        print(
            f"  Failed — stale page: {failed_stale:>3}  ({failed_stale / total * 100:.1f}%)"
        )
        print(
            f"  Failed — agent err : {failed_agent:>3}  ({failed_agent / total * 100:.1f}%)"
        )

        if total_failed > 0:
            stale_of_failed = failed_stale / total_failed * 100
            print(f"{'─' * 65}")
            print(
                f"  Stale-content failures as % of all failures: {stale_of_failed:.1f}%"
            )
            print(f"  (This is the lower-bound estimate — many stale pages may not")
            print(f"   contain obvious 404 text in their markdown rendering)")

        print(f"\nFailed tasks with stale-content signals:")
        for t in subset:
            if t["outcome"] == "failed_stale":
                instr = (t["instruction"] or "")[:70] + (
                    "..." if len(t["instruction"] or "") > 70 else ""
                )
                print(f"  [{t['task_index'] + 1:>2}] {t['website']}")
                print(f"       {instr}")
                for snip in t.get("stale_matches", [])[:2]:
                    print(f'       ↳ "{snip}"')
        print(f"{'=' * 65}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Validate the hypothesis that low success rates are partly caused "
            "by outdated tasks."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        default="data/insta-150k-test.csv",
        help="Path to test CSV (default: data/insta-150k-test.csv)",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=50,
        help="Number of tasks to check — should match your eval run (default: 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed — must match the seed used in your eval run (default: 42)",
    )
    parser.add_argument(
        "--output_dir",
        default="feasibility_results",
        help="Directory to write feasibility JSON output (default: feasibility_results)",
    )
    parser.add_argument(
        "--eval_results",
        nargs="+",
        default=None,
        metavar="PATH",
        help=(
            "Path(s) to eval results JSON files produced by evaluate_checkpoint.py. "
            "When provided, analyses those files in addition to (or instead of) "
            "running the fresh feasibility check."
        ),
    )
    parser.add_argument(
        "--skip_fresh",
        action="store_true",
        default=False,
        help="Skip the fresh HTTP+LLM feasibility check (only analyse --eval_results).",
    )
    parser.add_argument(
        "--resume_from",
        default=None,
        metavar="PATH",
        help=(
            "Path to a prior feasibility JSON output.  Tasks that were already "
            "successfully assessed in that file are skipped.  Useful after a "
            "quota reset to avoid re-running tasks from scratch."
        ),
    )
    parser.add_argument(
        "--vertex_project",
        default=None,
        metavar="PROJECT_ID",
        help=(
            "Google Cloud project ID to use Vertex AI instead of AI Studio. "
            "Requires GOOGLE_APPLICATION_CREDENTIALS to be set (service account "
            "JSON key) or gcloud ADC configured. Bypasses the free-tier daily quota."
        ),
    )
    parser.add_argument(
        "--vertex_location",
        default="us-central1",
        metavar="REGION",
        help="Vertex AI region (default: us-central1).",
    )
    args = parser.parse_args()

    # ── Configure Vertex AI globals if requested ───────────────────────────────
    if args.vertex_project:
        global _vertex_project, _vertex_location
        _vertex_project = args.vertex_project
        _vertex_location = args.vertex_location
        print(
            f"Using Vertex AI  project={_vertex_project}  location={_vertex_location}"
        )
    else:
        print("Using AI Studio API key auth (free tier)")

    # ── Fresh feasibility check ────────────────────────────────────────────────
    if not args.skip_fresh:
        print(f"\nLoading dataset: {args.dataset}")
        df = pd.read_csv(args.dataset)
        print(f"Dataset loaded: {len(df)} rows")
        run_fresh_feasibility_check(
            df,
            args.sample_size,
            args.seed,
            args.output_dir,
            resume_from=args.resume_from,
        )

    # ── Eval results analysis ──────────────────────────────────────────────────
    if args.eval_results:
        import glob as _glob

        paths = []
        for pattern in args.eval_results:
            expanded = _glob.glob(pattern)
            if expanded:
                paths.extend(expanded)
            elif os.path.isfile(pattern):
                paths.append(pattern)
            else:
                print(f"Warning: no files matched '{pattern}'")
        if paths:
            tasks = analyse_eval_results(paths)
            # Save the per-task breakdown
            os.makedirs(args.output_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = os.path.join(args.output_dir, f"eval_analysis_{ts}.json")
            with open(out, "w") as f:
                json.dump(tasks, f, indent=2)
            print(f"\nPer-task analysis saved → {out}")
        else:
            print("No eval results files found.")


if __name__ == "__main__":
    main()
