"""
Pull a small reproducible sample from the v2 log, grouped so the user can
manually spot-check the judge's calls — especially the 'HTTP 2xx but
classified WEBSITE_DOWN' cases.

Re-parses the v2 log (rather than the trimmed CSV) so we can include the
judge's reasoning text alongside the website and HTTP status.

Output: a single markdown file at feasibility_results/sanity_check_sample.md
"""

import random
import re
from pathlib import Path
from textwrap import shorten

LOG = Path(__file__).parent / "train_feasible_5k_v2.log"
OUT = Path(__file__).parent / "sanity_check_sample.md"

HEADER_RE = re.compile(r"\[Accepted:\s*(\d+)/\d+\s*\|\s*Checked:\s*(\d+)\]\s*(\S+)")
INSTR_RE = re.compile(r"Instruction:\s*(.*)")
HTTP_RE = re.compile(r"HTTP:\s*status=(\S+)\s+reachable=(\S+)(?:\s+error=(\S+))?")
CLS_RE = re.compile(r"Classification\s*:\s*(\S+)\s*\(confidence=([\d.]+)\)")
REASON_RE = re.compile(r"Reasoning\s*:\s*(.*)")
DECISION_RE = re.compile(r"^\s*([✓✗])\s+(Accepted|Skipped)")

records = []
cur = None
with LOG.open() as f:
    for raw in f:
        line = raw.rstrip("\n")
        m = HEADER_RE.search(line)
        if m:
            if cur is not None:
                records.append(cur)
            cur = {
                "website": m.group(3),
                "instruction": None,
                "http_status": None,
                "http_reachable": None,
                "classification": None,
                "confidence": None,
                "reasoning": None,
                "decision": None,
            }
            continue
        if cur is None:
            continue
        m = INSTR_RE.search(line)
        if m and cur["instruction"] is None:
            cur["instruction"] = m.group(1).strip()
            continue
        m = HTTP_RE.search(line)
        if m:
            cur["http_status"] = m.group(1)
            cur["http_reachable"] = m.group(2)
            continue
        m = CLS_RE.search(line)
        if m:
            cur["classification"] = m.group(1)
            cur["confidence"] = float(m.group(2))
            continue
        m = REASON_RE.search(line)
        if m and cur["reasoning"] is None:
            cur["reasoning"] = m.group(1).strip()
            continue
        m = DECISION_RE.match(line)
        if m:
            cur["decision"] = "Accepted" if m.group(1) == "✓" else "Skipped"
if cur is not None:
    records.append(cur)
records = [r for r in records if r["classification"] is not None]


def is_2xx(s):
    return s and s.isdigit() and 200 <= int(s) < 300


def is_4xx_5xx(s):
    return s and s.isdigit() and int(s) >= 400


def is_unreachable(s):
    return s in (None, "None", "")


buckets = {
    "WEBSITE_DOWN — HTTP 2xx (judge override of probe)":
        [r for r in records if r["classification"] == "WEBSITE_DOWN" and is_2xx(r["http_status"])],
    "WEBSITE_DOWN — HTTP 4xx/5xx":
        [r for r in records if r["classification"] == "WEBSITE_DOWN" and is_4xx_5xx(r["http_status"])],
    "WEBSITE_DOWN — unreachable":
        [r for r in records if r["classification"] == "WEBSITE_DOWN" and is_unreachable(r["http_status"])],
    "CONTENT_LIKELY_OUTDATED":
        [r for r in records if r["classification"] == "CONTENT_LIKELY_OUTDATED"],
    "UNCERTAIN":
        [r for r in records if r["classification"] == "UNCERTAIN"],
    "FEASIBLE — low confidence (rejected)":
        [r for r in records if r["classification"] == "FEASIBLE" and r["decision"] == "Skipped"],
    "FEASIBLE — accepted":
        [r for r in records if r["classification"] == "FEASIBLE" and r["decision"] == "Accepted"],
}

rng = random.Random(42)
N_PER_BUCKET = 4

with OUT.open("w") as f:
    f.write("# Sanity-check sample of feasibility judge\n\n")
    f.write("Random sample (seed=42) of ~4 tasks per category from the v2 log. ")
    f.write("Visit each website and verify the judge's call. ")
    f.write("Note the judge's reasoning is the *only* thing that determines the classification — ")
    f.write("the HTTP status was used as input but not as the decision.\n\n")
    for label, items in buckets.items():
        f.write(f"## {label}  *(pool size: {len(items)})*\n\n")
        sample = rng.sample(items, min(N_PER_BUCKET, len(items)))
        for r in sample:
            instr = shorten(r["instruction"] or "(no instruction)", 220, placeholder="…")
            reason = shorten(r["reasoning"] or "(no reasoning)", 350, placeholder="…")
            f.write(f"- **{r['website']}**  `HTTP {r['http_status']} reachable={r['http_reachable']}`  ")
            f.write(f"`conf={r['confidence']:.2f}`  `→ {r['decision']}`\n")
            f.write(f"  - *Task:* {instr}\n")
            f.write(f"  - *Judge:* {reason}\n\n")

print(f"wrote {OUT}")
print(f"\nbucket sizes:")
for label, items in buckets.items():
    print(f"  {len(items):>4}  {label}")
