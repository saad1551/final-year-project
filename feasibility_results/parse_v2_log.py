"""
Parse train_feasible_5k_v2.log into per-task records and emit:
  - per-class breakdown (FEASIBLE / WEBSITE_DOWN / CONTENT_LIKELY_OUTDATED / UNCERTAIN)
  - HTTP-probe vs Gemini-classifier disagreement (e.g. HTTP 200 but classified WEBSITE_DOWN)
  - a per-task CSV for reproducibility
"""

import csv
import re
from collections import Counter
from pathlib import Path

LOG = Path(__file__).parent / "train_feasible_5k_v2.log"
OUT_CSV = Path(__file__).parent / "v2_log_parsed.csv"

HEADER_RE = re.compile(r"\[Accepted:\s*(\d+)/\d+\s*\|\s*Checked:\s*(\d+)\]\s*(\S+)")
HTTP_RE = re.compile(r"HTTP:\s*status=(\S+)\s+reachable=(\S+)(?:\s+error=(\S+))?")
CLS_RE = re.compile(r"Classification\s*:\s*(\S+)\s*\(confidence=([\d.]+)\)")
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
                "accepted_so_far": int(m.group(1)),
                "checked_so_far": int(m.group(2)),
                "website": m.group(3),
                "http_status": None,
                "http_reachable": None,
                "http_error": None,
                "classification": None,
                "confidence": None,
                "decision": None,
            }
            continue
        if cur is None:
            continue
        m = HTTP_RE.search(line)
        if m:
            cur["http_status"] = m.group(1)
            cur["http_reachable"] = m.group(2)
            cur["http_error"] = m.group(3)
            continue
        m = CLS_RE.search(line)
        if m:
            cur["classification"] = m.group(1)
            cur["confidence"] = float(m.group(2))
            continue
        m = DECISION_RE.match(line)
        if m:
            cur["decision"] = "Accepted" if m.group(1) == "✓" else "Skipped"

if cur is not None:
    records.append(cur)

# Drop incomplete trailing record (no classification)
complete = [r for r in records if r["classification"] is not None and r["decision"] is not None]

print(f"parsed task blocks:       {len(records)}")
print(f"complete (cls+decision):  {len(complete)}")

decisions = Counter(r["decision"] for r in complete)
print(f"\ndecisions:")
for k, v in decisions.most_common():
    print(f"  {k}: {v}")

cls_all = Counter(r["classification"] for r in complete)
print(f"\nclassification breakdown (all complete tasks, n={len(complete)}):")
for k, v in cls_all.most_common():
    pct = 100 * v / len(complete)
    print(f"  {k:<25} {v:>4}  ({pct:5.1f}%)")

rejected = [r for r in complete if r["decision"] == "Skipped"]
cls_rej = Counter(r["classification"] for r in rejected)
print(f"\nrejected breakdown (n={len(rejected)}):")
for k, v in cls_rej.most_common():
    pct = 100 * v / len(rejected)
    print(f"  {k:<25} {v:>4}  ({pct:5.1f}%)")

# HTTP-probe vs Gemini disagreement: HTTP 2xx-reachable but classifier says WEBSITE_DOWN
http_200_down = [
    r for r in complete
    if r["classification"] == "WEBSITE_DOWN"
    and r["http_status"] not in (None, "None")
    and r["http_status"].isdigit()
    and 200 <= int(r["http_status"]) < 300
]
http_404_feasible = [
    r for r in complete
    if r["classification"] == "FEASIBLE"
    and r["http_status"] not in (None, "None")
    and r["http_status"].isdigit()
    and int(r["http_status"]) >= 400
]
print(f"\nHTTP-probe vs classifier disagreement:")
print(f"  HTTP 2xx but classifier=WEBSITE_DOWN: {len(http_200_down)}")
print(f"  HTTP >=400 but classifier=FEASIBLE:   {len(http_404_feasible)}")
total_down = sum(1 for r in complete if r["classification"] == "WEBSITE_DOWN")
if total_down:
    print(f"  -> {len(http_200_down)/total_down:.1%} of WEBSITE_DOWN had a 2xx HTTP probe")
    print(f"     (i.e., a reachability-only filter would have missed these)")

# HTTP probe alone success/fail vs classifier feasibility
print(f"\nHTTP probe outcomes by final classification:")
buckets = {}
for r in complete:
    s = r["http_status"]
    if s in (None, "None"):
        cat = "unreachable"
    elif s.isdigit() and 200 <= int(s) < 300:
        cat = "2xx"
    elif s.isdigit() and 300 <= int(s) < 400:
        cat = "3xx"
    elif s.isdigit() and 400 <= int(s) < 500:
        cat = "4xx"
    elif s.isdigit() and int(s) >= 500:
        cat = "5xx"
    else:
        cat = "other"
    buckets.setdefault(r["classification"], Counter())[cat] += 1
cats = ["2xx", "3xx", "4xx", "5xx", "unreachable", "other"]
print(f"  {'class':<25} " + " ".join(f"{c:>5}" for c in cats))
for cls, ctr in buckets.items():
    print(f"  {cls:<25} " + " ".join(f"{ctr.get(c,0):>5}" for c in cats))

# Write per-task CSV for reproducibility
with OUT_CSV.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(complete[0].keys()))
    w.writeheader()
    for r in complete:
        w.writerow(r)
print(f"\nwrote per-task CSV: {OUT_CSV}")
