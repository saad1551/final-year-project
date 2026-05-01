"""
Quick training-progress monitor with proper staleness detection.

Designed to run on the VM. Compares VM wall-clock to latest CSV mtime so
we don't fall for the trap of comparing against the local/conversation
clock (which can be in a different timezone or a different day entirely).
"""

import csv
import glob
import os
import statistics as st
import sys
from datetime import datetime, timezone

# Find latest non-aborted CSV
csvs = [
    c for c in glob.glob("/home/saadashraf/final-year-project/training_logs/*.csv")
    if "aborted" not in c
]
if not csvs:
    print("no CSV found")
    sys.exit(0)
latest = max(csvs, key=os.path.getmtime)
print(f"csv: {latest}")

# Staleness check: compare VM wall-clock to latest CSV mtime
mtime = datetime.fromtimestamp(os.path.getmtime(latest), tz=timezone.utc)
now = datetime.now(timezone.utc)
gap_min = (now - mtime).total_seconds() / 60
flag = ""
if gap_min > 30:
    flag = "  ⚠️ STALE (>30 min since last write — trajectory may be stuck)"
elif gap_min > 10:
    flag = "  ⚠️ slow (>10 min since last write)"
print(f"vm_now (UTC):     {now.isoformat(timespec='seconds')}")
print(f"latest csv mtime: {mtime.isoformat(timespec='seconds')}  (gap = {gap_min:.1f} min){flag}")
print()

rows = list(csv.DictReader(open(latest)))


def to_f(x):
    try:
        return float(x) if x not in ("", "None", None) else None
    except Exception:
        return None


print(f"trajectories so far: {len(rows)}")
fails = sum(1 for r in rows if r.get("skip_reason") == "trajectory_failed")
opt = sum(1 for r in rows if r["used_for_optimization"].lower() == "true")
print(f"  trajectory_failed: {fails}")
print(f"  used_for_optimization: {opt}")

non_fail = [r for r in rows if r.get("skip_reason") != "trajectory_failed"]
n = len(non_fail)
if n < 4:
    print("too few non-failed trajectories yet")
else:
    chunks = 4
    cs = n // chunks
    print(f"\n{chunks}-quartile breakdown over {n} non-failed trajectories ({cs}/each):")
    print(f"  {'quartile':>10} {'n':>4} {'reward':>8} {'success':>8} {'effic':>8} {'sc':>8} {'steps':>6}")
    for q in range(chunks):
        a = q * cs
        b = (q + 1) * cs if q < chunks - 1 else n
        chunk = non_fail[a:b]

        def col(key):
            v = [to_f(r[key]) for r in chunk]
            v = [x for x in v if x is not None]
            return st.mean(v) if v else 0.0

        steps = [int(r["num_steps"]) for r in chunk if r["num_steps"]]
        ms = st.mean(steps) if steps else 0.0
        print(
            f"  Q{q+1:<9} {len(chunk):>4} {col('reward'):>8.3f} {col('judge_success'):>8.3f} "
            f"{col('judge_efficiency'):>8.3f} {col('judge_self_correction'):>8.3f} {ms:>6.1f}"
        )

print(f"\nlast 5 trajectories:")
for r in rows[-5:]:
    rid = r["trajectory_id"]
    ds = r["dataset_index"]
    rwd = (r["reward"] or "-")[:6]
    suc = (r["judge_success"] or "-")[:6]
    op = "Y" if r["used_for_optimization"].lower() == "true" else "N"
    sk = r["skip_reason"] or "-"
    site = r["website"][:30]
    print(f"  {rid:>3} idx={ds:>3} reward={rwd:>6} succ={suc:>6} opt={op} {sk:<18} {site}")
