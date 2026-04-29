"""
Figure 1: InSTA benchmark dataset decay.
Two panels:
  (A) Feasibility-class breakdown over 2,598 tasks judged by Gemini.
  (B) HTTP-probe outcome within tasks the LLM judge classified WEBSITE_DOWN
      (showing that a reachability-only filter would miss ~24% of broken sites).

Reads parsed per-task data from v2_log_parsed.csv. Writes PNG + PDF.
"""

import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
SRC = HERE / "v2_log_parsed.csv"
OUT_PNG = HERE / "figure1_dataset_decay.png"
OUT_PDF = HERE / "figure1_dataset_decay.pdf"

with SRC.open() as f:
    rows = list(csv.DictReader(f))
N = len(rows)

CLASS_ORDER = ["FEASIBLE", "WEBSITE_DOWN", "CONTENT_LIKELY_OUTDATED", "UNCERTAIN"]
CLASS_LABEL = {
    "FEASIBLE": "Feasible",
    "WEBSITE_DOWN": "Website down",
    "CONTENT_LIKELY_OUTDATED": "Content outdated",
    "UNCERTAIN": "Uncertain",
}
CLASS_COLOR = {
    "FEASIBLE": "#2a9d8f",
    "WEBSITE_DOWN": "#e63946",
    "CONTENT_LIKELY_OUTDATED": "#f4a261",
    "UNCERTAIN": "#9aa0a6",
}

class_counts = Counter(r["classification"] for r in rows)


def http_bucket(status: str) -> str:
    if status in ("", "None"):
        return "unreachable"
    if not status.isdigit():
        return "other"
    s = int(status)
    if 200 <= s < 300:
        return "2xx"
    if 300 <= s < 400:
        return "3xx"
    if 400 <= s < 500:
        return "4xx"
    return "5xx"


down_rows = [r for r in rows if r["classification"] == "WEBSITE_DOWN"]
http_buckets_in_down = Counter(http_bucket(r["http_status"]) for r in down_rows)
n_down = len(down_rows)

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})

# Panel A — classification breakdown
labels = [CLASS_LABEL[c] for c in CLASS_ORDER]
counts = [class_counts[c] for c in CLASS_ORDER]
colors = [CLASS_COLOR[c] for c in CLASS_ORDER]
pcts = [100 * c / N for c in counts]

bars = axL.bar(labels, counts, color=colors, edgecolor="black", linewidth=0.5)
for bar, count, pct in zip(bars, counts, pcts):
    axL.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + N * 0.012,
        f"{count}\n({pct:.1f}%)",
        ha="center",
        va="bottom",
        fontsize=9,
    )
axL.set_ylabel("Number of tasks")
axL.set_title(f"(A) Feasibility classification of {N:,} InSTA tasks", fontsize=11)
axL.set_ylim(0, max(counts) * 1.18)
axL.spines["top"].set_visible(False)
axL.spines["right"].set_visible(False)
axL.tick_params(axis="x", labelsize=9)

# Panel B — HTTP probe outcomes within WEBSITE_DOWN
order = ["2xx", "3xx", "4xx", "5xx", "unreachable"]
display = {"2xx": "HTTP 2xx", "3xx": "HTTP 3xx", "4xx": "HTTP 4xx", "5xx": "HTTP 5xx", "unreachable": "Unreachable"}
b_counts = [http_buckets_in_down.get(k, 0) for k in order]
b_labels = [display[k] for k in order]
b_pcts = [100 * c / n_down for c in b_counts]
# Highlight the 2xx bar (the disagreement)
b_colors = ["#e63946" if k == "2xx" else "#9aa0a6" for k in order]

bars2 = axR.bar(b_labels, b_counts, color=b_colors, edgecolor="black", linewidth=0.5)
for bar, count, pct in zip(bars2, b_counts, b_pcts):
    axR.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + n_down * 0.015,
        f"{count}\n({pct:.1f}%)",
        ha="center",
        va="bottom",
        fontsize=9,
    )
axR.set_ylabel("Number of tasks")
axR.set_title(f"(B) HTTP probe within {n_down} WEBSITE_DOWN tasks", fontsize=11)
axR.set_ylim(0, max(b_counts) * 1.22)
axR.spines["top"].set_visible(False)
axR.spines["right"].set_visible(False)
axR.tick_params(axis="x", labelsize=9, rotation=15)

# Annotate the disagreement on Panel B
n_2xx = http_buckets_in_down.get("2xx", 0)
axR.annotate(
    f"{n_2xx/n_down:.0%} of broken\nsites returned 2xx\n(missed by\nreachability filter)",
    xy=(0, n_2xx),
    xytext=(0.55, max(b_counts) * 0.85),
    fontsize=8.5,
    ha="left",
    arrowprops=dict(arrowstyle="->", color="#e63946", lw=1.0),
    color="#7a1721",
)

fig.suptitle("Dataset decay in the InSTA web-agent benchmark", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
fig.savefig(OUT_PDF, bbox_inches="tight")
print(f"wrote {OUT_PNG}")
print(f"wrote {OUT_PDF}")
