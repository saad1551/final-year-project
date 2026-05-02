"""
Download the InSTA-150k dataset from Hugging Face.

We removed the dataset CSVs from the repo (`insta-150k-train.csv` was
~102 MB, plus a ~104 MB local-format copy under `data/insta-150k-local/`).
Anyone reproducing the work runs this script once to fetch them back.

Usage:
    python data/download.py

Requirements:
    pip install datasets huggingface_hub

The script writes:
    data/insta-150k-train.csv
    data/insta-150k-test.csv (already in repo, but re-fetched for parity)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Download the InSTA-150k dataset")
    ap.add_argument("--repo", default="btrabucco/insta-150k",
                    help="HuggingFace dataset repo (default: btrabucco/insta-150k)")
    ap.add_argument("--dest", type=Path, default=Path(__file__).resolve().parent,
                    help="Output directory (default: data/)")
    ap.add_argument("--force", action="store_true",
                    help="Re-download even if files already exist")
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: `datasets` package not installed. Run:  pip install datasets", file=sys.stderr)
        sys.exit(1)

    args.dest.mkdir(parents=True, exist_ok=True)
    train_csv = args.dest / "insta-150k-train.csv"
    test_csv = args.dest / "insta-150k-test.csv"

    if train_csv.exists() and test_csv.exists() and not args.force:
        print(f"Both CSVs already present at {args.dest}. Use --force to re-download.")
        return

    print(f"Loading {args.repo} from Hugging Face...")
    ds = load_dataset(args.repo)

    if "train" in ds:
        print(f"Writing train split → {train_csv}")
        ds["train"].to_csv(str(train_csv), index=False)
    if "test" in ds:
        print(f"Writing test split → {test_csv}")
        ds["test"].to_csv(str(test_csv), index=False)

    # Print sizes so the user can sanity-check
    for p in (train_csv, test_csv):
        if p.exists():
            print(f"  {p.name}: {p.stat().st_size/1e6:.1f} MB")
    print("Done.")


if __name__ == "__main__":
    main()
