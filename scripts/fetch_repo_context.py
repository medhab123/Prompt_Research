"""Fetch repository-level metadata (language, stars, size, age, ...) for every
repo in a prompt CSV. One GitHub API call per repo, cached to the output CSV.

    python scripts/fetch_repo_context.py outputs/full_run_v3_analysis/specstory_prompts_cleaned.csv \
        --output outputs/full_run_v3_analysis/repo_context.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from github_repo_miner.repo_context import fetch_repo_context


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch repo-level metadata for every repo in a prompt CSV.")
    parser.add_argument("input_csv", type=Path, help="CSV with a full_name column (e.g. specstory_prompts_cleaned.csv).")
    parser.add_argument("--output", type=Path, required=True, help="Where to write (and cache) repo_context.csv.")
    args = parser.parse_args()

    names = pd.read_csv(args.input_csv, usecols=["full_name"], low_memory=False)["full_name"].dropna().tolist()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df = fetch_repo_context(names, args.output)
    print(f"Wrote {args.output} ({len(df)} repos)")


if __name__ == "__main__":
    main()
