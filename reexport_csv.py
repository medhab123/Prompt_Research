"""Re-export an existing prompt CSV with spreadsheet-safe serialization."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from github_repo_miner.csv_export import export_prompts_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-export a prompt CSV for spreadsheet tools.")
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    stem = args.input_csv.stem
    full = export_prompts_csv(df, out / f"{stem}_spreadsheet.csv")
    prompts_only = export_prompts_csv(df, out / f"{stem}_prompts_only.csv", prompts_only=True)
    print("Wrote:", full)
    print("Wrote:", prompts_only)


if __name__ == "__main__":
    main()
