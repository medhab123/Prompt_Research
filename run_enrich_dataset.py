"""Enrich an existing turn-pair CSV with action labels and session features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from github_repo_miner.action_labeling import action_label_summary
from github_repo_miner.session_features import enrich_turn_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add action_type labels and session context to a turn-pair CSV."
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        nargs="?",
        default=Path("outputs/datasets/specstory_prompts_clustered.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV path (default: input stem + _enriched.csv in same dir).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_csv.exists():
        raise SystemExit(f"Input CSV not found: {args.input_csv}")

    df = pd.read_csv(args.input_csv)
    required = {"prompt_text", "full_name", "source_file"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Input CSV missing columns: {sorted(missing)}")

    enriched = enrich_turn_dataset(df)
    summary = action_label_summary(enriched)

    out = args.output
    if out is None:
        out = args.input_csv.with_name(f"{args.input_csv.stem}_enriched.csv")

    out.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"Rows: {len(enriched)}")
    print(f"Action label distribution:")
    for action, pct in summary["pct"].items():
        print(f"  {action}: {pct}% ({summary['counts'][action]})")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
