"""Run local prompt analysis on an existing SpecStory CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from github_repo_miner.data_cleaning import CleaningConfig
from github_repo_miner.prompt_analysis import AnalysisConfig, run_prompt_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean, embed, cluster, and analyze a SpecStory prompt CSV locally."
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="Path to specstory_prompts_*.csv or specstory_prompts_clustered*.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for cleaned/clustered outputs (default: outputs).",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        help="Force cpu or cuda. Default: auto-detect.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=25,
        help="Minimum prompt length for non-CJK text (default: 25).",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=4,
        help="Minimum word count for non-CJK text (default: 4).",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=8_000,
        help="Drop prompts longer than this (default: 8000).",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display matplotlib figures instead of only saving PNGs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_csv.exists():
        raise SystemExit(f"Input CSV not found: {args.input_csv}")

    run_prompt_analysis(
        AnalysisConfig(
            input_csv=args.input_csv.resolve(),
            output_dir=args.output_dir.resolve(),
            cleaning=CleaningConfig(
                min_chars=args.min_chars,
                min_words=args.min_words,
                max_chars=args.max_chars,
            ),
            device=args.device,
            show_plots=args.show_plots,
        )
    )


if __name__ == "__main__":
    main()
