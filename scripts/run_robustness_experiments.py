"""Run robustness baselines and keyword-ablation experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from github_repo_miner.ml_research import MLResearchConfig
from github_repo_miner.ml_research.robustness import run_robustness_experiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Robustness experiments: baselines, keyword ablation, label audit."
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        nargs="?",
        default=Path("outputs/datasets/specstory_prompts_clustered.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/ml"),
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        help="Force cpu or cuda for SBERT encoding.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_csv.exists():
        raise SystemExit(f"Input CSV not found: {args.input_csv}")

    report = run_robustness_experiments(
        MLResearchConfig(
            input_csv=args.input_csv.resolve(),
            output_dir=args.output_dir.resolve(),
            device=args.device,
        )
    )
    print(f"Robustness report written to: {report}")


if __name__ == "__main__":
    main()
