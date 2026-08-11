"""Predict action_type (implement/debug/explain/review/tool_only/other) from
developer prompts, with repo-held-out evaluation and a prompt-only vs.
prior-turn-history comparison. This targets the actual research spec's label,
unlike run_ml_research.py which still targets the older primary_intent label.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from github_repo_miner.ml_research import ActionPredictionConfig, run_action_prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict action_type from developer prompts (repo-held-out evaluation)."
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        nargs="?",
        default=Path("outputs/datasets/specstory_prompts_clustered.csv"),
        help="Clustered or cleaned prompt CSV containing an action_type column.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/ml_action"),
        help="Directory for artifacts (default: outputs/ml_action).",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        help="Force cpu or cuda for the embedding model.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of repos held out for testing (default: 0.2).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_csv.exists():
        raise SystemExit(f"Input CSV not found: {args.input_csv}")

    run_action_prediction(
        ActionPredictionConfig(
            input_csv=args.input_csv.resolve(),
            output_dir=args.output_dir.resolve(),
            device=args.device,
            test_size=args.test_size,
        )
    )


if __name__ == "__main__":
    main()
