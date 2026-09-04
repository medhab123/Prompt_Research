"""Run the supervised ML research pipeline on a SpecStory prompt dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from github_repo_miner.ml_research import MLResearchConfig, run_ml_research


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Supervised ML research on developer-LLM interaction prompts."
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        nargs="?",
        default=Path("outputs/datasets/specstory_prompts_clustered.csv"),
        help="Clustered or cleaned prompt CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/ml"),
        help="Directory for ML artifacts (default: outputs/ml).",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        help="Force cpu or cuda for embedding models.",
    )
    parser.add_argument(
        "--skip-code-embeddings",
        action="store_true",
        help="Skip code-aware embedding comparison (faster).",
    )
    parser.add_argument(
        "--min-intent-class-size",
        type=int,
        default=30,
        help="Drop intent classes with fewer than N training examples (default: 30).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_csv.exists():
        raise SystemExit(f"Input CSV not found: {args.input_csv}")

    run_ml_research(
        MLResearchConfig(
            input_csv=args.input_csv.resolve(),
            output_dir=args.output_dir.resolve(),
            device=args.device,
            skip_code_embeddings=args.skip_code_embeddings,
            min_intent_class_size=args.min_intent_class_size,
        )
    )


if __name__ == "__main__":
    main()
