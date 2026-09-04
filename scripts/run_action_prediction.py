"""Predict action_type (implement/debug/explain/review/tool_only/other) from
developer prompts, with repo-held-out evaluation and a prompt-only vs.
prior-turn-history comparison. This targets the actual research spec's label,
unlike run_ml_research.py which still targets the older primary_intent label.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
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
    parser.add_argument(
        "--label-col",
        default=None,
        help="Ground-truth column to target. Defaults to llm_action_type when present "
        "(human-validated as the better label, kappa 0.760 vs 0.614), else action_type.",
    )
    parser.add_argument(
        "--repo-context-csv",
        type=Path,
        default=None,
        help="Output of repo_context.py (full_name, repo_language, repo_stars, ...). "
        "Adds the 'repository context' input the spec promises but the corpus lacks by default.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Repository-held-out CV folds for statistical rigor (default: 5).",
    )
    parser.add_argument(
        "--max-context-turns",
        type=int,
        default=3,
        help="Max N for the prior-turn-text context-window sweep (default: 3).",
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
            label_col=args.label_col,
            repo_context_csv=args.repo_context_csv.resolve() if args.repo_context_csv else None,
            n_cv_folds=args.cv_folds,
            max_context_turns=args.max_context_turns,
        )
    )


if __name__ == "__main__":
    main()
