"""Module entry point for `python -m github_repo_miner`."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_pipeline
from .prompt_extraction import extract_prompts_from_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine GitHub repositories that show LLM-assisted development signals."
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for generated CSV files (default: outputs).",
    )
    parser.add_argument(
        "--rate-limit-delay",
        type=float,
        default=7.5,
        help="Delay between GitHub search requests in seconds.",
    )
    parser.add_argument(
        "--max-results-per-query",
        type=int,
        default=1000,
        help="Maximum GitHub search results to fetch per query (capped at 1000).",
    )
    parser.add_argument(
        "--relaxed-filters",
        action="store_true",
        help="Skip strict repo quality filters (useful for SpecStory discovery).",
    )
    parser.add_argument(
        "--extract-prompts",
        action="store_true",
        help="Extract prompt-like text from a mined candidate CSV instead of running discovery.",
    )
    parser.add_argument(
        "--candidate-csv",
        type=Path,
        help="Candidate dataset to extract prompts from.",
    )
    parser.add_argument(
        "--prompt-output-dir",
        default="outputs",
        help="Directory for extracted prompt CSV files.",
    )
    parser.add_argument(
        "--max-prompt-repos",
        type=int,
        help="Limit the number of repositories processed during prompt extraction.",
    )
    parser.add_argument(
        "--max-priority",
        type=int,
        choices=[1, 2, 3, 4],
        help=(
            "Keep only higher-value artifact types: "
            "1=session_prompt, 2=+inline_prompt_comment, 3=+instruction_file, 4=all."
        ),
    )
    parser.add_argument(
        "--artifact-types",
        help=(
            "Comma-separated artifact types to keep "
            "(session_prompt, inline_prompt_comment, instruction_file, rules_config)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.extract_prompts:
        candidate_csv = args.candidate_csv
        if candidate_csv is None:
            outputs = sorted(Path(args.prompt_output_dir).glob("llm_repos_candidate_dataset_*.csv"))
            if not outputs:
                raise SystemExit("No candidate CSV found. Pass --candidate-csv explicitly.")
            candidate_csv = outputs[-1]
        artifact_types = None
        if args.artifact_types:
            artifact_types = [value.strip() for value in args.artifact_types.split(",") if value.strip()]
        extract_prompts_from_candidates(
            candidate_csv,
            output_dir=args.prompt_output_dir,
            max_repos=args.max_prompt_repos,
            max_priority=args.max_priority,
            artifact_types=artifact_types,
        )
        return
    run_pipeline(
        output_dir=args.output_dir,
        rate_limit_delay=args.rate_limit_delay,
        max_results_per_query=args.max_results_per_query,
        relaxed_filters=args.relaxed_filters,
    )


if __name__ == "__main__":
    main()
