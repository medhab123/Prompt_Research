"""Module entry point for `python -m github_repo_miner`."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_pipeline
from .prompt_analysis import AnalysisConfig, run_prompt_analysis
from .prompt_extraction import extract_specstory_prompts_from_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover GitHub repos with SpecStory logs and extract user prompts."
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
        "--strict-filters",
        action="store_true",
        help="Apply star/age/size filters (default: relaxed filters for max coverage).",
    )
    parser.add_argument(
        "--discover-only",
        action="store_true",
        help="Only discover SpecStory repos; skip prompt extraction.",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Only extract SpecStory prompts from an existing candidate CSV.",
    )
    parser.add_argument(
        "--candidate-csv",
        type=Path,
        help="Candidate CSV for --extract-only (defaults to newest specstory_candidates_*.csv).",
    )
    parser.add_argument(
        "--max-repos",
        type=int,
        help="Limit repositories processed during prompt extraction.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Concurrent repos to extract at once during prompt extraction (default: 8).",
    )
    parser.add_argument(
        "--analyze-csv",
        type=Path,
        help="Analyze an existing SpecStory prompt CSV (clean, embed, cluster, report).",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        help="Device for embedding when using --analyze-csv.",
    )
    return parser.parse_args()


def _latest_candidate_csv(output_dir: str | Path) -> Path:
    outputs = sorted(Path(output_dir).glob("specstory_candidates_*.csv"))
    if not outputs:
        legacy = sorted(Path(output_dir).glob("llm_repos_candidate_dataset_*.csv"))
        if legacy:
            return legacy[-1]
        raise SystemExit(
            "No candidate CSV found. Run discovery first or pass --candidate-csv."
        )
    return outputs[-1]


def main() -> None:
    args = parse_args()

    if args.analyze_csv:
        if not args.analyze_csv.exists():
            raise SystemExit(f"Input CSV not found: {args.analyze_csv}")
        run_prompt_analysis(
            AnalysisConfig(
                input_csv=args.analyze_csv.resolve(),
                output_dir=Path(args.output_dir).resolve(),
                device=args.device,
            )
        )
        return

    if args.extract_only:
        candidate_csv = args.candidate_csv or _latest_candidate_csv(args.output_dir)
        extract_specstory_prompts_from_candidates(
            candidate_csv,
            output_dir=args.output_dir,
            max_repos=args.max_repos,
            max_workers=args.max_workers,
        )
        return

    result = run_pipeline(
        output_dir=args.output_dir,
        rate_limit_delay=args.rate_limit_delay,
        max_results_per_query=args.max_results_per_query,
        relaxed_filters=not args.strict_filters,
    )

    if args.discover_only or result.filtered_csv is None or result.filtered.empty:
        return

    print("\n" + "=" * 62)
    print("   Extracting SpecStory prompts from discovered repos")
    print("=" * 62)
    extract_specstory_prompts_from_candidates(
        result.filtered_csv,
        output_dir=args.output_dir,
        max_repos=args.max_repos,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
