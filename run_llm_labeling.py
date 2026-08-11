"""LLM-assisted action_type labeling (Gemini or Groq free tier).

Default mode labels a stratified sample first (recommended: validate kappa
against rule-based labels and do a human-review pass before committing the
time/quota to labeling the full corpus). Pass --full to label everything.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from github_repo_miner.ml_research.llm_labeling import (
    ACTION_TYPES,
    build_human_review_sheet,
    cohen_kappa,
    label_dataframe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-assisted action_type labeling.")
    parser.add_argument("input_csv", type=Path, help="Dataset CSV with prompt_text/agent_response/action_type.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/llm_labels"))
    parser.add_argument("--provider", choices=("gemini", "groq"), default=None, help="Default: auto-detect from .env")
    parser.add_argument("--model", default=None)
    parser.add_argument("--sample-size", type=int, default=300, help="Stratified sample size (ignored with --full).")
    parser.add_argument("--full", action="store_true", help="Label the entire dataset instead of a sample.")
    parser.add_argument("--rate-limit-delay", type=float, default=3.0, help="Seconds between API calls.")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    return parser.parse_args()


def _stratified_sample(df: pd.DataFrame, total: int, random_state: int = 42) -> pd.DataFrame:
    per_class = max(1, total // len(ACTION_TYPES))
    parts = []
    for action in ACTION_TYPES:
        pool = df[df["action_type"] == action]
        n = min(per_class, len(pool))
        if n:
            parts.append(pool.sample(n=n, random_state=random_state))
    return pd.concat(parts, ignore_index=True) if parts else df.head(0)


def main() -> None:
    args = parse_args()
    if not args.input_csv.exists():
        raise SystemExit(f"Input CSV not found: {args.input_csv}")

    df = pd.read_csv(args.input_csv, low_memory=False)
    df = df[df["action_type"].notna()].reset_index(drop=True)

    if args.full:
        target_df = df
        print(f"Labeling FULL dataset: {len(target_df)} rows")
    else:
        target_df = _stratified_sample(df, args.sample_size)
        print(f"Labeling stratified sample: {len(target_df)} rows ({args.sample_size} requested)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / "llm_label_cache.json"
    checkpoint_path = args.output_dir / "llm_labels_checkpoint.csv"

    labeled = label_dataframe(
        target_df,
        provider=args.provider,
        model=args.model,
        cache_path=cache_path,
        checkpoint_path=checkpoint_path,
        checkpoint_every=args.checkpoint_every,
        rate_limit_delay=args.rate_limit_delay,
    )

    suffix = "full" if args.full else "sample"
    out_path = args.output_dir / f"llm_labels_{suffix}.csv"
    labeled.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {out_path}")

    kappa = cohen_kappa(labeled["action_type"], labeled["llm_action_type"])
    print(f"\nCohen's kappa (LLM vs. rule-based action_type): {kappa:.3f}")
    print("\nAgreement by class:")
    for action in ACTION_TYPES:
        mask = labeled["action_type"] == action
        if mask.sum() == 0:
            continue
        agree = (labeled.loc[mask, "llm_action_type"] == action).mean()
        print(f"  {action:<10} n={int(mask.sum()):<5} LLM-agrees-with-rule={agree:.1%}")

    review_sheet = build_human_review_sheet(labeled, n_per_class=min(40, len(labeled) // len(ACTION_TYPES) or 1))
    review_path = args.output_dir / f"human_review_sheet_{suffix}.csv"
    review_sheet.to_csv(review_path, index=False, encoding="utf-8-sig")
    print(f"\nHuman review sheet ({len(review_sheet)} rows, fill in 'human_label'): {review_path}")


if __name__ == "__main__":
    main()
