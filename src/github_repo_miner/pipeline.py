"""Discover GitHub repos that contain SpecStory session logs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import GITHUB_TOKEN, RATE_LIMIT_DELAY
from .github_api import check_rate_limit_status, get_recent_commits, search_code, search_repos
from .queries import CODE_SEARCH_QUERIES, REPO_DETECTION_QUERIES


@dataclass
class PipelineResult:
    """Outputs produced by a single SpecStory discovery run."""

    raw: pd.DataFrame
    deduplicated: pd.DataFrame
    filtered: pd.DataFrame
    query_summary: pd.DataFrame
    filtered_csv: Path | None = None
    raw_csv: Path | None = None


def _deduplicate_and_score(
    all_repos: list[dict],
    *,
    relaxed_filters: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the raw, deduplicated, and filtered result frames."""
    df_raw = pd.DataFrame(all_repos)
    print(f"Raw rows (with duplicates): {len(df_raw)}")

    if df_raw.empty:
        empty = pd.DataFrame()
        return df_raw, empty, empty

    signal_agg = (
        df_raw.groupby("full_name")
        .agg(
            detected_by=("detection_query", lambda values: ", ".join(sorted(set(values)))),
            discovery_sources=("discovery_source", lambda values: ", ".join(sorted(set(values)))),
            matched_paths=("matched_path", lambda values: ", ".join(sorted({value for value in values if value}))),
            matched_files=("matched_file", lambda values: ", ".join(sorted({value for value in values if value}))),
            match_urls=("match_url", lambda values: ", ".join(sorted({value for value in values if value}))),
        )
        .reset_index()
    )
    signal_agg["signal_count"] = signal_agg["detected_by"].str.split(", ").str.len()
    signal_agg["source_count"] = signal_agg["discovery_sources"].str.split(", ").str.len()

    df = (
        df_raw.sort_values("stars", ascending=False)
        .drop_duplicates(subset="full_name")
        .merge(signal_agg, on="full_name")
        .drop(columns=["detection_query"])
    )

    print(f"After dedup: {len(df)} unique repos")

    if relaxed_filters:
        df_filtered = df.copy()
        print("Using relaxed filters (maximize SpecStory repo coverage)")
    else:
        df_filtered = df[
            (df["stars"] >= 5)
            & (df["age_months"] >= 3)
            & (df["size_kb"] >= 50)
            & (df["language"].notna())
        ].copy()

    df_filtered = df_filtered.sort_values(
        ["signal_count", "stars"], ascending=[False, False]
    ).reset_index(drop=True)

    print(f"Candidate repos: {len(df_filtered)}")
    if not df_filtered.empty:
        print("\nLanguage breakdown:")
        print(df_filtered["language"].value_counts().head(10).to_string())
    else:
        print("\nLanguage breakdown: no repositories found")

    return df_raw, df, df_filtered


def _print_summary(df_filtered: pd.DataFrame, query_summary: pd.DataFrame) -> None:
    print("=" * 62)
    print("   SpecStory Repo Discovery")
    print("=" * 62)
    print(f"   Candidate repos                 : {len(df_filtered)}")
    print(f"   Multi-signal repos (2+ signals) : {len(df_filtered[df_filtered.signal_count > 1]) if not df_filtered.empty else 0}")
    print(f"   Repository queries run          : {len(REPO_DETECTION_QUERIES)}")
    print(f"   Code queries run                : {len(CODE_SEARCH_QUERIES) if GITHUB_TOKEN else 0}")
    print("=" * 62)

    if not query_summary.empty:
        for source_name in ["repo_search", "code_search"]:
            source_summary = query_summary[query_summary["source"] == source_name].copy()
            if source_summary.empty:
                continue
            print(f"\nResults per {source_name.replace('_', ' ')}:")
            summary_df = source_summary[["signal", "fetched", "total_on_github"]].copy()
            summary_df.columns = ["Signal", "Fetched", "Total on GitHub"]
            print(summary_df.to_string(index=False))

    if df_filtered.empty:
        print("\nTop candidate repos: none found.")
        return

    print("\nTop 15 candidate repos:")
    cols = ["full_name", "language", "stars", "signal_count", "detected_by"]
    print(df_filtered[cols].head(15).to_string(index=False))


def _spot_check_top_repo(df_filtered: pd.DataFrame) -> None:
    if df_filtered.empty:
        return

    top = df_filtered.iloc[0]
    print(f"\nSpot-check: {top['full_name']}")
    print(f"URL       : {top['url']}")
    print(f"Detected  : {top['detected_by']}\n")

    commits = get_recent_commits(top["full_name"])
    if commits:
        print(pd.DataFrame(commits).to_string(index=False))
    else:
        print("Could not fetch commits.")


def _save_outputs(
    df_filtered: pd.DataFrame,
    df: pd.DataFrame,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M")
    filtered_file = output_path / f"specstory_candidates_{ts}.csv"
    raw_file = output_path / f"specstory_candidates_raw_{ts}.csv"

    df_filtered.to_csv(filtered_file, index=False, encoding="utf-8-sig")
    df.to_csv(raw_file, index=False, encoding="utf-8-sig")
    print(f"Candidate repos : {filtered_file.name}  ({len(df_filtered)} repos)")
    print(f"Raw discovery   : {raw_file.name}  ({len(df)} repos)")

    try:
        from google.colab import files  # type: ignore

        files.download(str(filtered_file))
        print("Download triggered")
    except ImportError:
        pass

    return filtered_file, raw_file


def run_pipeline(
    output_dir: str | Path = "outputs",
    rate_limit_delay: float = RATE_LIMIT_DELAY,
    max_results_per_query: int = 1000,
    relaxed_filters: bool = True,
) -> PipelineResult:
    """Discover public repos that contain SpecStory session logs."""
    remaining, reset = check_rate_limit_status()
    api_mode = "authenticated" if GITHUB_TOKEN else "unauthenticated"
    print(f"\nGitHub API ready ({api_mode})")
    total_query_count = len(REPO_DETECTION_QUERIES) + (len(CODE_SEARCH_QUERIES) if GITHUB_TOKEN else 0)
    print(f"Estimated runtime: ~{total_query_count * rate_limit_delay / 60:.1f} minutes")
    if remaining <= 0:
        print(f"Search quota exhausted; reset timestamp: {reset}")
    print(f"Max results per query: {max_results_per_query}")

    if not GITHUB_TOKEN:
        print("Set GITHUB_TOKEN or GH_TOKEN to enable code search (strongly recommended).")

    all_repos: list[dict] = []
    query_summary_rows: list[dict] = []

    discovery_sources = [
        ("repo_search", REPO_DETECTION_QUERIES, search_repos),
        ("code_search", CODE_SEARCH_QUERIES, search_code),
    ]

    if not GITHUB_TOKEN:
        discovery_sources = discovery_sources[:1]

    running_index = 0
    for source_name, source_queries, search_fn in discovery_sources:
        for signal_name, query in source_queries:
            running_index += 1
            print(f"[{running_index:02d}/{total_query_count}] {signal_name}")

            repos, total = search_fn(query, max_results=max_results_per_query)
            all_repos.extend(repos)
            query_summary_rows.append(
                {
                    "source": source_name,
                    "signal": signal_name,
                    "fetched": len(repos),
                    "total_on_github": total,
                }
            )
            print(f"        fetched {len(repos)}  |  total on GitHub: {total:,}")

            if running_index < total_query_count:
                import time

                time.sleep(rate_limit_delay)

    print(f"\nDiscovery complete: {len(all_repos)} raw hits")

    df_raw, df, df_filtered = _deduplicate_and_score(all_repos, relaxed_filters=relaxed_filters)
    query_summary = pd.DataFrame(query_summary_rows)
    _print_summary(df_filtered, query_summary)
    _spot_check_top_repo(df_filtered)
    filtered_csv, raw_csv = _save_outputs(df_filtered, df, output_dir)

    return PipelineResult(
        raw=df_raw,
        deduplicated=df,
        filtered=df_filtered,
        query_summary=query_summary,
        filtered_csv=filtered_csv,
        raw_csv=raw_csv,
    )
