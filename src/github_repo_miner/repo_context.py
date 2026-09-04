"""Fetch repository-level metadata for the "repository context" input the
research spec's Prediction Task section promises alongside prompt embeddings
and conversation history — currently absent from the mined corpus entirely.

One GitHub API call per unique repo (~251 in the current corpus), cached to
disk so this never needs to hit the API again once run. Independent of the
LLM labeling pass: this only needs `full_name`, which the corpus already has.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

from .config import BASE_URL, HEADERS

REPO_CONTEXT_COLS = [
    "repo_language",
    "repo_stars",
    "repo_forks",
    "repo_size_kb",
    "repo_open_issues",
    "repo_age_months",
    "repo_topics",
]


def _fetch_one(full_name: str) -> dict:
    from datetime import datetime

    response = requests.get(f"{BASE_URL}/repos/{full_name}", headers=HEADERS)
    if response.status_code != 200:
        return {"full_name": full_name, "_error": f"http_{response.status_code}"}

    repo = response.json()
    created_at = repo.get("created_at")
    age_months = 0.0
    if created_at:
        created = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")
        age_months = round((datetime.utcnow() - created).days / 30, 1)

    return {
        "full_name": full_name,
        "repo_language": repo.get("language") or "unknown",
        "repo_stars": repo.get("stargazers_count", 0),
        "repo_forks": repo.get("forks_count", 0),
        "repo_size_kb": repo.get("size", 0),
        "repo_open_issues": repo.get("open_issues_count", 0),
        "repo_age_months": age_months,
        "repo_topics": ", ".join(repo.get("topics", []) or []),
    }


def fetch_repo_context(
    full_names: list[str],
    cache_path: Path,
    *,
    rate_limit_delay: float = 0.3,
) -> pd.DataFrame:
    """Fetch (or load cached) repo-level metadata for every repo in full_names."""
    cached = pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame(columns=["full_name"])
    have = set(cached["full_name"]) if not cached.empty else set()

    to_fetch = [n for n in dict.fromkeys(full_names) if n not in have]
    print(f"Repo context: {len(have)} cached, {len(to_fetch)} to fetch")

    rows = []
    for i, name in enumerate(to_fetch, start=1):
        rows.append(_fetch_one(name))
        if i % 25 == 0 or i == len(to_fetch):
            print(f"  [{i}/{len(to_fetch)}] fetched")
        time.sleep(rate_limit_delay)

    if rows:
        fetched_df = pd.DataFrame(rows)
        cached = pd.concat([cached, fetched_df], ignore_index=True)
        cached.to_csv(cache_path, index=False, encoding="utf-8-sig")

    failed = cached[cached.get("_error").notna()] if "_error" in cached.columns else cached.iloc[0:0]
    if len(failed):
        print(f"  {len(failed)} repos failed to fetch (see _error column) — will retry on next run")

    return cached
