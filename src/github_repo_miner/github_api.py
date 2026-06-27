"""GitHub API helpers for searching repositories and commits."""

from __future__ import annotations

import time
from datetime import datetime
from functools import lru_cache

import requests

from .config import BASE_URL, GITHUB_TOKEN, HEADERS


def check_rate_limit_status() -> tuple[int, int]:
    """Return the remaining search quota and reset timestamp."""
    response = requests.get(f"{BASE_URL}/rate_limit", headers=HEADERS)
    if response.status_code == 200:
        data = response.json()
        search = data["resources"]["search"]
        core = data["resources"]["core"]
        print(f"Search API : {search['remaining']}/{search['limit']} remaining")
        print(f"Core API   : {core['remaining']}/{core['limit']} remaining")
        return search["remaining"], search["reset"]

    print(f"Could not check rate limit ({response.status_code}) — continuing anyway")
    return 10, int(time.time()) + 60


GITHUB_SEARCH_RESULT_CAP = 1000


def _search_page(
    endpoint: str,
    query: str,
    *,
    page: int,
    per_page: int,
) -> dict | None:
    """Run one GitHub search page with simple retry logic."""
    params = {
        "q": query,
        "sort": "updated",
        "order": "desc",
        "per_page": per_page,
        "page": page,
    }

    response = None
    for attempt in range(3):
        response = requests.get(
            f"{BASE_URL}/{endpoint}",
            headers=HEADERS,
            params=params,
        )
        if response.status_code == 200:
            return response.json()
        if response.status_code in (403, 429):
            reset_ts = int(response.headers.get("X-RateLimit-Reset", time.time() + 65))
            wait = max(reset_ts - time.time() + 2, 15)
            print(f"   ⏳ Rate limited. Waiting {int(wait)}s (attempt {attempt + 1}/3)...")
            time.sleep(wait)
            continue

        print(f"   ❌ Error {response.status_code}: {response.text[:120]}")
        return None

    print("   ❌ Failed after 3 attempts — skipping this page")
    return None


def _search_paginated(
    endpoint: str,
    query: str,
    max_results: int = 1000,
    *,
    page_delay: float = 2.0,
) -> tuple[list[dict], int]:
    """Fetch multiple search pages up to GitHub's 1,000-result cap."""
    per_page = 100
    max_results = min(max_results, GITHUB_SEARCH_RESULT_CAP)
    all_items: list[dict] = []
    total_count = 0
    page = 1

    while len(all_items) < max_results:
        payload = _search_page(endpoint, query, page=page, per_page=per_page)
        if not payload:
            break

        if page == 1:
            total_count = int(payload.get("total_count", 0))

        items = payload.get("items", [])
        if not items:
            break

        remaining = max_results - len(all_items)
        all_items.extend(items[:remaining])

        if len(items) < per_page or len(all_items) >= max_results:
            break

        page += 1
        if page * per_page > GITHUB_SEARCH_RESULT_CAP:
            break
        time.sleep(page_delay)

    return all_items, total_count


def _repo_record(
    repo: dict,
    query: str,
    discovery_source: str,
    match_url: str | None = None,
    matched_path: str = "",
    matched_file: str = "",
) -> dict:
    """Normalize repo metadata across repo search and code search results."""
    created_at = repo.get("created_at") or repo.get("pushed_at") or repo.get("updated_at")
    if created_at:
        created = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")
        age_months = (datetime.utcnow() - created).days / 30
    else:
        age_months = 0.0
    repo_url = repo.get("html_url") or f"https://github.com/{repo['full_name']}"
    return {
        "full_name": repo["full_name"],
        "url": repo_url,
        "match_url": match_url or repo_url,
        "description": (repo.get("description") or "")[:120],
        "language": repo.get("language"),
        "stars": repo["stargazers_count"],
        "forks": repo["forks_count"],
        "open_issues": repo["open_issues_count"],
        "size_kb": repo["size"],
        "created_at": repo["created_at"],
        "updated_at": repo["updated_at"],
        "age_months": round(age_months, 1),
        "topics": ", ".join(repo.get("topics", [])),
        "detection_query": query,
        "discovery_source": discovery_source,
        "matched_path": matched_path,
        "matched_file": matched_file,
    }


@lru_cache(maxsize=1024)
def _fetch_repo_details(full_name: str) -> dict:
    """Fetch complete repository metadata for code-search hits."""
    response = requests.get(f"{BASE_URL}/repos/{full_name}", headers=HEADERS)
    if response.status_code == 200:
        return response.json()
    return {"full_name": full_name, "html_url": f"https://github.com/{full_name}"}


def search_repos(query: str, max_results: int = 1000) -> tuple[list[dict], int]:
    """
    Search GitHub repos and return (list_of_repo_dicts, total_count_on_github).

    Paginates through results up to GitHub's 1,000-result search cap.
    """
    items, total_count = _search_paginated("search/repositories", query, max_results=max_results)
    repos: list[dict] = []

    for repo in items:
        repos.append(_repo_record(repo, query, discovery_source="repo_search"))

    return repos, total_count


def search_code(query: str, max_results: int = 1000) -> tuple[list[dict], int]:
    """Search GitHub code for exact prompt-artifact files.

    Code search requires authentication on this project. If no token is available,
    return an empty result set and let the repo-search path continue.
    """
    if not GITHUB_TOKEN:
        print(f"   ↪ skipping code search for '{query}' (set GITHUB_TOKEN or GH_TOKEN to enable)")
        return [], 0

    items, total_count = _search_paginated("search/code", query, max_results=max_results)
    repos: list[dict] = []
    for item in items:
        repository = item.get("repository")
        if not repository:
            continue
        repo_details = _fetch_repo_details(repository["full_name"])
        repos.append(
            _repo_record(
                repo_details,
                query,
                discovery_source="code_search",
                match_url=item.get("html_url"),
                matched_path=item.get("path", ""),
                matched_file=item.get("name", ""),
            )
        )

    return repos, total_count


def get_recent_commits(full_name: str, n: int = 15) -> list[dict]:
    """Fetch recent commits and flag messages that mention LLM-related keywords."""
    from .config import LLM_KEYWORDS

    url = f"{BASE_URL}/repos/{full_name}/commits"
    response = requests.get(url, headers=HEADERS, params={"per_page": n})
    if response.status_code != 200:
        print(f"Error {response.status_code}")
        return []

    commits = []
    for commit in response.json():
        message = commit["commit"]["message"]
        flagged = any(keyword in message.lower() for keyword in LLM_KEYWORDS)
        commits.append(
            {
                "sha": commit["sha"][:7],
                "date": commit["commit"]["author"]["date"][:10],
                "message": message.split("\n")[0][:75],
                "llm_signal": "✅" if flagged else "",
            }
        )
    return commits
