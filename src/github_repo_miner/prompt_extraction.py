"""Extract user prompts and agent turn pairs from SpecStory session logs."""

from __future__ import annotations

import base64
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd
import requests

from .action_labeling import action_label_summary
from .config import BASE_URL, GITHUB_TOKEN, HEADERS
from .csv_export import export_prompts_csv
from .data_cleaning import classify_noise
from .session_features import enrich_turn_dataset
from .specstory_parser import SpecStoryTurnPair, parse_specstory_turns

SPECSTORY_ROOT = ".specstory"
SPECSTORY_HISTORY = ".specstory/history"
SPECSTORY_PATH_FRAGMENT = ".specstory/history"


def _is_meaningful_prompt(text: str) -> bool:
    return classify_noise(text) is None


@dataclass
class SpecStoryTurnRecord:
    """One developer prompt paired with the following agent turn."""

    full_name: str
    repo_url: str
    source_file: str
    turn_index: int
    prompt_text: str
    agent_response: str = ""
    agent_model: str = ""
    generated_code: str = ""
    code_block_count: int = 0
    code_languages: str = ""
    has_generated_code: bool = False
    has_tool_use: bool = False


MAX_RESPONSE_BYTES = 25_000_000  # a real SpecStory session file has no business being this big


def _request_json(url: str, params: dict | None = None) -> dict | list | None:
    for attempt in range(3):
        try:
            response = requests.get(url, headers=HEADERS, params=params, timeout=30)
        except requests.RequestException:
            if attempt == 2:
                return None
            time.sleep(5)
            continue

        if response.status_code == 200:
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > MAX_RESPONSE_BYTES:
                print(f"   skipping oversized response ({int(content_length):,} bytes): {url}", flush=True)
                return None
            try:
                return response.json()
            except (ValueError, MemoryError) as exc:
                print(f"   skipping unparseable/oversized response ({exc}): {url}", flush=True)
                return None
        if response.status_code in (403, 429):
            reset_ts = int(response.headers.get("X-RateLimit-Reset", time.time() + 65))
            wait = max(reset_ts - time.time() + 2, 15)
            time.sleep(min(wait, 60))
            continue
        if response.status_code == 404:
            return None
        return None

    return None


@lru_cache(maxsize=1024)
def _fetch_repo(full_name: str) -> dict | None:
    data = _request_json(f"{BASE_URL}/repos/{full_name}")
    return data if isinstance(data, dict) else None


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_specstory_file(path: str) -> bool:
    lower = path.lower().replace("\\", "/")
    return SPECSTORY_PATH_FRAGMENT in lower or (
        SPECSTORY_ROOT in lower and lower.endswith(".md")
    )


def _walk_contents(full_name: str, path: str, depth: int = 0, max_depth: int = 5) -> list[dict]:
    url = f"{BASE_URL}/repos/{full_name}/contents/{path}".rstrip("/")
    payload = _request_json(url)
    if payload is None:
        return []

    if isinstance(payload, dict):
        return [payload] if payload.get("type") == "file" else []

    items: list[dict] = []
    for entry in payload:
        if entry.get("type") == "file":
            items.append(entry)
            continue
        if entry.get("type") == "dir" and depth < max_depth:
            items.extend(_walk_contents(full_name, entry["path"], depth + 1, max_depth))
    return items


def _search_repo_specstory_files(full_name: str) -> list[str]:
    """Fallback: code-search within a repo for any SpecStory markdown file."""
    if not GITHUB_TOKEN:
        return []

    paths: set[str] = set()
    for fragment in ("path:.specstory/history", "path:.specstory extension:md"):
        params = {
            "q": f"repo:{full_name} {fragment}",
            "per_page": 100,
            "page": 1,
        }
        payload = _request_json(f"{BASE_URL}/search/code", params=params)
        if not isinstance(payload, dict):
            continue
        for item in payload.get("items", []):
            path = item.get("path")
            if path and _is_specstory_file(path):
                paths.add(path)
    return sorted(paths)


def discover_specstory_files(full_name: str) -> list[str]:
    """Find every SpecStory history markdown file reachable in a repository."""
    files: dict[str, None] = {}

    for root in (SPECSTORY_HISTORY, SPECSTORY_ROOT):
        for entry in _walk_contents(full_name, root):
            path = entry.get("path") or ""
            if path and _is_specstory_file(path):
                files[path] = None

    # The directory walk (Contents API) is "core" quota (5000/hr) and finds
    # files in the standard location. Code search is scarce "search" quota
    # (30/min) — only spend it when the walk truly found nothing, not on
    # every repo regardless. At a few hundred repos, always-on search
    # fallback was the dominant bottleneck of a full extraction run.
    if not files:
        for path in _search_repo_specstory_files(full_name):
            files[path] = None

    return sorted(files)


def _decode_file_content(payload: dict) -> str:
    content = payload.get("content")
    if not content:
        return ""
    encoded = content.replace("\n", "")
    try:
        return base64.b64decode(encoded).decode("utf-8", errors="replace")
    except Exception:
        return ""


def fetch_file_text(full_name: str, path: str) -> str:
    payload = _request_json(f"{BASE_URL}/repos/{full_name}/contents/{path}")
    if not isinstance(payload, dict):
        return ""
    return _normalize_text(_decode_file_content(payload))


def _pair_to_record(
    full_name: str,
    repo_url: str,
    source_file: str,
    turn_index: int,
    pair: SpecStoryTurnPair,
) -> SpecStoryTurnRecord:
    return SpecStoryTurnRecord(
        full_name=full_name,
        repo_url=repo_url,
        source_file=source_file,
        turn_index=turn_index,
        prompt_text=pair.prompt_text,
        agent_response=pair.agent_response,
        agent_model=pair.agent_model,
        generated_code=pair.generated_code,
        code_block_count=pair.code_block_count,
        code_languages=pair.code_languages,
        has_generated_code=pair.has_generated_code,
        has_tool_use=pair.has_tool_use,
    )


def extract_turn_pairs_from_repo(
    full_name: str,
    repo_url: str | None = None,
    *,
    filter_noise: bool = True,
) -> list[SpecStoryTurnRecord]:
    """Pull full prompt/response turn pairs from SpecStory logs in one repository."""
    repo = _fetch_repo(full_name)
    if not repo:
        return []

    repo_url = repo_url or repo.get("html_url") or f"https://github.com/{full_name}"
    records: list[SpecStoryTurnRecord] = []

    for path in discover_specstory_files(full_name):
        text = fetch_file_text(full_name, path)
        if not text:
            continue

        for turn_index, pair in enumerate(parse_specstory_turns(text), start=1):
            if filter_noise and not _is_meaningful_prompt(pair.prompt_text):
                continue
            records.append(
                _pair_to_record(full_name, repo_url, path, turn_index, pair)
            )

    return records


def turn_records_to_dataframe(records: list[SpecStoryTurnRecord]) -> pd.DataFrame:
    """Convert turn records to a flat dataframe with standard column names."""
    rows: list[dict] = []
    for record in records:
        rows.append(
            {
                "full_name": record.full_name,
                "repo_url": record.repo_url,
                "source_file": record.source_file,
                "turn_index": record.turn_index,
                "artifact_type": "session_turn_pair",
                "prompt_text": record.prompt_text,
                "prompt_length": len(record.prompt_text),
                "prompt_word_count": len(record.prompt_text.split()),
                "agent_response": record.agent_response,
                "agent_response_length": len(record.agent_response),
                "agent_model": record.agent_model,
                "generated_code": record.generated_code,
                "generated_code_length": len(record.generated_code),
                "code_block_count": record.code_block_count,
                "code_languages": record.code_languages,
                "has_generated_code": record.has_generated_code,
                "has_tool_use": record.has_tool_use,
                "extraction_method": "specstory_turn_pair_parser",
            }
        )
    return pd.DataFrame(rows)


def _print_extraction_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("\nNo SpecStory turn pairs extracted.", flush=True)
        return

    repo_count = df["full_name"].nunique()
    file_count = df["source_file"].nunique()
    session_count = df["session_id"].nunique() if "session_id" in df.columns else file_count
    print(f"\nSpecStory turn pairs : {len(df)}", flush=True)
    print(f"Repos with turns     : {repo_count}", flush=True)
    print(f"Sessions (files)     : {session_count}", flush=True)
    print(f"Source files         : {file_count}", flush=True)
    if "has_generated_code" in df.columns:
        print(f"With generated code  : {int(df['has_generated_code'].sum())}", flush=True)
    if "has_tool_use" in df.columns:
        print(f"With tool use        : {int(df['has_tool_use'].sum())}", flush=True)
    if "action_type" in df.columns:
        print("Action types:", flush=True)
        for action, count in df["action_type"].value_counts().items():
            print(f"  {action}: {count}", flush=True)


def extract_specstory_turns_from_candidates(
    candidate_csv: str | Path,
    output_dir: str | Path = "outputs",
    max_repos: int | None = None,
    *,
    enrich: bool = True,
    checkpoint_every: int = 20,
    max_workers: int = 8,
) -> Path:
    """Extract full turn pairs (prompt + agent response) from candidate repositories.

    A run across hundreds of repos can take a long time and everything was
    previously held in memory until one save at the very end — a dropped
    connection or killed process meant losing all of it. Every
    `checkpoint_every` repos, progress so far is written to a fixed
    checkpoint path that gets overwritten in place, so an interrupted run
    still leaves usable, resumable data on disk.

    Extraction is I/O-bound (waiting on GitHub API responses), not CPU-bound,
    so repos are processed `max_workers` at a time via a thread pool instead
    of one at a time — a single repo's network latency no longer stalls the
    whole batch. GitHub's authenticated core-API limit is 5000 req/hour;
    8 concurrent workers stays comfortably inside that for a one-off run.
    """
    candidate_path = Path(candidate_csv)
    df = pd.read_csv(candidate_path)
    if df.empty:
        raise ValueError(f"No candidate rows found in {candidate_path}")

    if max_repos is not None:
        df = df.head(max_repos).copy()

    output_path = Path(output_dir)
    datasets_dir = output_path / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = datasets_dir / f"specstory_turn_pairs_{candidate_path.stem}_checkpoint.csv"

    seen: set[str] = set()
    repo_jobs: list[tuple[str, str | None]] = []
    for row in df.itertuples(index=False):
        full_name = getattr(row, "full_name")
        if full_name in seen:
            continue
        seen.add(full_name)
        repo_jobs.append((full_name, getattr(row, "url", None)))

    total = len(repo_jobs)
    records: list[SpecStoryTurnRecord] = []
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_repo = {
            executor.submit(extract_turn_pairs_from_repo, full_name, repo_url): full_name
            for full_name, repo_url in repo_jobs
        }
        for future in as_completed(future_to_repo):
            full_name = future_to_repo[future]
            completed += 1
            try:
                repo_records = future.result()
                records.extend(repo_records)
                print(f"[{completed:03d}/{total}] done: {full_name} ({len(repo_records)} turns)", flush=True)
            except Exception as exc:  # noqa: BLE001 - one bad repo must not kill a multi-hour run
                print(f"[{completed:03d}/{total}] skipping {full_name} after error: {exc!r}", flush=True)

            if checkpoint_every and completed % checkpoint_every == 0 and records:
                _save_checkpoint(records, checkpoint_file, enrich)

    if records:
        _save_checkpoint(records, checkpoint_file, enrich)

    extracted_df = turn_records_to_dataframe(records)
    if enrich and not extracted_df.empty:
        extracted_df = enrich_turn_dataset(extracted_df)

    _print_extraction_summary(extracted_df)

    ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M")
    output_file = datasets_dir / f"specstory_turn_pairs_{candidate_path.stem}_{ts}.csv"

    extracted_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    checkpoint_file.unlink(missing_ok=True)

    export_prompts_csv(extracted_df, datasets_dir / f"{output_file.stem}_spreadsheet.csv")
    export_prompts_csv(
        extracted_df,
        datasets_dir / f"{output_file.stem}_prompts_only.csv",
        prompts_only=True,
    )

    if enrich and not extracted_df.empty:
        summary = action_label_summary(extracted_df)
        pd.DataFrame([summary["counts"]]).to_csv(
            datasets_dir / f"{output_file.stem}_action_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

    print(f"Saved: {output_file}", flush=True)
    return output_file


def _save_checkpoint(records: list[SpecStoryTurnRecord], checkpoint_file: Path, enrich: bool) -> None:
    checkpoint_df = turn_records_to_dataframe(records)
    if enrich:
        checkpoint_df = enrich_turn_dataset(checkpoint_df)
    checkpoint_df.to_csv(checkpoint_file, index=False, encoding="utf-8-sig")
    print(f"   checkpoint saved ({len(records)} turn pairs so far): {checkpoint_file}", flush=True)


def extract_specstory_prompts_from_repo(
    full_name: str,
    repo_url: str | None = None,
) -> list[SpecStoryTurnRecord]:
    """Back-compat alias: returns full turn records (not prompt-only)."""
    return extract_turn_pairs_from_repo(full_name, repo_url=repo_url)


def extract_specstory_prompts_from_candidates(
    candidate_csv: str | Path,
    output_dir: str | Path = "outputs",
    max_repos: int | None = None,
    *,
    max_workers: int = 8,
) -> Path:
    """Back-compat alias: now extracts full turn pairs by default."""
    return extract_specstory_turns_from_candidates(
        candidate_csv,
        output_dir=output_dir,
        max_repos=max_repos,
        enrich=True,
        max_workers=max_workers,
    )


# Backwards-compatible aliases for older imports / callers.
PromptRecord = SpecStoryTurnRecord
SpecStoryPromptRecord = SpecStoryTurnRecord
extract_prompts_from_repo = extract_turn_pairs_from_repo
extract_prompts_from_candidates = extract_specstory_prompts_from_candidates
