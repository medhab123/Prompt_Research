"""Extract user prompts from SpecStory session logs (`.specstory/history/*.md`)."""

from __future__ import annotations

import base64
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd
import requests

from .config import BASE_URL, GITHUB_TOKEN, HEADERS
from .data_cleaning import classify_noise, sanitize_prompt_text

SPECSTORY_ROOT = ".specstory"
SPECSTORY_HISTORY = ".specstory/history"
SPECSTORY_PATH_FRAGMENT = ".specstory/history"

# Roles that mark a turn in a SpecStory / chat transcript.
_ROLE_WORDS = r"(User|Human|Developer|Assistant|Agent|AI|System|Tool)"

# Speaker markers, ordered from most to least specific.
SPEAKER_PATTERNS = (
    re.compile(r"^\s*#{1,6}\s*\*{0,3}\s*" + _ROLE_WORDS + r"\b.*$", re.IGNORECASE),
    re.compile(r"^\s*_{0,3}\*{1,3}\s*" + _ROLE_WORDS + r"\b.*$", re.IGNORECASE),
    re.compile(r"^\s*" + _ROLE_WORDS + r"\s*:\s*.*$", re.IGNORECASE),
)

ROLE_COLON_RE = re.compile(r"^\s*" + _ROLE_WORDS + r"\s*:\s*(.*)$", re.IGNORECASE)
USER_ROLES = {"user", "human", "developer"}
PROMPT_FIELD_RE = re.compile(r"^\s*(prompt|query|request|instruction)\s*:\s*(.+?)\s*$", re.IGNORECASE)


def _speaker_role(line: str) -> str | None:
    for pattern in SPEAKER_PATTERNS:
        match = pattern.match(line)
        if match:
            return match.group(1).lower()
    return None


def _inline_after_role(line: str) -> str:
    match = ROLE_COLON_RE.match(line)
    if match:
        return match.group(2).strip()
    return ""


def _clean_turn(lines: list[str]) -> str:
    return sanitize_prompt_text("\n".join(lines))


def _is_meaningful_prompt(text: str) -> bool:
    return classify_noise(text) is None


@dataclass
class SpecStoryPromptRecord:
    full_name: str
    repo_url: str
    source_file: str
    prompt_index: int
    prompt_text: str


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
            return response.json()
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


def parse_specstory_prompts(text: str) -> list[str]:
    """Extract user turns from SpecStory-style transcripts."""
    prompts: list[str] = []
    current: list[str] = []
    capturing = False
    saw_marker = False

    def flush() -> None:
        nonlocal current
        candidate = _clean_turn(current)
        current = []
        if candidate:
            prompts.append(candidate)

    for raw_line in text.splitlines():
        role = _speaker_role(raw_line)
        if role is not None:
            saw_marker = True
            if capturing and current:
                flush()
            capturing = role in USER_ROLES
            current = []
            if capturing:
                inline = _inline_after_role(raw_line)
                if inline:
                    current.append(inline)
            continue
        if capturing:
            current.append(raw_line)

    if capturing and current:
        flush()

    if not saw_marker:
        for raw_line in text.splitlines():
            match = PROMPT_FIELD_RE.match(raw_line)
            if match:
                prompts.append(_normalize_text(match.group(2)))

    seen: set[str] = set()
    cleaned: list[str] = []
    for prompt in prompts:
        if not _is_meaningful_prompt(prompt):
            continue
        if prompt in seen:
            continue
        seen.add(prompt)
        cleaned.append(prompt)
    return cleaned


def extract_specstory_prompts_from_repo(
    full_name: str,
    repo_url: str | None = None,
) -> list[SpecStoryPromptRecord]:
    """Pull every user prompt from SpecStory logs in one repository."""
    repo = _fetch_repo(full_name)
    if not repo:
        return []

    repo_url = repo_url or repo.get("html_url") or f"https://github.com/{full_name}"
    records: list[SpecStoryPromptRecord] = []

    for path in discover_specstory_files(full_name):
        text = fetch_file_text(full_name, path)
        if not text:
            continue

        for index, prompt in enumerate(parse_specstory_prompts(text), start=1):
            records.append(
                SpecStoryPromptRecord(
                    full_name=full_name,
                    repo_url=repo_url,
                    source_file=path,
                    prompt_index=index,
                    prompt_text=prompt,
                )
            )

    return records


def _print_extraction_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("\nNo SpecStory prompts extracted.", flush=True)
        return

    repo_count = df["full_name"].nunique()
    file_count = df["source_file"].nunique()
    print(f"\nSpecStory prompts : {len(df)}", flush=True)
    print(f"Repos with prompts: {repo_count}", flush=True)
    print(f"Source files      : {file_count}", flush=True)


def extract_specstory_prompts_from_candidates(
    candidate_csv: str | Path,
    output_dir: str | Path = "outputs",
    max_repos: int | None = None,
) -> Path:
    """Extract SpecStory user prompts from mined candidate repositories."""
    candidate_path = Path(candidate_csv)
    df = pd.read_csv(candidate_path)
    if df.empty:
        raise ValueError(f"No candidate rows found in {candidate_path}")

    if max_repos is not None:
        df = df.head(max_repos).copy()

    seen: set[str] = set()
    rows: list[dict] = []

    for index, row in enumerate(df.itertuples(index=False), start=1):
        full_name = getattr(row, "full_name")
        if full_name in seen:
            continue
        seen.add(full_name)

        repo_url = getattr(row, "url", None)
        print(f"[{index:03d}/{len(df)}] SpecStory extract: {full_name}", flush=True)
        records = extract_specstory_prompts_from_repo(full_name, repo_url=repo_url)
        for record in records:
            rows.append(
                {
                    "full_name": record.full_name,
                    "repo_url": record.repo_url,
                    "source_file": record.source_file,
                    "artifact_type": "session_prompt",
                    "prompt_index": record.prompt_index,
                    "prompt_text": record.prompt_text,
                    "prompt_length": len(record.prompt_text),
                    "prompt_word_count": len(record.prompt_text.split()),
                    "extraction_method": "specstory_turn_parser",
                }
            )

    extracted_df = pd.DataFrame(rows)
    _print_extraction_summary(extracted_df)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M")
    output_file = output_path / f"specstory_prompts_{candidate_path.stem}_{ts}.csv"

    extracted_df.to_csv(output_file, index=False)

    print(f"Saved: {output_file.name}", flush=True)
    return output_file


# Backwards-compatible aliases for older imports / callers.
PromptRecord = SpecStoryPromptRecord
extract_prompts_from_repo = extract_specstory_prompts_from_repo
extract_prompts_from_candidates = extract_specstory_prompts_from_candidates
