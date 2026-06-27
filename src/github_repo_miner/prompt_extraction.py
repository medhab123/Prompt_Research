"""Extract prompt-like text from mined repositories."""

from __future__ import annotations

import base64
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from .config import BASE_URL, GITHUB_TOKEN, HEADERS

TARGET_ROOTS = [
    ".specstory/history",
    ".cursor/rules",
    ".github/copilot-instructions.md",
    "copilot-instructions.md",
    "CLAUDE.md",
    "AGENTS.md",
    ".cursorrules",
]

INSTRUCTION_FILE_NAMES = {
    "copilot-instructions.md",
    "claude.md",
    "agents.md",
}

RULES_PATH_FRAGMENTS = (
    ".cursor/rules",
    ".cursorrules",
)

SPECSTORY_PATH_FRAGMENT = ".specstory/history"

ARTIFACT_PRIORITY = {
    "session_prompt": 1,
    "inline_prompt_comment": 2,
    "instruction_file": 3,
    "rules_config": 4,
}

# Roles that mark a turn in a SpecStory / chat transcript.
_ROLE_WORDS = r"(User|Human|Developer|Assistant|Agent|AI|System|Tool)"

# Speaker markers, ordered from most to least specific. These cover:
#   _**User (2026-01-16 16:48:38Z)**_   (SpecStory v2 emphasis markers)
#   **User** / **Assistant**             (bold inline)
#   ## User / ### Human                  (markdown headings)
#   User:                                (prefix label)
SPEAKER_PATTERNS = (
    re.compile(r"^\s*#{1,6}\s*\*{0,3}\s*" + _ROLE_WORDS + r"\b.*$", re.IGNORECASE),
    re.compile(r"^\s*_{0,3}\*{1,3}\s*" + _ROLE_WORDS + r"\b.*$", re.IGNORECASE),
    re.compile(r"^\s*" + _ROLE_WORDS + r"\s*:\s*.*$", re.IGNORECASE),
)

# Only text after a "Role:" label is treated as inline content on the marker line.
ROLE_COLON_RE = re.compile(r"^\s*" + _ROLE_WORDS + r"\s*:\s*(.*)$", re.IGNORECASE)

USER_ROLES = {"user", "human", "developer"}

PROMPT_FIELD_RE = re.compile(r"^\s*(prompt|query|request|instruction)\s*:\s*(.+?)\s*$", re.IGNORECASE)

# CJK ranges so short non-space-delimited prompts are not dropped by word counting.
CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")

# Noise blocks that belong to assistant/tool output, stripped if they leak into a turn.
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_THINK_BLOCK_RE = re.compile(r"<think\b.*?</think>", re.DOTALL | re.IGNORECASE)
_TOOL_BLOCK_RE = re.compile(r"<tool-use\b.*?</tool-use>", re.DOTALL | re.IGNORECASE)
_SEPARATOR_LINES = {"---", "***", "___"}


def _speaker_role(line: str) -> str | None:
    """Return the lowercased role if the line is a speaker marker, else None."""
    for pattern in SPEAKER_PATTERNS:
        match = pattern.match(line)
        if match:
            return match.group(1).lower()
    return None


def _inline_after_role(line: str) -> str:
    """Return content that appears after a `Role:` label on the marker line."""
    match = ROLE_COLON_RE.match(line)
    if match:
        return match.group(2).strip()
    return ""


def _has_cjk(text: str) -> bool:
    return bool(CJK_RE.search(text))


def _clean_turn(lines: list[str]) -> str:
    """Join and clean the lines of a single captured user turn."""
    text = "\n".join(lines)
    text = _HTML_COMMENT_RE.sub("", text)
    text = _THINK_BLOCK_RE.sub("", text)
    text = _TOOL_BLOCK_RE.sub("", text)
    kept = [ln for ln in text.splitlines() if ln.strip() not in _SEPARATOR_LINES]
    return _normalize_text("\n".join(kept))


def _is_meaningful_prompt(text: str) -> bool:
    """Keep real prompts; drop fragments and empty turns (CJK-aware)."""
    stripped = text.strip()
    if len(stripped) < 5:
        return False
    if len(stripped.split()) >= 3:
        return True
    return _has_cjk(stripped) and len(stripped) >= 6

INLINE_PROMPT_PATTERNS = (
    re.compile(r"^\s*#\s*Prompt:\s*(.+?)\s*$", re.IGNORECASE),
    re.compile(r"^\s*//\s*Prompt:\s*(.+?)\s*$", re.IGNORECASE),
    re.compile(r"^\s*/\*\s*Prompt:\s*(.+?)\s*\*/\s*$", re.IGNORECASE),
)


@dataclass
class PromptRecord:
    full_name: str
    repo_url: str
    source_file: str
    artifact_type: str
    research_priority: int
    prompt_index: int
    prompt_text: str
    extraction_method: str
    source_line: int | None = None


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


def classify_artifact(path: str) -> tuple[str, str]:
    """Return (artifact_type, extraction_method) for a repository file path."""
    lower = path.lower().replace("\\", "/")
    name = Path(lower).name

    if SPECSTORY_PATH_FRAGMENT in lower:
        return "session_prompt", "specstory_turn_parser"

    if lower == ".cursorrules" or any(fragment in lower for fragment in RULES_PATH_FRAGMENTS):
        return "rules_config", "rules_file"

    if name in INSTRUCTION_FILE_NAMES or ".github/copilot-instructions.md" in lower:
        return "instruction_file", "instruction_file"

    if lower.endswith(".md") and "prompt" in lower:
        return "rules_config", "rules_file"

    return "instruction_file", "instruction_file"


def _is_prompt_file(path: str) -> bool:
    lower = path.lower().replace("\\", "/")
    name = Path(lower).name
    artifact_type, _ = classify_artifact(path)
    return artifact_type in {"session_prompt", "instruction_file", "rules_config"} or (
        lower.endswith(".md") and "prompt" in lower
    )


def _candidate_roots() -> Iterable[str]:
    return TARGET_ROOTS


def _walk_contents(full_name: str, path: str, depth: int = 0, max_depth: int = 2) -> list[dict]:
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


def discover_prompt_files(full_name: str) -> list[dict]:
    """Find prompt-bearing files in a repository."""
    files: dict[str, dict] = {}
    for root in _candidate_roots():
        for entry in _walk_contents(full_name, root):
            path = entry.get("path") or ""
            if path and _is_prompt_file(path):
                files[path] = entry
    return list(files.values())


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
    """Extract user turns from SpecStory-style transcripts.

    Handles modern SpecStory speaker markers such as ``_**User (timestamp)**_``
    and ``_**Agent (model timestamp)**_`` as well as older heading/bold/label
    formats. User turns are captured and stop cleanly at the next non-user
    speaker marker, so assistant replies and tool output are not merged in.
    """
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

    # Fallback for non-conversation files: frontmatter prompt:/query: fields.
    if not saw_marker:
        for raw_line in text.splitlines():
            match = PROMPT_FIELD_RE.match(raw_line)
            if match:
                prompts.append(_normalize_text(match.group(2)))

    # Drop fragments and de-duplicate while preserving order.
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


def parse_inline_prompt_comments(text: str) -> list[tuple[int, str]]:
    """Extract inline prompt comments such as `# Prompt:` or `// Prompt:`."""
    prompts: list[tuple[int, str]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        for pattern in INLINE_PROMPT_PATTERNS:
            match = pattern.match(raw_line)
            if not match:
                continue
            prompt = _normalize_text(match.group(1))
            if len(prompt.split()) >= 3:
                prompts.append((line_number, prompt))
            break
    return prompts


def _search_repo_code(full_name: str, query_fragment: str) -> list[str]:
    """Return file paths in a repo that match a code-search fragment."""
    if not GITHUB_TOKEN:
        return []

    params = {
        "q": f"repo:{full_name} {query_fragment}",
        "per_page": 30,
        "page": 1,
    }
    payload = _request_json(f"{BASE_URL}/search/code", params=params)
    if not isinstance(payload, dict):
        return []

    paths: list[str] = []
    for item in payload.get("items", []):
        path = item.get("path")
        if path:
            paths.append(path)
    return paths


def discover_inline_prompt_files(full_name: str) -> list[str]:
    """Find source files that contain inline prompt comments."""
    paths: set[str] = set()
    for fragment in ('"# Prompt:"', '"// Prompt:"', '"AI-generated"'):
        paths.update(_search_repo_code(full_name, fragment))
    return sorted(paths)


def _record_from_parts(
    *,
    full_name: str,
    repo_url: str,
    source_file: str,
    artifact_type: str,
    extraction_method: str,
    prompt_index: int,
    prompt_text: str,
    source_line: int | None = None,
) -> PromptRecord:
    return PromptRecord(
        full_name=full_name,
        repo_url=repo_url,
        source_file=source_file,
        artifact_type=artifact_type,
        research_priority=ARTIFACT_PRIORITY[artifact_type],
        prompt_index=prompt_index,
        prompt_text=prompt_text,
        extraction_method=extraction_method,
        source_line=source_line,
    )


def extract_prompts_from_repo(full_name: str, repo_url: str | None = None) -> list[PromptRecord]:
    repo = _fetch_repo(full_name)
    if not repo:
        return []

    repo_url = repo_url or repo.get("html_url") or f"https://github.com/{full_name}"
    records: list[PromptRecord] = []

    for entry in discover_prompt_files(full_name):
        path = entry.get("path") or ""
        if not path:
            continue

        text = fetch_file_text(full_name, path)
        if not text:
            continue

        artifact_type, extraction_method = classify_artifact(path)
        if artifact_type == "session_prompt":
            prompts = parse_specstory_prompts(text)
        else:
            prompts = [_normalize_text(text)]

        for index, prompt in enumerate(prompts, start=1):
            records.append(
                _record_from_parts(
                    full_name=full_name,
                    repo_url=repo_url,
                    source_file=path,
                    artifact_type=artifact_type,
                    extraction_method=extraction_method,
                    prompt_index=index,
                    prompt_text=prompt,
                )
            )

    inline_paths = discover_inline_prompt_files(full_name)
    for path in inline_paths:
        text = fetch_file_text(full_name, path)
        if not text:
            continue

        inline_prompts = parse_inline_prompt_comments(text)
        for index, (line_number, prompt) in enumerate(inline_prompts, start=1):
            records.append(
                _record_from_parts(
                    full_name=full_name,
                    repo_url=repo_url,
                    source_file=path,
                    artifact_type="inline_prompt_comment",
                    extraction_method="inline_comment_parser",
                    prompt_index=index,
                    prompt_text=prompt,
                    source_line=line_number,
                )
            )

    return records


def _apply_artifact_filters(
    df: pd.DataFrame,
    *,
    max_priority: int | None = None,
    artifact_types: list[str] | None = None,
) -> pd.DataFrame:
    filtered = df.copy()
    if max_priority is not None:
        filtered = filtered[filtered["research_priority"] <= max_priority]
    if artifact_types:
        allowed = {value.strip() for value in artifact_types if value.strip()}
        filtered = filtered[filtered["artifact_type"].isin(allowed)]
    return filtered.reset_index(drop=True)


def _print_extraction_summary(df: pd.DataFrame) -> None:
    print("\nArtifact breakdown:", flush=True)
    if df.empty:
        print("  (no rows)", flush=True)
        return

    counts = df["artifact_type"].value_counts().sort_index()
    for artifact_type, count in counts.items():
        priority = ARTIFACT_PRIORITY[artifact_type]
        print(f"  priority {priority} | {artifact_type}: {count}", flush=True)


def extract_prompts_from_candidates(
    candidate_csv: str | Path,
    output_dir: str | Path = "outputs",
    max_repos: int | None = None,
    max_priority: int | None = None,
    artifact_types: list[str] | None = None,
) -> Path:
    """Extract prompt-like text from the mined candidate repositories."""
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
        print(f"[{index:03d}/{len(df)}] extracting from {full_name}", flush=True)
        records = extract_prompts_from_repo(full_name, repo_url=repo_url)
        for record in records:
            rows.append(
                {
                    "full_name": record.full_name,
                    "repo_url": record.repo_url,
                    "source_file": record.source_file,
                    "source_line": record.source_line,
                    "artifact_type": record.artifact_type,
                    "research_priority": record.research_priority,
                    "prompt_index": record.prompt_index,
                    "prompt_text": record.prompt_text,
                    "prompt_length": len(record.prompt_text),
                    "prompt_word_count": len(record.prompt_text.split()),
                    "extraction_method": record.extraction_method,
                }
            )

    extracted_df = pd.DataFrame(rows)
    _print_extraction_summary(extracted_df)

    filtered_df = _apply_artifact_filters(
        extracted_df,
        max_priority=max_priority,
        artifact_types=artifact_types,
    )
    if len(filtered_df) != len(extracted_df):
        print(
            f"\nApplied artifact filter: {len(extracted_df)} -> {len(filtered_df)} rows",
            flush=True,
        )
        _print_extraction_summary(filtered_df)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M")
    output_file = output_path / f"extracted_prompts_{candidate_path.stem}_{ts}.csv"

    filtered_df.to_csv(output_file, index=False)

    print(f"\nExtracted prompts: {len(filtered_df)}", flush=True)
    print(f"Saved prompt dataset: {output_file.name}", flush=True)
    return output_file
