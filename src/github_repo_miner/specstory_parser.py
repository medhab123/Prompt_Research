"""Shared SpecStory transcript parser used by extraction and audits."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .data_cleaning import sanitize_turn_text

_ROLE_WORDS = r"(User|Human|Developer|Assistant|Agent|AI|System|Tool)"

SPEAKER_PATTERNS = (
    re.compile(r"^\s*#{1,6}\s*\*{0,3}\s*" + _ROLE_WORDS + r"\b.*$", re.IGNORECASE),
    re.compile(r"^\s*_{0,3}\*{1,3}\s*" + _ROLE_WORDS + r"\b.*$", re.IGNORECASE),
    re.compile(r"^\s*" + _ROLE_WORDS + r"\s*:\s*.*$", re.IGNORECASE),
)
ROLE_COLON_RE = re.compile(r"^\s*" + _ROLE_WORDS + r"\s*:\s*(.*)$", re.IGNORECASE)
USER_ROLES = {"user", "human", "developer"}
ASSISTANT_ROLES = {"assistant", "agent", "ai"}
PROMPT_FIELD_RE = re.compile(
    r"^\s*(prompt|query|request|instruction)\s*:\s*(.+?)\s*$",
    re.IGNORECASE,
)
_SPEAKER_MODEL_RE = re.compile(r"\(([^)]+)\)")
_CODE_FENCE_RE = re.compile(r"```(\w+)?[^\n]*\n(.*?)```", re.DOTALL)
_TOOL_BLOCK_RE = re.compile(r"<tool-use\b.*?</tool-use>", re.DOTALL | re.IGNORECASE)
_TOOL_NAME_RE = re.compile(r'data-tool-name="([^"]*)"')
CODE_BLOCK_JOINER = "\n\n<<CODE_BLOCK>>\n\n"


@dataclass
class SpecStoryTurnPair:
    prompt_text: str
    agent_response: str = ""
    agent_model: str = ""
    generated_code: str = ""
    code_block_count: int = 0
    code_languages: str = ""
    has_generated_code: bool = False
    has_tool_use: bool = False


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _speaker_role(line: str) -> str | None:
    for pattern in SPEAKER_PATTERNS:
        match = pattern.match(line)
        if match:
            return match.group(1).lower()
    return None


def _inline_after_role(line: str) -> str:
    match = ROLE_COLON_RE.match(line)
    return match.group(2).strip() if match else ""


def _speaker_model(line: str) -> str:
    match = _SPEAKER_MODEL_RE.search(line)
    return match.group(1).strip() if match else ""


def _clean_turn(lines: list[str]) -> str:
    return sanitize_turn_text("\n".join(lines))


def _summarize_tool_blocks(text: str) -> str:
    """Replace <tool-use>...</tool-use> with a short `[tool: Name]` marker.

    sanitize_turn_text strips tool-use blocks entirely, which is right for
    developer prompts but wrong for agent responses: a turn that's purely
    "call Edit, call Bash, call Edit" (no surrounding prose) would otherwise
    clean down to an empty string, even though real work happened.
    """

    def _replace(match: re.Match[str]) -> str:
        name_match = _TOOL_NAME_RE.search(match.group(0))
        name = name_match.group(1) if name_match else "tool"
        return f"[tool: {name}]"

    return _TOOL_BLOCK_RE.sub(_replace, text)


def _clean_agent_turn(lines: list[str]) -> str:
    return sanitize_turn_text(_summarize_tool_blocks("\n".join(lines)))


def _extract_code_blocks(text: str) -> tuple[list[str], list[str]]:
    blocks: list[str] = []
    langs: list[str] = []
    for match in _CODE_FENCE_RE.finditer(text or ""):
        lang = (match.group(1) or "").strip().lower() or "unknown"
        code = match.group(2).strip()
        if code:
            blocks.append(code)
            langs.append(lang)
    return blocks, langs


def _turn_has_tool_use(lines: list[str]) -> bool:
    return bool(_TOOL_BLOCK_RE.search("\n".join(lines)))


def parse_specstory_turns(text: str) -> list[SpecStoryTurnPair]:
    """Parse user prompts paired with the next assistant turn and code metadata."""
    turns_meta: list[dict] = []
    current_role: str | None = None
    current_lines: list[str] = []
    current_model = ""
    saw_marker = False

    def flush() -> None:
        nonlocal current_role, current_lines, current_model
        if current_role is None:
            current_lines = []
            return
        turns_meta.append({
            "role": current_role,
            "model": current_model,
            "lines": list(current_lines),
            "has_tool_use": _turn_has_tool_use(current_lines),
        })
        current_role = None
        current_lines = []
        current_model = ""

    for raw_line in text.splitlines():
        role = _speaker_role(raw_line)
        if role is not None:
            saw_marker = True
            flush()
            current_role = role
            current_model = _speaker_model(raw_line)
            inline = _inline_after_role(raw_line)
            current_lines = [inline] if inline else []
            continue
        if current_role is not None:
            current_lines.append(raw_line)
    flush()

    if not saw_marker:
        pairs: list[SpecStoryTurnPair] = []
        for raw_line in text.splitlines():
            match = PROMPT_FIELD_RE.match(raw_line)
            if match:
                prompt = _normalize_text(match.group(2))
                if prompt:
                    pairs.append(SpecStoryTurnPair(prompt_text=prompt))
        return pairs

    pairs: list[SpecStoryTurnPair] = []
    for idx, turn in enumerate(turns_meta):
        if turn["role"] not in USER_ROLES:
            continue
        prompt = _clean_turn(turn["lines"])
        if not prompt:
            continue

        # SpecStory logs split one logical agent turn into several consecutive
        # "_**Agent (...)**_" blocks — one per tool call, plus prose — rather
        # than a single block. Merge every consecutive assistant block up to
        # the next user turn, or the response collapses to just the first
        # tool call and silently drops the actual summary text.
        agent_lines: list[str] = []
        agent_model = ""
        has_tool_use = False
        j = idx + 1
        while j < len(turns_meta) and turns_meta[j]["role"] in ASSISTANT_ROLES:
            nxt = turns_meta[j]
            if not agent_model:
                agent_model = nxt["model"]
            if agent_lines and nxt["lines"]:
                agent_lines.append("")
            agent_lines.extend(nxt["lines"])
            has_tool_use = has_tool_use or nxt["has_tool_use"]
            j += 1
        agent_response = _clean_agent_turn(agent_lines)

        blocks, langs = _extract_code_blocks(agent_response)
        pairs.append(
            SpecStoryTurnPair(
                prompt_text=prompt,
                agent_response=agent_response,
                agent_model=agent_model,
                generated_code=CODE_BLOCK_JOINER.join(blocks),
                code_block_count=len(blocks),
                code_languages=", ".join(sorted(set(langs))) if langs else "",
                has_generated_code=len(blocks) > 0,
                has_tool_use=has_tool_use,
            )
        )

    seen: set[str] = set()
    deduped: list[SpecStoryTurnPair] = []
    for pair in pairs:
        if pair.prompt_text in seen:
            continue
        seen.add(pair.prompt_text)
        deduped.append(pair)
    return deduped


def parse_specstory_prompts(text: str) -> list[str]:
    return [pair.prompt_text for pair in parse_specstory_turns(text)]
