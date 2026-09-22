"""Rule-based agent action labels from prompt + response pairs."""

from __future__ import annotations

import re
from typing import Literal

ActionType = Literal[
    "implement",
    "explain",
    "debug",
    "review",
    "tool_only",
    "other",
]

DEBUG_PROMPT = re.compile(
    r"\b(error|errors|bug|bugs|crash|crashes|fail|failing|failed|exception|"
    r"stack\s*trace|traceback|broken|doesn'?t work|not working|undefined|"
    r"uncaught|referenceerror|typeerror)\b",
    re.IGNORECASE,
)
# v3: problem-report phrasings the v1 list missed ("didn't work", "still not
# centered", "can't click", "throws"). Kept only phrasings whose new debug calls
# agreed with the LLM label >=60% of the time on the corpus (2026-09-17); generic
# words ("issue", "problem", "warning", "failure") agreed <50% and were left out
# because they also appear in ordinary feature requests.
DEBUG_PROMPT_V3_EXTRA = re.compile(
    r"\b(didn'?t work|does not work|do not work|isn'?t working|aren'?t working|no longer works?|"
    r"stopped working|still (?:not|doesn'?t|don'?t|isn'?t|aren'?t|broken|wrong|happening|there|shows?|"
    r"showing|appears?|getting|seeing|fails?|failing)|not (?:correct|right|showing|displaying|loading|"
    r"rendering|updating|saving|responsive|centered|aligned|clickable|visible|appearing|triggering|"
    r"firing|found)|doesn'?t (?:show|display|load|appear|update|save|respond|render|open|close|compile|"
    r"build|run|start|exist)|can'?t (?:click|see|find|load|open|access|connect|resolve|import)|"
    r"cannot (?:find|read|resolve|access|connect|import|load|open)|unable to|nothing happens|wrong|"
    r"incorrect|messed up|throws?|thrown|segfault|panic(?:ked)?|hangs?|freez(?:es|ing)|stuck)\b"
    r"|Error:|\bERROR\b",
    re.IGNORECASE,
)
EXPLAIN_PROMPT = re.compile(
    r"\b(why|how|what|explain|describe|clarify|understand|difference between|"
    r"walk me through|help me understand)\b",
    re.IGNORECASE,
)
IMPLEMENT_PROMPT = re.compile(
    r"\b(implement|create|add|write|build|make|develop|scaffold|generate|"
    r"refactor|update|fix|modify|change|introduce|port|migrate)\b",
    re.IGNORECASE,
)
REVIEW_PROMPT = re.compile(
    r"\b(review|look at|audit|check|feedback|critique|evaluate|assess|"
    r"can we|should we|thoughts on|what do you think)\b",
    re.IGNORECASE,
)
TEST_PROMPT = re.compile(
    r"\b(test|tests|testing|pytest|unittest|coverage|spec)\b",
    re.IGNORECASE,
)

# Rough thresholds for "substantial" generated code vs explanatory snippet.
SUBSTANTIAL_CODE_CHARS = 350
SUBSTANTIAL_CODE_BLOCKS = 2

# v1: the original rules, behind the rule-baseline numbers in paper/NUMBERS.md.
# v2 (2026-09-17): same keyword lists and thresholds, two ordering bugs fixed:
#   - debug keywords were only consulted after the code-volume rule, so a bug
#     report answered with a multi-block fix was always "implement";
#   - "tools used, no code block" returned tool_only before looking at what the
#     tools did, so edits made through Edit/apply_patch were never "implement".
# v3 (2026-09-17): v2 plus DEBUG_PROMPT_V3_EXTRA problem-report phrasings.
RULE_VERSIONS = ("v1", "v2", "v3")
RULE_VERSION = "v3"

# Tool names as they survive in agent_response: "[tool: X]" markers,
# "Tool use: **X**", and legacy "<summary>Edit file" collapsibles.
_TOOL_NAME_IN_RESPONSE_RE = re.compile(
    r"\[tool: ([^\]\n]+)\]|Tool use: \*\*([\w\-]+)\*\*|<summary>\s*(Edit file|Apply patch|Deleted file)",
    re.IGNORECASE,
)
_EDIT_TOOL_RE = re.compile(r"edit|write|replace|patch|create_?file|delete", re.IGNORECASE)
_NON_EDIT_TOOL_RE = re.compile(r"todo|stdin|plan", re.IGNORECASE)
_TOOL_MARKUP_RE = re.compile(
    r"\[tool: [^\]\n]+\]|Tool use: \*\*[\w\-]+\*\*|<details>.*?</details>|_\*\*\*\*_|^[ \t]*Read file: [^\n]*",
    re.IGNORECASE | re.DOTALL | re.MULTILINE,
)
# Prose left after tool markup is stripped, above which a read-only tool turn
# counts as having answered (explain/review) rather than just exploring.
TOOL_TURN_ANSWER_CHARS = 200


def response_uses_edit_tools(agent_response: str) -> bool:
    """True if the response shows a file-modifying tool call (Edit, apply_patch, …)."""
    for match in _TOOL_NAME_IN_RESPONSE_RE.finditer(str(agent_response or "")):
        name = next(g for g in match.groups() if g)
        if _EDIT_TOOL_RE.search(name) and not _NON_EDIT_TOOL_RE.search(name):
            return True
    return False


def infer_agent_action(
    prompt_text: str,
    agent_response: str = "",
    *,
    has_generated_code: bool = False,
    has_tool_use: bool = False,
    code_block_count: int = 0,
    generated_code: str = "",
    rule_version: str = RULE_VERSION,
) -> ActionType:
    """
    Infer what the agent primarily did on this turn.

    Uses both the developer prompt and the agent response. This is intentionally
    rule-based: it is a bootstrap label to train on and audit, not ground truth.
    `rule_version="v1"` reproduces the original labels.
    """
    if rule_version not in RULE_VERSIONS:
        raise ValueError(f"unknown rule_version {rule_version!r}; expected one of {RULE_VERSIONS}")
    prompt = str(prompt_text or "").strip()
    response = str(agent_response or "").strip()
    if response == "nan":
        response = ""
    code_len = len(str(generated_code or "").strip())
    blocks = int(code_block_count or 0)

    if rule_version in ("v2", "v3"):
        is_debug_prompt = bool(DEBUG_PROMPT.search(prompt)) or (
            rule_version == "v3" and bool(DEBUG_PROMPT_V3_EXTRA.search(prompt))
        )
        is_explain_prompt = bool(EXPLAIN_PROMPT.search(prompt)) or "?" in prompt
        is_implement_prompt = bool(IMPLEMENT_PROMPT.search(prompt))
        is_review_prompt = bool(REVIEW_PROMPT.search(prompt))

        # A reported problem the agent acted on is debug, whatever form the
        # fix took (code blocks, tool edits, or a substantive written diagnosis).
        if is_debug_prompt and (has_generated_code or has_tool_use or len(response) > 80):
            return "debug"

        if has_tool_use and blocks == 0:
            if response_uses_edit_tools(response):
                return "implement"
            prose = _TOOL_MARKUP_RE.sub(" ", response).strip()
            if len(prose) >= TOOL_TURN_ANSWER_CHARS:
                if is_explain_prompt and not is_implement_prompt:
                    return "explain"
                if is_review_prompt:
                    return "review"
            return "tool_only"
        # Remaining cases are unchanged from v1 (debug prompts already handled).

    is_debug_prompt = bool(DEBUG_PROMPT.search(prompt))
    is_explain_prompt = bool(EXPLAIN_PROMPT.search(prompt)) or "?" in prompt
    is_implement_prompt = bool(IMPLEMENT_PROMPT.search(prompt))
    is_review_prompt = bool(REVIEW_PROMPT.search(prompt))

    # Agent used tools and did not surface code in the final reply.
    if has_tool_use and blocks == 0:
        return "tool_only"

    # Multi-block or long code output → implementation regardless of prompt wording.
    if has_generated_code and (
        blocks >= SUBSTANTIAL_CODE_BLOCKS or code_len >= SUBSTANTIAL_CODE_CHARS
    ):
        return "implement"

    # Single code block: disambiguate with prompt intent.
    if has_generated_code and blocks == 1:
        if is_debug_prompt:
            return "debug"
        if is_explain_prompt and not is_implement_prompt:
            return "explain"
        if is_implement_prompt:
            return "implement"
        if code_len < 120:
            return "explain"
        return "implement"

    # No code in response.
    if is_debug_prompt:
        return "debug" if len(response) > 80 else "explain"
    if is_explain_prompt:
        return "explain"
    if is_review_prompt:
        return "review"
    if is_implement_prompt:
        # User asked to build something; agent replied without code (plan/discussion).
        return "review"
    if has_tool_use:
        return "tool_only"
    if len(response) < 40 and not has_generated_code:
        return "other"
    return "explain" if len(response) > 0 else "other"


def add_action_labels(df, *, rule_version: str = RULE_VERSION):
    """Add `action_type` column to a turn-pair dataframe."""
    import pandas as pd

    out = df.copy()
    out["action_type"] = [
        infer_agent_action(
            row.get("prompt_text", ""),
            row.get("agent_response", ""),
            has_generated_code=bool(row.get("has_generated_code", False)),
            has_tool_use=bool(row.get("has_tool_use", False)),
            code_block_count=int(row.get("code_block_count", 0) or 0),
            generated_code=str(row.get("generated_code", "") or ""),
            rule_version=rule_version,
        )
        for _, row in out.iterrows()
    ]
    return out


def action_label_summary(df) -> dict:
    """Distribution of action_type labels."""
    if "action_type" not in df.columns:
        raise ValueError("DataFrame missing action_type column")
    counts = df["action_type"].value_counts().to_dict()
    total = len(df)
    return {
        "n_rows": total,
        "counts": counts,
        "pct": {k: round(v / total * 100, 2) for k, v in counts.items()},
    }
