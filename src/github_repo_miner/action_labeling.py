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


def infer_agent_action(
    prompt_text: str,
    agent_response: str = "",
    *,
    has_generated_code: bool = False,
    has_tool_use: bool = False,
    code_block_count: int = 0,
    generated_code: str = "",
) -> ActionType:
    """
    Infer what the agent primarily did on this turn.

    Uses both the developer prompt and the agent response. This is intentionally
    rule-based: it is a bootstrap label to train on and audit, not ground truth.
    """
    prompt = str(prompt_text or "").strip()
    response = str(agent_response or "").strip()
    if response == "nan":
        response = ""
    code_len = len(str(generated_code or "").strip())
    blocks = int(code_block_count or 0)

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


def add_action_labels(df):
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
