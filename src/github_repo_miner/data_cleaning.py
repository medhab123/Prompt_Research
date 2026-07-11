"""Clean and filter mined SpecStory prompt datasets."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")

_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_THINK_BLOCK_RE = re.compile(r"<think\b.*?</think>", re.DOTALL | re.IGNORECASE)
_TOOL_BLOCK_RE = re.compile(r"<tool-use\b.*?</tool-use>", re.DOTALL | re.IGNORECASE)
_XML_TAG_RE = re.compile(r"</?(?:command-name|command-message|command-args|local-command-\w+)[^>]*>", re.IGNORECASE)

_NOISE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("local_command_xml", re.compile(r"<local-command|</local-command|<command-name>|<command-message>", re.I)),
    ("local_command_caveat", re.compile(r"DO NOT respond to these messages|local-command-caveat", re.I)),
    ("request_interrupted", re.compile(r"^\[?request interrupted by user", re.I)),
    ("slash_command", re.compile(r"^/\w+(?:\s|$)", re.I)),
    ("ci_runner_log", re.compile(r"/home/runner/work/|hostedtoolcache|::error::|goreleaser-action@", re.I)),
    ("browser_console_log", re.compile(
        r"react-dom_client\.js|hot-reloader-client\.js|performWorkOnRoot|"
        r"Download the React DevTools|\[Fast Refresh\]|dispatchDiscreteEvent",
        re.I,
    )),
    ("stack_trace_dump", re.compile(r"(?:\bat\s+\S+\s+\([^)]+\.(?:tsx?|jsx?):\d+\)\n){5,}", re.I)),
    ("agent_template_dump", re.compile(r"^\s*(?:VAN QA|Response:\s*OK VAN QA|BEGINNING TECHNICAL VALIDATION)", re.I)),
]

_ACK_RE = re.compile(
    r"^(?:ok|yes|no|thanks?|thank you|done|continue|go ahead|sounds good|looks good|"
    r"lgtm|yep|nope|got it|understood|perfect|great|nice|cool|sure|i installed it|"
    r"i added the token)\.?!?$",
    re.I,
)

_DERIVED_TEXT_COLS = ("prompt_length", "prompt_word_count")


@dataclass
class CleaningConfig:
    """Thresholds and toggles for prompt dataset cleaning."""

    min_chars: int = 25
    min_words: int = 4
    min_chars_cjk: int = 6
    max_chars: int = 8_000
    drop_noise_patterns: bool = True
    drop_acknowledgments: bool = True
    drop_duplicates: bool = True
    dedupe_subset: tuple[str, ...] = ("prompt_text",)


@dataclass
class CleaningReport:
    """Summary of rows removed during cleaning."""

    rows_in: int = 0
    rows_out: int = 0
    dropped_empty: int = 0
    dropped_noise: int = 0
    dropped_too_short: int = 0
    dropped_too_long: int = 0
    dropped_ack: int = 0
    dropped_duplicates: int = 0
    noise_reasons: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, int | dict[str, int]]:
        return {
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "dropped_empty": self.dropped_empty,
            "dropped_noise": self.dropped_noise,
            "dropped_too_short": self.dropped_too_short,
            "dropped_too_long": self.dropped_too_long,
            "dropped_ack": self.dropped_ack,
            "dropped_duplicates": self.dropped_duplicates,
            "noise_reasons": dict(self.noise_reasons),
        }


def _has_cjk(text: str) -> bool:
    return bool(CJK_RE.search(text))


def sanitize_prompt_text(text: str) -> str:
    """Normalize whitespace and strip known SpecStory / agent markup."""
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)

    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\ufeff", "")
    text = _HTML_COMMENT_RE.sub("", text)
    text = _THINK_BLOCK_RE.sub("", text)
    text = _TOOL_BLOCK_RE.sub("", text)
    text = _XML_TAG_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def classify_noise(text: str, *, config: CleaningConfig | None = None) -> str | None:
    """Return a drop reason when text is clearly not a developer prompt."""
    config = config or CleaningConfig()
    stripped = text.strip()
    if not stripped:
        return "empty"

    if config.drop_acknowledgments and _ACK_RE.match(stripped):
        return "acknowledgment"

    if config.drop_noise_patterns:
        for reason, pattern in _NOISE_PATTERNS:
            if pattern.search(stripped):
                return reason

    char_len = len(stripped)
    word_len = len(stripped.split())

    if char_len > config.max_chars:
        return "too_long"

    if _has_cjk(stripped):
        if char_len < config.min_chars_cjk:
            return "too_short"
    elif char_len < config.min_chars or word_len < config.min_words:
        return "too_short"

    return None


def is_quality_prompt(text: str, *, config: CleaningConfig | None = None) -> bool:
    return classify_noise(text, config=config) is None


def _recompute_text_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["prompt_text"] = out["prompt_text"].astype(str)
    out["prompt_length"] = out["prompt_text"].str.len()
    out["prompt_word_count"] = out["prompt_text"].str.split().str.len()
    return out


def clean_prompts_dataframe(
    df: pd.DataFrame,
    *,
    config: CleaningConfig | None = None,
    text_column: str = "prompt_text",
) -> tuple[pd.DataFrame, pd.DataFrame, CleaningReport]:
    """Return cleaned prompts, dropped rows (with reasons), and a summary report."""
    config = config or CleaningConfig()
    report = CleaningReport(rows_in=len(df))

    if df.empty:
        return df.copy(), df.copy(), report

    working = df.copy()
    working[text_column] = working[text_column].map(sanitize_prompt_text)

    empty_mask = working[text_column].str.len() == 0
    report.dropped_empty = int(empty_mask.sum())

    reasons: list[str | None] = []
    for text in working[text_column]:
        reasons.append(classify_noise(text, config=config))
    working["_drop_reason"] = reasons

    reason_series = working["_drop_reason"]
    report.dropped_ack = int((reason_series == "acknowledgment").sum())
    report.dropped_too_short = int((reason_series == "too_short").sum())
    report.dropped_too_long = int((reason_series == "too_long").sum())

    noise_mask = reason_series.notna() & ~reason_series.isin({"empty", "acknowledgment", "too_short", "too_long"})
    report.dropped_noise = int(noise_mask.sum())
    for reason, count in reason_series[noise_mask].value_counts().items():
        report.noise_reasons[str(reason)] = int(count)

    drop_mask = empty_mask | reason_series.notna()
    dropped = working[drop_mask].copy()
    cleaned = working[~drop_mask].drop(columns=["_drop_reason"])

    before_dedupe = len(cleaned)
    if config.drop_duplicates and not cleaned.empty:
        cleaned = cleaned.drop_duplicates(subset=list(config.dedupe_subset)).reset_index(drop=True)
        report.dropped_duplicates = before_dedupe - len(cleaned)
    else:
        cleaned = cleaned.reset_index(drop=True)

    cleaned = _recompute_text_metrics(cleaned)
    report.rows_out = len(cleaned)
    return cleaned, dropped.reset_index(drop=True), report


def print_cleaning_report(report: CleaningReport) -> None:
    """Pretty-print cleaning stats for notebooks / CLI."""
    total_dropped = report.rows_in - report.rows_out
    print(f"Rows in          : {report.rows_in}")
    print(f"Rows out         : {report.rows_out}")
    print(f"Rows dropped     : {total_dropped} ({100 * total_dropped / max(report.rows_in, 1):.1f}%)")
    print(f"  empty          : {report.dropped_empty}")
    print(f"  noise patterns : {report.dropped_noise}")
    print(f"  too short      : {report.dropped_too_short}")
    print(f"  too long       : {report.dropped_too_long}")
    print(f"  acknowledgments: {report.dropped_ack}")
    print(f"  duplicates     : {report.dropped_duplicates}")
    if report.noise_reasons:
        print("Noise breakdown:")
        for reason, count in sorted(report.noise_reasons.items(), key=lambda item: -item[1]):
            print(f"  {reason}: {count}")
