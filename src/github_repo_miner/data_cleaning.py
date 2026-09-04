"""Clean and filter mined SpecStory prompt datasets."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
_CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")

_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_THINK_BLOCK_RE = re.compile(r"<think\b.*?</think>", re.DOTALL | re.IGNORECASE)
_TOOL_BLOCK_RE = re.compile(r"<tool-use\b.*?</tool-use>", re.DOTALL | re.IGNORECASE)
_XML_TAG_RE = re.compile(r"</?(?:command-name|command-message|command-args|local-command-\w+)[^>]*>", re.IGNORECASE)
_SEPARATOR_LINES = {"---", "***", "___"}
_PATH_PROMPT_RE = re.compile(
    r"^(?:[A-Za-z]:\\|/|\./|\.\./).+\.(?:tsx?|jsx?|py|ts|js|md|json|yaml|yml)\b",
    re.IGNORECASE,
)

_NOISE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("local_command_xml", re.compile(r"<local-command|</local-command|<command-name>|<command-message>", re.I)),
    ("local_command_caveat", re.compile(r"DO NOT respond to these messages|local-command-caveat", re.I)),
    ("request_interrupted", re.compile(r"^\[?request interrupted by user", re.I)),
    ("slash_command", re.compile(r"^/\w+(?:\s|$)", re.I)),
    ("agent_action_artifact", re.compile(r'^@(?:agent|cloud|workspace|copilot)\s+\S+:\s*"', re.I)),
    ("ci_runner_log", re.compile(r"/home/runner/work/|hostedtoolcache|::error::|goreleaser-action@", re.I)),
    ("browser_console_log", re.compile(
        r"react-dom_client\.js|hot-reloader-client\.js|performWorkOnRoot|"
        r"Download the React DevTools|\[Fast Refresh\]|dispatchDiscreteEvent|"
        r"chunk-[A-Z0-9]+\.js\?v=|Uncaught (?:Error|ReferenceError|TypeError|RangeError)|"
        r"Maximum update depth exceeded|(?:\.tsx?|\.jsx?):\d+\s+Uncaught",
        re.I,
    )),
    ("stack_trace_dump", re.compile(r"(?:\bat\s+\S+\s+\([^)]+\.(?:tsx?|jsx?):\d+\)\n){5,}", re.I)),
    ("agent_template_dump", re.compile(r"^\s*(?:VAN QA|Response:\s*OK VAN QA|BEGINNING TECHNICAL VALIDATION)", re.I)),
    ("ide_diagnostic_dump", re.compile(
        r'^\s*\[\s*\{\s*(?:\r?\n\s*)?"(?:resource|owner|severity|message)"\s*:',
        re.I,
    )),
]

_ACK_LEAD_RE = re.compile(
    r"^(?:ok(?:ay)?|k|yes|yep|yeah|yup|nope|no|thank you|thanks?|done|sure|cool|nice|great|"
    r"perfect|awesome|good|fine|alright|lgtm|looks good|sounds (?:good|reasonable)|well|"
    r"approved|got it|understood|i installed it|i added the token|upon review,? looks good)\b"
    r"[,.!\s]*(?:let'?s\s+|lets\s+)?",
    re.I,
)
_ACK_TRAIL_RE = re.compile(
    r"(?:[,.!\s]+(?:please|pls|plz|thanks?|thank you|for now|then|too|buddy))*[,.!\s]*$",
    re.I,
)
_ACK_CORE_RE = re.compile(
    r"^(?:continue|proceed|keep going|keep changes and proceed|go ahead|go for it|go on|"
    r"next (?:step|checklist item|checklist tier)|do it|make it so|ship it|move on|start|begin|"
    r"retry|try again|i guess)?$",
    re.I,
)


def _is_acknowledgment(text: str) -> bool:
    """True for confirmation/continuation messages that carry no independent intent.

    Strips a leading interjection (ok/yes/sounds good/...) and trailing filler
    (please/thanks/...), then checks whether anything topical is left. A bare
    "Ok, continue." or "Sounds good, go ahead!" is dropped; "Ok, use PyQt6 and
    continue." is kept because "use PyQt6" survives the strip.
    """
    core = _ACK_TRAIL_RE.sub("", _ACK_LEAD_RE.sub("", text, count=1)).strip()
    return bool(_ACK_CORE_RE.match(core))


_DERIVED_TEXT_COLS = ("prompt_length", "prompt_word_count")


@dataclass
class CleaningConfig:
    """Thresholds and toggles for prompt dataset cleaning."""

    min_chars: int = 25
    min_words: int = 4
    min_chars_cjk: int = 18
    min_cjk_chars: int = 8
    max_chars: int = 8_000
    drop_noise_patterns: bool = True
    drop_acknowledgments: bool = True
    drop_duplicates: bool = True
    dedupe_subset: tuple[str, ...] = ("prompt_text",)
    # Repository skew control: a handful of chatty repos otherwise dominate the
    # corpus (e.g. 50%+ of rows from 5 repos). None disables the cap.
    max_prompts_per_repo: int | None = None
    repo_column: str = "full_name"
    repo_cap_seed: int = 42


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
    dropped_repo_cap: int = 0
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
            "dropped_repo_cap": self.dropped_repo_cap,
            "noise_reasons": dict(self.noise_reasons),
        }


def _has_cjk(text: str) -> bool:
    return bool(CJK_RE.search(text))


def _count_cjk_chars(text: str) -> int:
    return len(_CJK_CHAR_RE.findall(text))


def _is_ide_diagnostic_dump(text: str) -> bool:
    stripped = text.strip()
    if not stripped.startswith("[{"):
        return False
    markers = ('"resource"', '"owner"', '"severity"', '"startLineNumber"', '"modelVersionId"')
    return sum(marker in stripped for marker in markers) >= 3


def _is_path_heavy_prompt(text: str) -> bool:
    stripped = text.strip()
    if not _PATH_PROMPT_RE.match(stripped):
        return False
    without_paths = re.sub(
        r"(?:[A-Za-z]:)?[\\/][\w./\\\-:]+|"
        r"\b[\w./\\\-]+\.(?:tsx?|jsx?|py|ts|js|md|json|yaml|yml)\b",
        " ",
        stripped,
        flags=re.IGNORECASE,
    )
    return len(without_paths.strip()) < 12


def sanitize_turn_text(text: str) -> str:
    """Normalize a SpecStory turn during extraction (not dataset filtering)."""
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)

    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\ufeff", "")
    text = _HTML_COMMENT_RE.sub("", text)
    text = _THINK_BLOCK_RE.sub("", text)
    text = _TOOL_BLOCK_RE.sub("", text)
    text = _XML_TAG_RE.sub("", text)
    kept = [ln for ln in text.splitlines() if ln.strip() not in _SEPARATOR_LINES]
    text = "\n".join(kept)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def sanitize_prompt_text(text: str) -> str:
    """Normalize whitespace and strip known SpecStory / agent markup."""
    return sanitize_turn_text(text)


def classify_noise(text: str, *, config: CleaningConfig | None = None) -> str | None:
    """Return a drop reason when text is clearly not a developer prompt."""
    config = config or CleaningConfig()
    stripped = text.strip()
    if not stripped:
        return "empty"

    if config.drop_acknowledgments and _is_acknowledgment(stripped):
        return "acknowledgment"

    if _is_ide_diagnostic_dump(stripped):
        return "ide_diagnostic_dump"

    if _is_path_heavy_prompt(stripped):
        return "path_only_prompt"

    if config.drop_noise_patterns:
        for reason, pattern in _NOISE_PATTERNS:
            if pattern.search(stripped):
                return reason

    char_len = len(stripped)
    word_len = len(stripped.split())

    if char_len > config.max_chars:
        return "too_long"

    if _has_cjk(stripped):
        if char_len < config.min_chars_cjk or _count_cjk_chars(stripped) < config.min_cjk_chars:
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


_STRATA_CANDIDATE_COLS = ("action_type", "primary_intent")


def _pick_strata_column(df: pd.DataFrame) -> str | None:
    """Prefer an existing intent/action label to stratify the repo cap by."""
    for col in _STRATA_CANDIDATE_COLS:
        if col in df.columns and df[col].notna().any():
            return col
    return None


def _synthetic_strata(df: pd.DataFrame) -> pd.Series:
    """Fallback diversity signal when no intent label exists yet: the
    code/tool-use combination already captured at extraction time. Coarse,
    but still keeps a repo's mix of "wrote code" / "used tools" / "talked only"
    turns intact instead of sampling those away at random.
    """
    code = df["has_generated_code"].astype(bool) if "has_generated_code" in df.columns else False
    tools = df["has_tool_use"].astype(bool) if "has_tool_use" in df.columns else False
    return pd.Series(code, index=df.index).astype(int).astype(str) + "_" + pd.Series(tools, index=df.index).astype(int).astype(str)


def _allocate_proportional(counts: pd.Series, total: int) -> pd.Series:
    """Largest-remainder allocation of `total` items across strata sizes.

    Keeps the kept subset's stratum mix proportional to the original mix
    (instead of a flat n-per-stratum split, which would over-represent rare
    strata and under-represent common ones), while never allocating more to
    a stratum than it actually has.
    """
    if total <= 0 or counts.sum() == 0:
        return pd.Series(0, index=counts.index, dtype=int)

    raw = counts / counts.sum() * total
    base = raw.apply(lambda x: int(x)).clip(upper=counts)
    remainder = total - int(base.sum())

    if remainder > 0:
        room = counts - base
        frac = (raw - base).where(room > 0, -1.0)
        order = frac.sort_values(ascending=False).index
        for stratum in order:
            if remainder <= 0:
                break
            if room[stratum] > 0:
                base[stratum] += 1
                remainder -= 1

    return base.astype(int)


def _cap_repo_group(group: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    """Downsample one repo's rows to `cap`, preserving its intent mix.

    Rather than a flat random sample (which can wipe out a repo's only
    "debug" or "review" examples by chance), rows are kept proportionally
    from each intent/action stratum present, so the retained subset still
    reflects the full range of developer intent seen in that repo.
    """
    if len(group) <= cap:
        return group

    strata_col = _pick_strata_column(group)
    if strata_col is not None:
        strata = group[strata_col]
    else:
        strata = _synthetic_strata(group)

    counts = strata.value_counts()
    alloc = _allocate_proportional(counts, cap)

    parts = [
        group[strata == stratum].sample(n=n, random_state=seed)
        for stratum, n in alloc.items()
        if n > 0
    ]
    return pd.concat(parts)


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

    if config.max_prompts_per_repo is not None and not cleaned.empty and config.repo_column in cleaned.columns:
        repo_col = config.repo_column
        cap = config.max_prompts_per_repo
        sizes = cleaned.groupby(repo_col)[repo_col].transform("size")
        over_mask = sizes > cap
        if over_mask.any():
            before_cap = len(cleaned)
            over_df = cleaned[over_mask]
            capped_parts = [
                _cap_repo_group(group, cap, config.repo_cap_seed)
                for _, group in over_df.groupby(repo_col, sort=False)
            ]
            capped_over = pd.concat(capped_parts)
            cleaned = pd.concat([cleaned[~over_mask], capped_over]).sort_index().reset_index(drop=True)
            report.dropped_repo_cap = before_cap - len(cleaned)

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
    print(f"  repo cap       : {report.dropped_repo_cap}")
    if report.noise_reasons:
        print("Noise breakdown:")
        for reason, count in sorted(report.noise_reasons.items(), key=lambda item: -item[1]):
            print(f"  {reason}: {count}")
