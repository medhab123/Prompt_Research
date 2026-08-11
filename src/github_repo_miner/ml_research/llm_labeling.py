"""LLM-based action_type labeling using a free-tier LLM (Gemini or Groq).

`action_labeling.py`'s rule-based labels are cheap but brittle keyword
matching. This asks an LLM to read the prompt + response + metadata and
assign one of the six action_type categories directly, with a confidence
score and a one-line rationale — the "LLM-assisted weak supervision" layer
the research spec's Dataset Strategy calls for, on top of (not replacing)
the "human validation" step, which still needs an actual human.

Both providers have free tiers but tight rate limits, so this is built to
survive a long, slow, interruptible run: content-hash caching (skip rows
already labeled), periodic checkpointing, and per-row exception isolation —
same lessons learned from the mining pipeline's extraction run.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ACTION_TYPES = ["implement", "debug", "explain", "review", "tool_only", "other"]

ACTION_DEFINITIONS = {
    "implement": "The agent writes substantial code, modifies files, or creates new functionality.",
    "debug": "The agent diagnoses failures, interprets errors, or proposes fixes.",
    "explain": "The agent explains concepts, code behavior, or provides guidance without major implementation.",
    "review": "The agent evaluates, critiques, or suggests improvements to existing code.",
    "tool_only": "The agent primarily performs tool-based actions such as searching, inspecting files, or navigating repositories.",
    "other": "Short, ambiguous, incomplete, or unclear interactions.",
}

_SYSTEM_INSTRUCTIONS = f"""You are labeling turns from real developer-to-AI-coding-agent conversations
for a research dataset. Given a developer prompt, the agent's response, and
some response metadata, classify what the agent's PRIMARY action was into
exactly one of these six categories:

{chr(10).join(f"- {k}: {v}" for k, v in ACTION_DEFINITIONS.items())}

Rules:
- Pick exactly one category, even if the turn has elements of several — pick the dominant one.
- Base your label on what the agent actually did in its response, not just what the developer asked for.
- "other" is for genuinely short/ambiguous/incomplete turns, not a catch-all for uncertainty — if you're
  unsure between two specific categories, pick the more likely one and lower your confidence instead.
- Respond with ONLY the JSON object, no other text.
"""

_PROMPT_TRUNCATE = 1500
_RESPONSE_TRUNCATE = 2000


@dataclass
class LabelResult:
    action_type: str
    confidence: float
    rationale: str
    raw: str = ""
    error: str = ""


def _truncate(text: str, max_chars: int) -> str:
    text = "" if pd.isna(text) else str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"...[truncated, {len(text)} chars total]"


def _row_hash(prompt_text: str, agent_response: str) -> str:
    key = f"{prompt_text}||{agent_response}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def select_exemplars(df: pd.DataFrame, n_per_class: int = 1, label_col: str = "action_type") -> list[dict]:
    """Pull real, clearly-labeled rows from the corpus as few-shot exemplars.

    Grounds the LLM in this dataset's actual style (SpecStory turns) instead
    of generic made-up examples. Prefers rows with a real agent_response so
    the example demonstrates the full prompt+response+metadata judgment.
    """
    exemplars = []
    has_response = df["agent_response"].fillna("").str.len() > 40 if "agent_response" in df.columns else pd.Series([True] * len(df))
    for action in ACTION_TYPES:
        pool = df[(df[label_col] == action) & has_response]
        if pool.empty:
            pool = df[df[label_col] == action]
        if pool.empty:
            continue
        for _, row in pool.head(n_per_class).iterrows():
            exemplars.append(
                {
                    "prompt_text": _truncate(row.get("prompt_text", ""), 400),
                    "agent_response": _truncate(row.get("agent_response", ""), 400),
                    "has_generated_code": bool(row.get("has_generated_code", False)),
                    "has_tool_use": bool(row.get("has_tool_use", False)),
                    "action_type": action,
                }
            )
    return exemplars


def _format_exemplars(exemplars: list[dict]) -> str:
    if not exemplars:
        return ""
    blocks = ["Examples from this dataset:\n"]
    for ex in exemplars:
        blocks.append(
            f"Prompt: {ex['prompt_text']!r}\n"
            f"Response: {ex['agent_response']!r}\n"
            f"Metadata: has_generated_code={ex['has_generated_code']}, has_tool_use={ex['has_tool_use']}\n"
            f"-> {{\"action_type\": \"{ex['action_type']}\", \"confidence\": 0.9, \"rationale\": \"...\"}}\n"
        )
    return "\n".join(blocks)


def build_row_prompt(
    prompt_text: str,
    agent_response: str,
    *,
    has_generated_code: bool = False,
    has_tool_use: bool = False,
    code_block_count: int = 0,
    exemplars_text: str = "",
) -> str:
    return f"""{_SYSTEM_INSTRUCTIONS}

{exemplars_text}
Now classify this turn:

Developer prompt: {_truncate(prompt_text, _PROMPT_TRUNCATE)!r}
Agent response: {_truncate(agent_response, _RESPONSE_TRUNCATE)!r}
Metadata: has_generated_code={has_generated_code}, has_tool_use={has_tool_use}, code_block_count={code_block_count}

Respond with only this JSON object:
{{"action_type": "<one of {ACTION_TYPES}>", "confidence": <0.0-1.0>, "rationale": "<one sentence>"}}"""


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_response(raw: str) -> LabelResult:
    """Parse the model's JSON reply. Anything short of a real, well-formed
    answer is flagged via `error` rather than silently defaulting to
    action_type="other" — a degenerate `{}` (missing keys, seen under load
    from smaller/free-tier models) must NOT be indistinguishable from a
    genuine "other" classification, or it silently poisons the dataset.
    """
    match = _JSON_RE.search(raw)
    if not match:
        return LabelResult(action_type="other", confidence=0.0, rationale="", raw=raw, error="no_json_found")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return LabelResult(action_type="other", confidence=0.0, rationale="", raw=raw, error=f"json_error:{exc}")

    if "action_type" not in data:
        return LabelResult(action_type="other", confidence=0.0, rationale="", raw=raw, error="missing_action_type_key")

    action = str(data.get("action_type", "")).strip().lower()
    if action not in ACTION_TYPES:
        return LabelResult(action_type="other", confidence=0.0, rationale="", raw=raw, error=f"invalid_action_type:{action}")
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    rationale = str(data.get("rationale", ""))[:300]
    return LabelResult(action_type=action, confidence=confidence, rationale=rationale, raw=raw)


# --- provider clients -------------------------------------------------


class GeminiLabeler:
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        from google import genai

        self.client = genai.Client(api_key=api_key)
        self.model = model

    def label(self, prompt: str) -> LabelResult:
        from google.genai import types

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )
        return _parse_response(response.text or "")


class GroqLabeler:
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        from groq import Groq

        self.client = Groq(api_key=api_key)
        self.model = model

    def label(self, prompt: str) -> LabelResult:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        return _parse_response(completion.choices[0].message.content or "")


def resolve_provider(provider: str | None = None) -> tuple[str, str]:
    """Pick a provider from an explicit choice or whichever key is in the environment."""
    import os

    if provider == "gemini" or (provider is None and (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))):
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY (or GOOGLE_API_KEY) not set in environment/.env")
        return "gemini", key
    if provider == "groq" or (provider is None and os.getenv("GROQ_API_KEY")):
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("GROQ_API_KEY not set in environment/.env")
        return "groq", key
    raise ValueError(
        "No LLM provider available. Set GEMINI_API_KEY (or GOOGLE_API_KEY) or GROQ_API_KEY in .env, "
        "or pass --provider explicitly."
    )


def make_labeler(provider: str, api_key: str, model: str | None = None):
    if provider == "gemini":
        return GeminiLabeler(api_key, model=model or "gemini-2.0-flash")
    if provider == "groq":
        return GroqLabeler(api_key, model=model or "llama-3.3-70b-versatile")
    raise ValueError(f"Unknown provider: {provider}")


# --- cache + batch runner ----------------------------------------------


def load_label_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_label_cache(path: Path, cache: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


def label_dataframe(
    df: pd.DataFrame,
    *,
    provider: str | None = None,
    model: str | None = None,
    cache_path: Path,
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 25,
    rate_limit_delay: float = 3.0,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Label every row of df; returns df with llm_action_type/llm_confidence/llm_rationale columns."""
    provider_name, api_key = resolve_provider(provider)
    labeler = make_labeler(provider_name, api_key, model=model)
    print(f"Using provider={provider_name} model={model or '(default)'}")

    exemplars = select_exemplars(df) if "action_type" in df.columns else []
    exemplars_text = _format_exemplars(exemplars)

    cache = load_label_cache(cache_path)
    out = df.copy()
    out["llm_action_type"] = ""
    out["llm_confidence"] = 0.0
    out["llm_rationale"] = ""
    out["llm_error"] = ""

    n = len(out)
    labeled_this_run = 0
    rows_completed = 0
    for i, (idx, row) in enumerate(out.iterrows(), start=1):
        prompt_text = str(row.get("prompt_text", ""))
        agent_response = str(row.get("agent_response", ""))
        key = _row_hash(prompt_text, agent_response)

        if key in cache:
            cached = cache[key]
            out.at[idx, "llm_action_type"] = cached["action_type"]
            out.at[idx, "llm_confidence"] = cached["confidence"]
            out.at[idx, "llm_rationale"] = cached["rationale"]
            rows_completed = i
            continue

        row_prompt = build_row_prompt(
            prompt_text,
            agent_response,
            has_generated_code=bool(row.get("has_generated_code", False)),
            has_tool_use=bool(row.get("has_tool_use", False)),
            code_block_count=int(row.get("code_block_count", 0) or 0),
            exemplars_text=exemplars_text,
        )

        result = None
        quota_exhausted = False
        for attempt in range(max_retries):
            try:
                candidate = labeler.label(row_prompt)
            except Exception as exc:  # noqa: BLE001 - a bad row must not kill a multi-hour run
                # A per-day token quota (as opposed to a per-minute rate limit) doesn't
                # recover within seconds — retrying just burns through remaining rows
                # producing nothing but failures. Bail out of the whole run instead.
                msg = str(exc).lower()
                if "per day" in msg or "tpd" in msg or "daily" in msg:
                    quota_exhausted = True
                    print(f"  [{i}/{n}] daily quota exhausted: {exc!r}")
                    break
                wait = rate_limit_delay * (2**attempt)
                print(f"  [{i}/{n}] error (attempt {attempt + 1}/{max_retries}): {exc!r} — waiting {wait:.0f}s")
                time.sleep(wait)
                continue

            if candidate.error:
                # Got a response, but it didn't parse into a real label (e.g. a
                # degenerate {} under load) — retry rather than silently accepting
                # a fake "other" as if the model had actually classified the row.
                wait = rate_limit_delay * (2**attempt)
                print(f"  [{i}/{n}] malformed response (attempt {attempt + 1}/{max_retries}): {candidate.error} — retrying in {wait:.0f}s")
                time.sleep(wait)
                continue

            result = candidate
            break

        if quota_exhausted:
            print(f"Stopping at row {i}/{n} — daily quota exhausted. Already-labeled rows are cached; "
                  f"rerun later (same command) or with a different --model to pick up where this left off.")
            break

        if result is None:
            # Do NOT cache this — it's a transient failure, not a real label. Caching it
            # would make a rerun treat this row as "already done" and skip it forever.
            out.at[idx, "llm_action_type"] = "other"
            out.at[idx, "llm_confidence"] = 0.0
            out.at[idx, "llm_rationale"] = ""
            out.at[idx, "llm_error"] = "failed_after_retries"
            rows_completed = i
            time.sleep(rate_limit_delay)
            continue

        out.at[idx, "llm_action_type"] = result.action_type
        out.at[idx, "llm_confidence"] = result.confidence
        out.at[idx, "llm_rationale"] = result.rationale
        out.at[idx, "llm_error"] = result.error
        cache[key] = {"action_type": result.action_type, "confidence": result.confidence, "rationale": result.rationale}
        labeled_this_run += 1
        rows_completed = i

        if i % 10 == 0 or i == n:
            print(f"  [{i}/{n}] labeled (this run: {labeled_this_run}, from cache: {i - labeled_this_run})")

        if checkpoint_path and checkpoint_every and i % checkpoint_every == 0:
            out.iloc[:i].to_csv(checkpoint_path, index=False, encoding="utf-8-sig")
            save_label_cache(cache_path, cache)

        time.sleep(rate_limit_delay)

    save_label_cache(cache_path, cache)
    return out.iloc[:rows_completed].reset_index(drop=True) if rows_completed < n else out


def cohen_kappa(labels_a: pd.Series, labels_b: pd.Series) -> float:
    from sklearn.metrics import cohen_kappa_score

    return float(cohen_kappa_score(labels_a, labels_b, labels=ACTION_TYPES))


def build_human_review_sheet(df: pd.DataFrame, n_per_class: int = 40, random_state: int = 42) -> pd.DataFrame:
    """Stratified sample for a human to actually adjudicate LLM vs rule-based labels."""
    parts = []
    for action in ACTION_TYPES:
        pool = df[df["llm_action_type"] == action]
        n = min(n_per_class, len(pool))
        if n:
            parts.append(pool.sample(n=n, random_state=random_state))
    sample = pd.concat(parts, ignore_index=True) if parts else df.head(0)
    cols = [
        c
        for c in [
            "full_name",
            "prompt_text",
            "agent_response",
            "action_type",
            "llm_action_type",
            "llm_confidence",
            "llm_rationale",
        ]
        if c in sample.columns
    ]
    sheet = sample[cols].copy()
    sheet["human_label"] = ""
    sheet["human_notes"] = ""
    return sheet
