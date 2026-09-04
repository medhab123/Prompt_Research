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
from threading import Lock

import pandas as pd

ACTION_TYPES = ["implement", "debug", "explain", "review", "tool_only", "other"]

# Definitions below encode the working taxonomy established during the human
# adjudication pass over rule-vs-LLM disagreements (2026-08-26), not just the
# original generic descriptions — that review found the LLM systematically
# under-called `debug` (calling a diagnosed-and-fixed bug `explain` when there
# was no hard stack trace) and that `review` needed an explicit "no reported
# problem driving it" criterion to stop it bleeding into `debug`/`implement`.
ACTION_DEFINITIONS = {
    "implement": "The agent writes substantial code, modifies files, or creates new functionality.",
    "debug": (
        "The agent diagnoses a REPORTED problem (an explicit crash/exception/error/build-failure, "
        "OR a 'this isn't behaving as expected' bug report) and identifies or starts fixing a root "
        "cause. A concrete root-cause diagnosis is what matters here, NOT whether there's a stack "
        "trace — a bug fixed via reasoning about behavior (no exception) is still debug."
    ),
    "explain": "The agent explains concepts, code behavior, or provides guidance without major implementation.",
    "review": (
        "The agent checks, audits, or critiques existing code, config, or state WITHOUT a reported "
        "problem driving it — e.g. 'review the config for X', 'check these files for Y, don't change "
        "anything', verifying something looks right, auditing behavior, or reviewing/summarizing "
        "project status against a plan. If investigating a 'ran fine but result is missing/wrong' "
        "report and no concrete root cause has been found yet, lean review, not debug — it flips to "
        "debug once a root cause is identified and a fix starts."
    ),
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
- debug vs review: the deciding question is "was there a reported problem (crash, error, or 'not
  behaving as expected') driving this?" If yes and a root cause was identified/fixed, it's debug —
  regardless of whether there was a stack trace. If the agent is auditing/checking without such a
  report, it's review, even if it's evaluating code quality or correctness.
- implement vs explain: this is the fuzziest boundary. If the response contains actual code/diffs/file
  changes, lean implement. If it's a prose-only description or walkthrough with no visible code change,
  lean explain, even if it describes what a change would look like.
- "other" is for genuinely short/ambiguous/incomplete turns, not a catch-all for uncertainty — if you're
  unsure between two specific categories, pick the more likely one and lower your confidence instead.
- Respond with ONLY the JSON object, no other text.
"""

# Generous now that this runs on a billed key — the corpus's median agent_response
# is ~2000 chars and 50% of rows exceeded the old 2000-char cap entirely (some up
# to 300K chars), meaning the model was judging over half the corpus on a chopped-
# off view of what the agent actually did. Raising these costs about $1.70 extra
# across the full 8,404-row corpus (measured), which is trivial next to the
# accuracy this recovers for the implement/debug/explain boundary cases.
_PROMPT_TRUNCATE = 4000
_RESPONSE_TRUNCATE = 20000

# Bump this whenever ACTION_DEFINITIONS, _SYSTEM_INSTRUCTIONS, or the truncation
# limits change — it's folded into the cache key so stale labels from an older,
# less-accurate prompt version can never silently masquerade as already-done
# under a corrected one. This is what makes a corpus-wide prompt fix safe to ship
# without a manual cache wipe.
PROMPT_VERSION = "v2-taxonomy-fix"


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
    key = f"{PROMPT_VERSION}||{prompt_text}||{agent_response}"
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
        return GeminiLabeler(api_key, model=model or "gemini-3.1-flash-lite")
    if provider == "groq":
        return GroqLabeler(api_key, model=model or "openai/gpt-oss-120b")
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


class _RateLimiter:
    """Paces calls across ALL worker threads to a single aggregate rate.

    Concurrency without this just multiplies the effective request rate by
    max_workers, since each worker would otherwise pace itself independently
    — 4 workers each waiting `rate_limit_delay` between their own calls means
    ~4x the intended aggregate rate, which trips free-tier limits harder and
    makes things slower overall (backoff pile-up), not faster. This makes
    max_workers purely about hiding per-call latency, not about how many
    calls/second get sent — that's controlled by min_interval alone.
    """

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._lock = Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next_allowed - now)
            self._next_allowed = max(now, self._next_allowed) + self.min_interval
        if delay > 0:
            time.sleep(delay)


def _label_one_row(
    labeler,
    prompt_text: str,
    agent_response: str,
    *,
    has_generated_code: bool,
    has_tool_use: bool,
    code_block_count: int,
    exemplars_text: str,
    rate_limiter: "_RateLimiter",
    max_retries: int,
    stop_event,
) -> tuple[str, LabelResult | None]:
    """Label one row. Pure w.r.t. shared state — returns a status + result for
    the caller to apply, so cache/dataframe/file writes only ever happen on
    one thread (the consumer loop), not from inside worker threads.
    """
    if stop_event.is_set():
        return "skipped", None

    row_prompt = build_row_prompt(
        prompt_text,
        agent_response,
        has_generated_code=has_generated_code,
        has_tool_use=has_tool_use,
        code_block_count=code_block_count,
        exemplars_text=exemplars_text,
    )

    for attempt in range(max_retries):
        if stop_event.is_set():
            return "skipped", None
        rate_limiter.wait()
        try:
            candidate = labeler.label(row_prompt)
        except Exception as exc:  # noqa: BLE001 - a bad row must not kill a multi-hour run
            # A per-day token quota (as opposed to a per-minute rate limit) doesn't
            # recover within seconds — retrying just burns quota producing nothing
            # but failures. Signal every other in-flight/queued worker to stop too.
            msg = str(exc).lower()
            if "per day" in msg or "tpd" in msg or "daily" in msg:
                stop_event.set()
                return "quota_exhausted", None
            time.sleep(rate_limiter.min_interval * (2**attempt))
            continue

        if candidate.error:
            # Got a response, but it didn't parse into a real label (e.g. a
            # degenerate {} under load) — retry rather than silently accepting
            # a fake "other" as if the model had actually classified the row.
            time.sleep(rate_limiter.min_interval * (2**attempt))
            continue

        return "ok", candidate

    return "failed", None


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
    max_workers: int = 4,
) -> pd.DataFrame:
    """Label every row of df; returns df with llm_action_type/llm_confidence/llm_rationale columns.

    Labeling is one API call per row — I/O-bound, so rows are labeled
    `max_workers` at a time via a thread pool rather than one at a time.
    `rate_limit_delay` is the minimum interval between calls **in aggregate**
    across all workers (see _RateLimiter) — max_workers controls how many
    calls can be in flight at once (hides per-call latency), not how many
    calls/second get sent (that's rate_limit_delay alone). Without this
    split, N workers each pacing independently would multiply the effective
    send rate by N, tripping free-tier limits harder and making a run slower
    overall, not faster.
    A shared stop signal means a detected daily-quota exhaustion halts every
    worker promptly instead of each one independently burning through retries
    on an already-dead quota.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from threading import Event

    provider_name, api_key = resolve_provider(provider)
    labeler = make_labeler(provider_name, api_key, model=model)
    rate_limiter = _RateLimiter(rate_limit_delay)
    print(f"Using provider={provider_name} model={model or '(default)'} workers={max_workers} "
          f"min_interval={rate_limit_delay}s", flush=True)

    exemplars = select_exemplars(df) if "action_type" in df.columns else []
    exemplars_text = _format_exemplars(exemplars)

    cache = load_label_cache(cache_path)
    out = df.copy()
    out["llm_action_type"] = ""
    out["llm_confidence"] = 0.0
    out["llm_rationale"] = ""
    out["llm_error"] = ""

    to_label: list[tuple[object, str]] = []  # (row index, content-hash key)
    for idx, row in out.iterrows():
        prompt_text = str(row.get("prompt_text", ""))
        agent_response = str(row.get("agent_response", ""))
        key = _row_hash(prompt_text, agent_response)
        if key in cache:
            cached = cache[key]
            out.at[idx, "llm_action_type"] = cached["action_type"]
            out.at[idx, "llm_confidence"] = cached["confidence"]
            out.at[idx, "llm_rationale"] = cached["rationale"]
        else:
            to_label.append((idx, key))

    n_total = len(out)
    n_to_label = len(to_label)
    print(f"{n_to_label} rows to label ({n_total - n_to_label} already cached)", flush=True)

    stop_event = Event()
    completed = 0
    labeled_this_run = 0
    consecutive_failures = 0
    # A hard daily quota doesn't always say "daily" in its error text (e.g. Gemini's
    # free-tier RESOURCE_EXHAUSTED message doesn't) and can look superficially like a
    # transient rate limit ("retry in 26s") when it actually won't recover for hours.
    # Treating N fully-failed rows in a row as systemic (regardless of error text) is
    # what actually catches this — this loop is single-threaded (as_completed consumer),
    # so no lock is needed for the counter.
    consecutive_failure_limit = 10

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {}
        for idx, key in to_label:
            row = out.loc[idx]
            future = executor.submit(
                _label_one_row,
                labeler,
                str(row.get("prompt_text", "")),
                str(row.get("agent_response", "")),
                has_generated_code=bool(row.get("has_generated_code", False)),
                has_tool_use=bool(row.get("has_tool_use", False)),
                code_block_count=int(row.get("code_block_count", 0) or 0),
                exemplars_text=exemplars_text,
                rate_limiter=rate_limiter,
                max_retries=max_retries,
                stop_event=stop_event,
            )
            future_to_item[future] = (idx, key)

        for future in as_completed(future_to_item):
            idx, key = future_to_item[future]
            completed += 1
            status, result = future.result()

            if status == "quota_exhausted":
                out.at[idx, "llm_error"] = "quota_exhausted"
            elif status == "skipped":
                out.at[idx, "llm_error"] = "skipped_after_quota_exhausted"
            elif status == "failed" or result is None:
                # Leave llm_action_type blank (its initialized default) rather than
                # a fake "other" — a real classification and "the call kept failing"
                # must stay distinguishable downstream (kappa scoring, resumability).
                out.at[idx, "llm_confidence"] = 0.0
                out.at[idx, "llm_error"] = "failed_after_retries"
                consecutive_failures += 1
                if consecutive_failures >= consecutive_failure_limit and not stop_event.is_set():
                    stop_event.set()
                    print(f"\n{consecutive_failures} rows in a row failed every retry — "
                          f"treating this as a systemic/quota problem and stopping early "
                          f"instead of burning hours retrying a dead quota.", flush=True)
            else:
                out.at[idx, "llm_action_type"] = result.action_type
                out.at[idx, "llm_confidence"] = result.confidence
                out.at[idx, "llm_rationale"] = result.rationale
                out.at[idx, "llm_error"] = result.error
                # Only a genuinely parsed, successful label is cached — a failed
                # or skipped row must not look "already done" on the next run.
                cache[key] = {
                    "action_type": result.action_type,
                    "confidence": result.confidence,
                    "rationale": result.rationale,
                }
                labeled_this_run += 1
                consecutive_failures = 0

            if completed % 10 == 0 or completed == n_to_label:
                print(f"  [{completed}/{n_to_label}] labeled (this run: {labeled_this_run})", flush=True)

            if checkpoint_path and checkpoint_every and completed % checkpoint_every == 0:
                out.to_csv(checkpoint_path, index=False, encoding="utf-8-sig")
                save_label_cache(cache_path, cache)

    if stop_event.is_set():
        print("Daily quota was exhausted partway through — some rows are unlabeled (llm_error column). "
              "Already-labeled rows are cached; rerun later or with a different --model to finish the rest.")

    save_label_cache(cache_path, cache)
    return out


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
