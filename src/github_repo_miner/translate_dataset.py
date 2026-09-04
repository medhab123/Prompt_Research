"""Translate dataset text columns to English for human-readable exports."""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

from .data_cleaning import CJK_RE

# Columns translated for the English view export.
DEFAULT_TRANSLATE_COLUMNS = ("prompt_text", "agent_response")

# Code blocks are preserved verbatim inside agent responses.
_CODE_FENCE_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)

_CACHE_VERSION = 1
_MAX_CHUNK_CHARS = 3500
_TRANSLATE_SLEEP_SEC = 0.15


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_translation_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != _CACHE_VERSION:
            return {}
        return payload.get("entries", {})
    except Exception:
        return {}


def save_translation_cache(path: Path, cache: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": _CACHE_VERSION, "entries": cache}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def needs_translation(text: str) -> bool:
    """Return True if text is likely non-English."""
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    if len(stripped) < 2 or stripped.lower() == "nan":
        return False

    if CJK_RE.search(stripped):
        return True

    # Heuristic for other non-Latin scripts.
    latin = sum(1 for ch in stripped if ch.isascii())
    if latin / max(len(stripped), 1) < 0.85:
        return True

    try:
        from langdetect import detect

        sample = stripped[:800]
        return detect(sample) != "en"
    except Exception:
        return False


def _chunk_text(text: str, max_chars: int = _MAX_CHUNK_CHARS) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for paragraph in re.split(r"(\n\n+)", text):
        if current_len + len(paragraph) > max_chars and current:
            chunks.append("".join(current))
            current = [paragraph]
            current_len = len(paragraph)
        else:
            current.append(paragraph)
            current_len += len(paragraph)

    if current:
        chunks.append("".join(current))
    return chunks


def _translate_plain(text: str, *, target: str = "en") -> str:
    from deep_translator import GoogleTranslator

    translator = GoogleTranslator(source="auto", target=target)
    chunks = _chunk_text(text)
    if len(chunks) == 1:
        return translator.translate(chunks[0])

    translated: list[str] = []
    for chunk in chunks:
        translated.append(translator.translate(chunk))
        time.sleep(_TRANSLATE_SLEEP_SEC)
    return "".join(translated)


def translate_preserving_code(text: str, cache: dict[str, str]) -> tuple[str, bool]:
    """
    Translate prose to English; leave fenced code blocks unchanged.
    Returns (translated_text, was_translated).
    """
    if not needs_translation(text):
        return text, False

    key = _text_hash(text)
    if key in cache:
        return cache[key], True

    placeholders: dict[str, str] = {}

    def _stash(match: re.Match[str]) -> str:
        token = f"__CODE_BLOCK_{len(placeholders)}__"
        placeholders[token] = match.group(0)
        return token

    stripped = text.strip()
    scaffold = _CODE_FENCE_RE.sub(_stash, stripped)

    if not needs_translation(scaffold):
        return text, False

    try:
        translated = _translate_plain(scaffold)
    except Exception:
        return text, False

    for token, code in placeholders.items():
        translated = translated.replace(token, code)

    cache[key] = translated
    time.sleep(_TRANSLATE_SLEEP_SEC)
    return translated, True


def translate_text_column(
    values: list[str],
    cache: dict[str, str],
    *,
    preserve_code: bool = False,
    show_progress: bool = True,
    cache_path: Path | None = None,
    save_every: int = 100,
) -> tuple[list[str], int]:
    """Translate a list of cell values; return translated values and count changed.

    A row of thousands can take tens of minutes (one network call per
    untranslated cell) and the cache was previously only written to disk
    once, at the very end — an interruption anywhere in the middle lost
    every translation done so far. If cache_path is given, the cache is
    persisted every `save_every` translated rows so a rerun resumes instead
    of starting over.
    """
    translated_values: list[str] = []
    changed = 0

    iterator = enumerate(values)
    if show_progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(list(iterator), desc="Translating", unit="row")
        except ImportError:
            pass

    for _, raw in iterator:
        text = "" if raw is None or (isinstance(raw, float) and str(raw) == "nan") else str(raw)
        if not text.strip() or text.strip().lower() == "nan":
            translated_values.append(text)
            continue

        if preserve_code:
            out, did = translate_preserving_code(text, cache)
        else:
            if not needs_translation(text):
                out, did = text, False
            else:
                key = _text_hash(text)
                if key in cache:
                    out, did = cache[key], True
                else:
                    try:
                        out = _translate_plain(text)
                        cache[key] = out
                        did = True
                        time.sleep(_TRANSLATE_SLEEP_SEC)
                    except Exception:
                        out, did = text, False

        if did:
            changed += 1
        translated_values.append(out)

        if cache_path is not None and did and changed % save_every == 0:
            save_translation_cache(cache_path, cache)

    if cache_path is not None:
        save_translation_cache(cache_path, cache)

    return translated_values, changed


def build_english_view_dataframe(
    df,
    *,
    columns: tuple[str, ...] = DEFAULT_TRANSLATE_COLUMNS,
    cache_path: Path | None = None,
    show_progress: bool = True,
):
    """
    Return a copy of df with selected text columns translated to English.

    Original columns are preserved; English versions are suffixed with `_en`.
    Adds `any_translated` flag column.
    """
    import pandas as pd

    out = df.copy()
    cache_file = cache_path or Path("outputs/datasets/.translation_cache.json")
    cache = load_translation_cache(cache_file)

    any_translated = [False] * len(out)
    summary: dict[str, int] = {}

    for col in columns:
        if col not in out.columns:
            continue
        en_col = f"{col}_en"
        preserve_code = col == "agent_response"
        translated, changed = translate_text_column(
            out[col].tolist(),
            cache,
            preserve_code=preserve_code,
            show_progress=show_progress and col == columns[0],
            cache_path=cache_file,
        )
        out[en_col] = translated
        summary[col] = changed
        for i, (orig, new) in enumerate(zip(out[col].astype(str), translated)):
            if orig != new:
                any_translated[i] = True

    out["any_translated"] = any_translated
    save_translation_cache(cache_file, cache)
    return out, summary


def export_english_view_csv(
    input_csv: Path,
    output_csv: Path,
    *,
    columns: tuple[str, ...] = DEFAULT_TRANSLATE_COLUMNS,
    excel_friendly: bool = True,
    max_rows: int | None = None,
    show_progress: bool = True,
) -> dict:
    """
    Create an English-readable CSV without modifying the source file.

    The export uses `_en` text columns for reading; originals stay intact.
    """
    import pandas as pd

    df = pd.read_csv(input_csv)
    if max_rows is not None:
        df = df.head(max_rows).copy()

    cache_path = output_csv.parent / ".translation_cache.json"
    enriched, summary = build_english_view_dataframe(
        df,
        columns=columns,
        cache_path=cache_path,
        show_progress=show_progress,
    )

    # Reader-friendly column order: English text near the front.
    front = [c for c in columns if f"{c}_en" in enriched.columns]
    front += [f"{c}_en" for c in columns if f"{c}_en" in enriched.columns]
    rest = [c for c in enriched.columns if c not in front]
    export_df = enriched[front + rest]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    encoding = "utf-8-sig" if excel_friendly else "utf-8"
    export_df.to_csv(output_csv, index=False, encoding=encoding)

    return {
        "input": str(input_csv),
        "output": str(output_csv),
        "rows": len(export_df),
        "translated_cells": summary,
        "rows_with_any_translation": int(enriched["any_translated"].sum()),
    }
