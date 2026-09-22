"""Repair the canonical corpus in place for two extraction bugs found 2026-09-17.

1. Failed translations: 225 prompts whose `prompt_text` is Google's
   "Error 500 (Server Error)" page instead of a translation. Poisoned entries
   are purged from the translation cache and the originals re-translated (the
   translator now rejects error pages; a prompt that still fails keeps its
   original text with was_translated=False).
2. `has_tool_use` missed older SpecStory/Cursor tool-call formats. The flag is
   backfilled from the tool traces that survive in `agent_response`, then
   `action_type` is recomputed as the pipeline does (rule labels on the
   pre-translation prompt) and `prev_turn_had_tools` / `prev_action_type` are
   updated wherever the previous turn is in the corpus.

The raw SpecStory files are not kept locally, so this patches the existing rows
rather than re-extracting; the row set and order are unchanged. The input is
backed up first. `llm_labels_full.csv` is NOT touched — its LLM labels were
produced from the old inputs and are refreshed by relabeling.

Usage:
    python scripts/repair_corpus.py            # dry run: report only
    python scripts/repair_corpus.py --write
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from github_repo_miner.action_labeling import add_action_labels
from github_repo_miner.specstory_parser import text_has_tool_use
from github_repo_miner.translate_dataset import (
    _text_hash,
    load_translation_cache,
    looks_like_failed_translation,
    save_translation_cache,
    translate_preserving_code,
)

DEFAULT_CORPUS = ROOT / "outputs/full_run_v3_analysis/specstory_prompts_cleaned.csv"
SESSION_COLS = ["prev_turn_had_tools", "prev_action_type"]
BACKUP_DIR = ROOT / "outputs/provenance/corpus_pre_repair_2026-09-17"
PER_ROW_SLEEP_SEC = 1.0  # well under Google's 5 req/s; the original burst got rate-limited


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("corpus_csv", type=Path, nargs="?", default=DEFAULT_CORPUS)
    parser.add_argument("--write", action="store_true", help="Back up and overwrite the corpus (default: dry run).")
    parser.add_argument("--skip-translation", action="store_true", help="Only repair tool-use flags and labels.")
    parser.add_argument(
        "--translator", choices=["google", "gemini"], default="google",
        help="Backend for re-translating the failed prompts. 'gemini' is a fallback for when Google's free "
             "endpoint keeps refusing requests; rows it translates are listed in the provenance folder.",
    )
    return parser.parse_args()


def use_gemini_translator(model: str = "gemini-3.1-flash-lite") -> None:
    """Swap translate_dataset's backend for Gemini. Code blocks are still replaced by
    placeholders before the call and restored after, exactly as with Google."""
    import os

    from dotenv import load_dotenv
    from google import genai
    from google.genai import types

    import github_repo_miner.translate_dataset as td

    load_dotenv(ROOT / ".env")
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    instructions = (
        "Translate the text below into English. Output only the translation, nothing else. "
        "Keep placeholders such as __CODE_BLOCK_0__, file paths, identifiers, URLs, commands and any "
        "code exactly as they are. If part of the text is already English, keep it unchanged.\n\nText:\n"
    )

    def _translate_plain(text: str, *, target: str = "en") -> str:
        out = client.models.generate_content(
            model=model, contents=instructions + text, config=types.GenerateContentConfig(temperature=0.0)
        ).text
        if td.looks_like_failed_translation(text, out):
            raise td.TranslationError("gemini returned empty or error output")
        return out.strip()

    td._translate_plain = _translate_plain


def repair_translations(
    df: pd.DataFrame, cache_path: Path, backup_csv: Path, *, write: bool, translator: str = "google"
) -> pd.DataFrame:
    cache = load_translation_cache(cache_path)
    original = df["prompt_text_original"].fillna("").astype(str)
    # On a re-run the error pages are already replaced by the original text, so
    # also take the affected rows from the pre-repair backup (same row order).
    source = pd.read_csv(backup_csv, usecols=["prompt_text"])["prompt_text"] if backup_csv.exists() else df["prompt_text"]
    bad = [
        i for i in df.index
        if looks_like_failed_translation(original[i], str(df.at[i, "prompt_text"]))
        or (looks_like_failed_translation(original[i], str(source[i])) and df.at[i, "prompt_text"] == original[i])
    ]
    print(f"[translation] rows needing re-translation: {len(bad)}")

    # Only Google's error page itself; other markers (e.g. "<html") also occur in
    # legitimate translations of prompts that quote HTML.
    poisoned = [k for k, v in cache.items() if re.search(r"That[’']s an error", v)]
    print(f"[translation] error-page entries in {cache_path.name}: {len(poisoned)}")
    if not write:
        return df

    for key in poisoned:
        del cache[key]
    for i in bad:
        cache.pop(_text_hash(original[i]), None)

    fixed = failed = streak = 0
    for n, i in enumerate(bad, 1):
        text, did = translate_preserving_code(original[i], cache)
        if did and not looks_like_failed_translation(original[i], text):
            df.at[i, "prompt_text"] = text
            fixed += 1
            streak = 0
        else:
            df.at[i, "prompt_text"] = original[i]
            failed += 1
            streak += 1
        if streak >= 10:
            # Almost certainly rate-limited (TooManyRequests); stop instead of hammering.
            # Remaining rows keep their original text — re-run this script later.
            failed += len(bad) - n
            print(f"  10 consecutive failures at {n}/{len(bad)} — likely rate-limited; stopping. Re-run later.")
            for j in bad[n:]:
                df.at[j, "prompt_text"] = original[j]
            break
        time.sleep(PER_ROW_SLEEP_SEC)
        if n % 25 == 0:
            save_translation_cache(cache_path, cache)
            print(f"  {n}/{len(bad)} re-translated")
    save_translation_cache(cache_path, cache)
    df["was_translated"] = df["prompt_text_original"].fillna("") != df["prompt_text"].fillna("")
    print(f"[translation] re-translated: {fixed}; still failing (kept original text): {failed}")
    if translator != "google" and fixed:
        log = BACKUP_DIR / f"retranslated_with_{translator}.csv"
        done = [i for i in bad if df.at[i, "prompt_text"] != original[i]]
        rows = df.loc[done, ["full_name", "source_file", "turn_index"]].assign(translator=translator)
        if log.exists():
            rows = pd.concat([pd.read_csv(log), rows]).drop_duplicates(["full_name", "source_file", "turn_index"], keep="last")
        rows.to_csv(log, index=False, encoding="utf-8-sig")
        print(f"[translation] rows translated with {translator} listed in {log}")
    return df


def repair_tool_flags(df: pd.DataFrame) -> pd.DataFrame:
    old = df[["has_tool_use", "action_type", *SESSION_COLS]].copy()
    detected = df["agent_response"].map(text_has_tool_use)
    df["has_tool_use"] = df["has_tool_use"].astype(bool) | detected
    print(f"[tools] has_tool_use True: {int(old['has_tool_use'].sum())} -> {int(df['has_tool_use'].sum())}")

    # Rule labels are computed on the pre-translation prompt, as in the pipeline.
    as_extracted = df.assign(prompt_text=df["prompt_text_original"].where(df["prompt_text_original"].notna(), df["prompt_text"]))
    df["action_type"] = add_action_labels(as_extracted)["action_type"].values

    # Session features were computed on the full extraction, *before* cleaning
    # dropped rows (1,848 of 3,197 sessions have turn_index gaps), so prev_* point
    # at the true previous turn. Re-running add_session_features here would
    # silently redefine them as "previous kept turn". Instead, update prev_* only
    # where that true previous turn is still in the corpus; the rest keep their
    # stored values (the dropped turn's text isn't available to re-check).
    by_turn = df.set_index(["session_id", "turn_index"])[["has_tool_use", "action_type"]]
    prev_keys = pd.MultiIndex.from_arrays([df["session_id"], df["turn_index"] - 1])
    present = prev_keys.isin(by_turn.index)
    prev = by_turn.reindex(prev_keys[present])
    df.loc[present, "prev_turn_had_tools"] = prev["has_tool_use"].astype(bool).values
    df.loc[present, "prev_action_type"] = prev["action_type"].values
    has_prev = df["turns_before"] > 0
    print(f"[session] turns with a previous turn: {int(has_prev.sum())}; previous turn in corpus: "
          f"{int((present & has_prev).sum())}; previous turn dropped by cleaning (kept stored prev_*): "
          f"{int((~present & has_prev).sum())}")

    changed = old["action_type"].values != df["action_type"].values
    print(f"[labels] action_type changed on {int(changed.sum())} rows")
    print(pd.crosstab(old["action_type"].values[changed], df["action_type"].values[changed], rownames=["before"], colnames=["after"]))
    print("[labels] distribution after:", df["action_type"].value_counts().to_dict())
    for col in SESSION_COLS:
        diff = old[col].fillna("").astype(str).values != df[col].fillna("").astype(str).values
        print(f"[session] {col} changed on {int(diff.sum())} rows")
    return df


def main() -> None:
    args = parse_args()
    corpus = args.corpus_csv
    df = pd.read_csv(corpus)
    columns = list(df.columns)
    print(f"Loaded {len(df)} rows from {corpus}")

    cache_path = corpus.parent / ".translation_cache.json"
    backup_dir = BACKUP_DIR
    if args.write:
        # Back up before anything is written (re-translation saves the cache as it goes).
        backup_dir.mkdir(parents=True, exist_ok=True)
        for src in (corpus, cache_path):
            dst = backup_dir / src.name
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
        print(f"Backed up originals to {backup_dir}")

    if not args.skip_translation:
        if args.translator == "gemini":
            use_gemini_translator()
        df = repair_translations(df, cache_path, backup_dir / corpus.name, write=args.write, translator=args.translator)
    df = repair_tool_flags(df)

    if not args.write:
        print("\nDry run — nothing written. Re-run with --write to back up and overwrite.")
        return

    df[columns].to_csv(corpus, index=False, encoding="utf-8-sig")
    print(f"\nWrote {corpus}")


if __name__ == "__main__":
    main()
