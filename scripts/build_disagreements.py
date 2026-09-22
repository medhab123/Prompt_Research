"""Build the rule-vs-LLM disagreement set from the current corpus.

Takes every column from the canonical corpus (so it always reflects the latest
rule labels, translations and tool flags) and ONLY the LLM columns from
`llm_labels_full.csv` — that file's copies of the corpus columns predate the
2026-09-17 repair and must not be used. Rows are paired by
(full_name, source_file, turn_index). Turns whose LLM call failed are dropped.

Each disagreement is assigned one diagnostic cluster (first match wins):
  C0  empty agent response — neither label means much; set aside
  C5  broken / non-human turn (auto-generated prompt, garbled text, stub
      response, prompt that was an "Error 500" translation page, or either
      side said `other`)
  C7  regex debug -> LLM not debug: a problem word is present but the turn is
      not driven by a reported problem (or the LLM under-called debug)
  C1  regex implement -> LLM debug: bug report the regex keywords don't catch
  C2  regex implement -> LLM explain/review/tool_only: code in the response
      that was not applied
  C3  regex tool_only -> anything: tool turn whose outcome the rules can't read
  C4  regex explain/review -> LLM implement/tool_only: work done but not
      visible as code or tool calls
  C6  regex explain/review -> LLM explain/review/debug: prompt wording vs what
      the response does

Case IDs are `T<corpus row>` so they stay stable across rebuilds.

Usage:
    python scripts/build_disagreements.py
    python scripts/build_disagreements.py --old-ids path/to/disagreement_clusters.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "outputs/full_run_v3_analysis/specstory_prompts_cleaned.csv"
LLM_LABELS = ROOT / "outputs/llm_labels_v3/llm_labels_full.csv"
PRE_REPAIR = ROOT / "outputs/provenance/corpus_pre_repair_2026-09-17/specstory_prompts_cleaned.csv"
OUT = ROOT / "outputs/disagreements/rule_vs_llm_disagreements.csv"

KEYS = ["full_name", "source_file", "turn_index"]
LLM_COLS = ["llm_action_type", "llm_confidence", "llm_rationale", "llm_error"]
ERROR_PAGE = r"That[’']s an error"
AUTO_PROMPT = (
    r"<system_notification>|This session is being continued|^\s*\*\*Result:\*\*|<task-notification>|"
    r"Codex agent history|^Caveat:|<command-name>|tool_use_id|^\s*\[Request interrupted"
)
TOOL_MARKER = r"\[tool:|Tool use: \*\*|<summary>|Read file:|_\*\*\*\*_"

CLUSTER_NAMES = {
    "C0": "C0_empty_response",
    "C1": "C1_bugfix_missed_by_regex",
    "C2": "C2_code_shown_not_applied",
    "C3": "C3_tool_turn_outcome_unread",
    "C4": "C4_work_invisible_to_regex",
    "C5": "C5_broken_or_nonhuman_turn",
    "C6": "C6_prompt_wording_vs_response_intent",
    "C7": "C7_problem_word_not_problem_turn",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument(
        "--old-ids",
        type=Path,
        help="Earlier disagreement CSV with case_id,row_index; adds an old_case_id column for cross-reference.",
    )
    return parser.parse_args()


def load_pairs() -> pd.DataFrame:
    corpus = pd.read_csv(CORPUS)
    llm = pd.read_csv(LLM_LABELS, usecols=KEYS + LLM_COLS)
    if corpus.duplicated(KEYS).any() or llm.duplicated(KEYS).any():
        raise SystemExit("Duplicate (full_name, source_file, turn_index) keys; cannot pair rows safely.")
    corpus["row_index"] = range(len(corpus))
    df = corpus.merge(llm, on=KEYS, how="left", validate="one_to_one")
    missing = int(df["llm_action_type"].isna().sum())
    if missing:
        print(f"Turns without an LLM label (failed call or not in label file), dropped: {missing}")
    df = df[df["llm_action_type"].notna() & df["llm_error"].isna()].copy()

    if PRE_REPAIR.exists():
        before = pd.read_csv(PRE_REPAIR, usecols=KEYS + ["prompt_text"]).rename(columns={"prompt_text": "_pre_prompt"})
        df = df.merge(before, on=KEYS, how="left")
        df["was_error500_translation"] = df["_pre_prompt"].astype(str).str.contains(ERROR_PAGE, regex=True)
        df = df.drop(columns="_pre_prompt")
    else:
        df["was_error500_translation"] = df["prompt_text"].astype(str).str.contains(ERROR_PAGE, regex=True)
    return df


def assign_clusters(d: pd.DataFrame) -> pd.Series:
    prompt = d["prompt_text"].fillna("").astype(str)
    resp = d["agent_response"].fillna("").astype(str).str.strip()
    rg, llm = d["action_type"], d["llm_action_type"]

    empty = resp.eq("") | resp.str.lower().eq("nan")
    garbled = (prompt.str.count("[-�]") + prompt.str.count("�")) > 3
    stub = (resp.str.len() < 40) & ~resp.str.contains(TOOL_MARKER, regex=True)
    broken = (
        d["was_error500_translation"]
        | prompt.str.contains(AUTO_PROMPT, regex=True)
        | garbled
        | stub
        | rg.eq("other")
        | llm.eq("other")
    )

    cluster = pd.Series("", index=d.index)
    rules = [
        ("C0", empty),
        ("C5", broken),
        ("C7", rg.eq("debug")),
        ("C1", rg.eq("implement") & llm.eq("debug")),
        ("C2", rg.eq("implement")),
        ("C3", rg.eq("tool_only")),
        ("C4", rg.isin(["explain", "review"]) & llm.isin(["implement", "tool_only"])),
        ("C6", rg.isin(["explain", "review"])),
    ]
    for code, mask in rules:
        cluster[(cluster == "") & mask] = CLUSTER_NAMES[code]
    if (cluster == "").any():
        raise SystemExit(f"{int((cluster == '').sum())} disagreements matched no cluster rule.")
    return cluster


def main() -> None:
    args = parse_args()
    df = load_pairs()
    print(f"Paired turns: {len(df)}")

    dis = df[df["action_type"] != df["llm_action_type"]].copy()
    dis.insert(0, "case_id", "T" + dis["row_index"].astype(str).str.zfill(4))
    dis["cluster"] = assign_clusters(dis)
    dis = dis.rename(columns={"action_type": "regex_label", "llm_action_type": "llm_label"})

    if args.old_ids:
        old = pd.read_csv(args.old_ids, usecols=["case_id", "row_index"]).rename(columns={"case_id": "old_case_id"})
        dis = dis.merge(old, on="row_index", how="left")
    else:
        dis["old_case_id"] = ""

    cols = [
        "case_id", "old_case_id", "cluster", "regex_label", "llm_label", "llm_confidence", "llm_rationale",
        "row_index", "full_name", "source_file", "turn_index",
        "was_translated", "was_error500_translation", "has_tool_use", "code_block_count",
        "prompt_text", "prompt_text_original", "agent_response",
    ]
    dis = dis.sort_values("row_index")[cols]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dis.to_csv(args.output, index=False, encoding="utf-8-sig")

    agree = 1 - len(dis) / len(df)
    print(f"Disagreements: {len(dis)} of {len(df)} ({1 - agree:.1%}); agreement {agree:.1%}")
    print(dis["cluster"].value_counts().sort_index().to_string())
    print(pd.crosstab(dis["regex_label"], dis["llm_label"], margins=True).to_string())
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
