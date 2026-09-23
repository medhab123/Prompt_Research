"""Recompute every number used in the MSR draft from the current data.

Writes paper/msr2027/NUMBERS_draft.md. Run from anywhere.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, cohen_kappa_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = REPO / "outputs"
A = ["implement", "debug", "explain", "review", "tool_only", "other"]
KEYS = ["full_name", "source_file", "turn_index"]
lines: list[str] = []


def add(k: str, v) -> None:
    lines.append(f"- {k}: {v}")


def kappa(a, b) -> float:
    return cohen_kappa_score(a, b, labels=A)


# ---------------- corpus
c = pd.read_csv(OUT / "full_run_v3_analysis/specstory_prompts_cleaned.csv")
llm = pd.read_csv(OUT / "llm_labels_v4/llm_labels_full.csv", usecols=KEYS + ["llm_action_type", "llm_error"])
c = c.merge(llm, on=KEYS, how="left", validate="one_to_one")
lines.append("## Corpus")
add("turns", len(c))
add("repositories", c["full_name"].nunique())
add("sessions (history files)", c["session_id"].nunique())
per_sess = c.groupby("session_id").size()
add("turns per session: median / mean / max", f"{per_sess.median():.0f} / {per_sess.mean():.2f} / {per_sess.max()}")
per_repo = c.groupby("full_name").size().sort_values()
add("turns per repo: median / max", f"{per_repo.median():.0f} / {per_repo.max()}")
x = np.sort(per_repo.values.astype(float)); n = len(x)
gini = (2 * np.sum(np.arange(1, n + 1) * x) / (n * x.sum())) - (n + 1) / n
add("repo Gini", f"{gini:.3f}")
add("top-10 repos share of turns", f"{per_repo.sort_values(ascending=False).head(10).sum() / len(c):.1%}")
dates = c["source_file"].astype(str).str.extract(r"(\d{4}-\d{2}-\d{2})")[0].dropna()
add("session dates found in file names", f"{len(dates)} turns; range {dates.min()} to {dates.max()}")
add("translated prompts", f"{int(c['was_translated'].sum())} ({c['was_translated'].mean():.1%})")
add("has_tool_use", f"{int(c['has_tool_use'].sum())} ({c['has_tool_use'].mean():.1%})")
add("has_generated_code", f"{int(c['has_generated_code'].sum())} ({c['has_generated_code'].mean():.1%})")
empty = c["agent_response"].fillna("").astype(str).str.strip().eq("")
add("empty agent response", f"{int(empty.sum())} ({empty.mean():.1%})")
add("prompt words: median", f"{c['prompt_word_count'].median():.0f}")
add("response chars: median / 90th pct", f"{c['agent_response_length'].median():.0f} / {c['agent_response_length'].quantile(.9):.0f}")

# ---------------- labels
ok = c["llm_action_type"].notna() & c["llm_error"].isna()
d = c[ok]
lines.append("\n## Labels (rule = current rule labeler; LLM = corpus LLM labels)")
add("LLM-labeled turns", len(d))
add("LLM calls failed", int((~ok).sum()))
add("agreement", f"{(d['action_type'] == d['llm_action_type']).mean():.1%}")
add("kappa", f"{kappa(d['action_type'], d['llm_action_type']):.3f}")
for lab in A:
    r, l = d["action_type"] == lab, d["llm_action_type"] == lab
    add(f"{lab}: rule / LLM prevalence; both/either",
        f"{r.mean():.1%} / {l.mean():.1%}; {int((r & l).sum())}/{int((r | l).sum())} = {(r & l).sum() / (r | l).sum():.1%}")
ct = pd.crosstab(d["action_type"], d["llm_action_type"]).reindex(index=A, columns=A, fill_value=0)
lines.append("\nConfusion (rows = rule, cols = LLM):\n")
lines.append(ct.to_markdown())

# ---------------- blind human reference set
lines.append("\n## Blind random reference set (100 sampled, 2 excluded as used in drafting)")
sc = pd.read_csv(OUT / "gold_v2/scored_rows.csv")
add("graded turns scored", len(sc))
add("grader label counts", sc["your_label"].value_counts().to_dict())
add("confidence", sc["confidence"].astype(str).str.lower().value_counts().to_dict())
for col, name in [("regex_current", "rule labeler"), ("llm_v2_legacy", "earlier definition text"),
                  ("PROMPT_BOOK", "detailed codebook"), ("PROMPT_STRICT", "operational codebook, before fix"),
                  ("PROMPT_STRICT_FIXED", "operational codebook (production)")]:
    m = sc[sc[col].notna()]
    hi = m[m["confidence"].astype(str).str.lower() == "high"]
    add(name, f"{(m[col] == m['your_label']).mean():.1%}, kappa {kappa(m['your_label'], m[col]):.3f}; "
              f"high-confidence rows {(hi[col] == hi['your_label']).mean():.1%} (n={len(hi)})")
import itertools
from statsmodels.stats.contingency_tables import mcnemar
cands = ["llm_v2_legacy", "PROMPT_BOOK", "PROMPT_STRICT", "PROMPT_STRICT_FIXED"]
okc = {c: (sc[c] == sc["your_label"]) for c in cands}
for a, b in itertools.combinations(cands, 2):
    n01, n10 = int((~okc[a] & okc[b]).sum()), int((okc[a] & ~okc[b]).sum())
    add(f"McNemar {a} vs {b}", f"{n10} / {n01}, p={mcnemar([[0, n01], [n10, 0]], exact=True).pvalue:.3f}")
rep = classification_report(sc["your_label"], sc["regex_current"], labels=A, output_dict=True, zero_division=0)
add("rule labeler per-category recall", {k: round(rep[k]["recall"], 2) for k in A})

# ---------------- definition sensitivity across the whole corpus
prev = pd.read_csv(OUT / "llm_labels_v3/llm_labels_full.csv", usecols=KEYS + ["llm_action_type", "llm_error"])
prev = prev.rename(columns={"llm_action_type": "prev_label"})
cmp = c.merge(prev[KEYS + ["prev_label", "llm_error"]].rename(columns={"llm_error": "prev_err"}), on=KEYS, how="left")
both = cmp["llm_action_type"].notna() & cmp["prev_label"].notna() & cmp["prev_err"].isna() & cmp["llm_error"].isna()
b2 = cmp[both]
lines.append("\n## Definition sensitivity across the whole corpus (same model and inputs, two definition texts)")
add("turns labeled by both texts", len(b2))
add("same label", f"{(b2['prev_label'] == b2['llm_action_type']).mean():.1%}, kappa {kappa(b2['prev_label'], b2['llm_action_type']):.3f}")
add("labels changed", int((b2["prev_label"] != b2["llm_action_type"]).sum()))
flows = (b2[b2["prev_label"] != b2["llm_action_type"]].groupby(["prev_label", "llm_action_type"]).size()
         .sort_values(ascending=False).head(6))
add("largest changes", {f"{a} -> {b}": int(n) for (a, b), n in flows.items()})
add("prevalence under the earlier text", (b2["prev_label"].value_counts(normalize=True) * 100).round(1).to_dict())

# ---------------- disagreement clusters
dis = pd.read_csv(OUT / "disagreements/rule_vs_llm_disagreements.csv")
lines.append("\n## Disagreement clusters")
add("disagreements", len(dis))
for k, v in dis["cluster"].value_counts().sort_index().items():
    add(k, f"{v} ({v / len(dis):.1%})")

# ---------------- adjudicated set
g = pd.read_csv(OUT / "rubric_tests/2026-09-22_gold_check/gold_labels_scored.csv")
lines.append("\n## Adjudicated contested turns")
add("rows matched", len(g))
add("human label counts", g["human_label"].value_counts().to_dict())
for col, name in [("llm_corpus", "LLM (corpus)"), ("regex_v3", "rule (current)"), ("G_v3", "LLM codebook A"), ("G_v41", "LLM codebook B")]:
    add(name, f"{(g[col] == g['human_label']).mean():.1%}, kappa {kappa(g['human_label'], g[col]):.3f}")

# ---------------- definition sensitivity (200 disagreement-stratified turns)
R = OUT / "rubric_tests/2026-09-21_v41/runs"
lab = lambda f: pd.read_csv(R / f"{f}.csv").set_index("case_id")["label"].where(lambda s: s.fillna("") != "")
s200 = pd.read_csv(R / "sample.csv").set_index("case_id")
S = pd.DataFrame({k: lab(k) for k in ["G_v3_t0", "G_v3_full", "G_v4_t0", "G_v4_para", "G_v41_t0", "G_v41_para", "G_v41_full"]})
S = S.join(s200[["llm_label"]])
lines.append("\n## Definition sensitivity (200 turns, 40 from each of C1, C2, C4, C6, C7)")


def pair(a, b, label):
    m = S[[a, b]].dropna()
    add(label, f"n={len(m)}, same label {(m[a] == m[b]).mean():.1%}, kappa {kappa(m[a], m[b]):.3f}, changed {int((m[a] != m[b]).sum())}")


pair("G_v3_full", "G_v41_full", "codebook A vs codebook B (full length)")
pair("G_v4_t0", "G_v41_t0", "B0 (applied-change test not tightened) vs B (3k chars)")
pair("G_v4_t0", "G_v4_para", "B0 original vs paraphrase")
pair("G_v41_t0", "G_v41_para", "B original vs paraphrase")
pair("G_v3_t0", "G_v3_full", "A: 3,000 vs 20,000 chars")
pair("G_v41_t0", "G_v41_full", "B: 3,000 vs 20,000 chars")
pair("llm_label", "G_v3_full", "corpus LLM label vs codebook A (full length)")
for col in ["G_v3_full", "G_v41_full"]:
    add(f"{col} label counts", S[col].value_counts().to_dict())
T = OUT / "rubric_tests/2026-09-21_v4_draft/runs"
t = pd.DataFrame({k: pd.read_csv(T / f"{k}.csv").set_index("case_id")["label"] for k in ["G_old_t0", "G_old_t07", "G_new_t0", "G_new_t07"]})
for a, b, nm in [("G_old_t0", "G_old_t07", "A: temp 0 vs 0.7 (50 turns)"), ("G_new_t0", "G_new_t07", "B0: temp 0 vs 0.7 (50 turns)")]:
    add(nm, f"same {(t[a] == t[b]).mean():.1%}, kappa {kappa(t[a], t[b]):.3f}")

# ---------------- review sub-clusters
lines.append("\n## Review sub-clusters (silhouette, best over k=2..8)")
for m in ["all-MiniLM-L6-v2", "nomic-ai_CodeRankEmbed"]:
    sil = pd.read_csv(OUT / f"review_subclusters/{m}/silhouette.csv")
    ks = [c for c in sil.columns if c.startswith("k=")]
    for _, r in sil.iterrows():
        best = max(ks, key=lambda k: r[k])
        add(f"{m} | {r['view']} | {r['set']} (n={r['n']})", f"{r[best]:.3f} at {best}")
rv = pd.read_csv(OUT / "review_subclusters/all-MiniLM-L6-v2/review_turns_clustered.csv")
add("review turns (non-empty) by source", rv["review_by"].value_counts().to_dict())

# ---------------- repairs
lines.append("\n## Extraction defects")
gem = pd.read_csv(OUT / "provenance/corpus_pre_repair_2026-09-17/retranslated_with_gemini.csv")
pre = pd.read_csv(OUT / "provenance/corpus_pre_repair_2026-09-17/specstory_prompts_cleaned.csv", usecols=["has_tool_use"])
add("prompts whose stored translation was an error page", len(gem))
add("has_tool_use before / after legacy-format detection", f"{int(pre['has_tool_use'].sum())} / {int(c['has_tool_use'].sum())}")

(HERE / "NUMBERS_draft.md").write_text("# Numbers used in the MSR draft\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
