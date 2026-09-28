"""Uncertainty for the label-validity numbers in the MSR draft.

Every headline figure gets an interval: agreement and kappa between the two
constructions on the corpus, accuracy and kappa of each construction against
the human reference set, per-category precision and recall against that set,
and the share of labels that a different definition text changes. Resampling is
clustered by repository, since turns within a repository are not independent.

Writes paper/msr2027/INTERVALS.md.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, cohen_kappa_score
from statsmodels.stats.contingency_tables import mcnemar

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "outputs"
A = ["implement", "debug", "explain", "review", "tool_only", "other"]
KEYS = ["full_name", "source_file", "turn_index"]
N_BOOT = 2000
rng = np.random.default_rng(42)
lines: list[str] = []


def cluster_boot(fn, groups: np.ndarray) -> tuple[float, float]:
    uniq = np.unique(groups)
    idx_by_group = {g: np.where(groups == g)[0] for g in uniq}
    vals = []
    for _ in range(N_BOOT):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in drawn])
        try:
            vals.append(fn(idx))
        except Exception:
            continue
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ---------- corpus: the two constructions against each other
c = pd.read_csv(OUT / "full_run_v3_analysis/specstory_prompts_cleaned.csv")
llm = pd.read_csv(OUT / "llm_labels_v4/llm_labels_full.csv", usecols=KEYS + ["llm_action_type", "llm_error"])
c = c.merge(llm, on=KEYS, how="left", validate="one_to_one")
c = c[c["llm_action_type"].notna() & c["llm_error"].isna()].reset_index(drop=True)
rule, judge, repos = c["action_type"].to_numpy(), c["llm_action_type"].to_numpy(), c["full_name"].to_numpy()

lines.append("## Rule labeler vs LLM judge, whole corpus\n")
agree = float((rule == judge).mean())
lo, hi = cluster_boot(lambda i: float((rule[i] == judge[i]).mean()), repos)
lines.append(f"- agreement: {agree:.3f} [{lo:.3f}, {hi:.3f}] (n={len(c)}, {len(np.unique(repos))} repositories)")
k = cohen_kappa_score(rule, judge, labels=A)
lo, hi = cluster_boot(lambda i: cohen_kappa_score(rule[i], judge[i], labels=A), repos)
lines.append(f"- Cohen's kappa: {k:.3f} [{lo:.3f}, {hi:.3f}]")
for lab in A:  # per-category overlap with interval
    r, j = rule == lab, judge == lab
    val = float((r & j).sum() / max((r | j).sum(), 1))
    lo, hi = cluster_boot(lambda i, lab=lab: float(((rule[i] == lab) & (judge[i] == lab)).sum()
                                                   / max(((rule[i] == lab) | (judge[i] == lab)).sum(), 1)), repos)
    lines.append(f"- {lab}: both/either {val:.3f} [{lo:.3f}, {hi:.3f}]")

# ---------- human reference set
g = pd.read_csv(OUT / "gold_v2/scored_rows.csv")
idx = pd.read_csv(OUT / "gold_v2/sample_index.csv")
g = g.merge(idx[["grade_id", "full_name"]], on="grade_id")
human = g["your_label"].to_numpy()
hrepos = g["full_name"].to_numpy()
lines.append("\n## Against the human reference set (n=%d)\n" % len(g))
cols = [("PROMPT_STRICT_FIXED", "LLM judge (production definitions)"),
        ("PROMPT_BOOK", "LLM judge (prose definitions)"),
        ("llm_v2_legacy", "LLM judge (earlier definitions)"),
        ("regex_current", "rule labeler")]
for col, name in cols:
    pred = g[col].to_numpy()
    acc = float((pred == human).mean())
    alo, ahi = cluster_boot(lambda i, p=pred: float((p[i] == human[i]).mean()), hrepos)
    kk = cohen_kappa_score(human, pred, labels=A)
    klo, khi = cluster_boot(lambda i, p=pred: cohen_kappa_score(human[i], p[i], labels=A), hrepos)
    lines.append(f"- {name}: accuracy {acc:.3f} [{alo:.3f}, {ahi:.3f}]; kappa {kk:.3f} [{klo:.3f}, {khi:.3f}]")

judge_ok = g["PROMPT_STRICT_FIXED"].to_numpy() == human
rule_ok = g["regex_current"].to_numpy() == human
n01, n10 = int((~judge_ok & rule_ok).sum()), int((judge_ok & ~rule_ok).sum())
p = mcnemar([[0, n01], [n10, 0]], exact=True).pvalue
lines.append(f"- judge vs rule labeler, exact McNemar: judge only correct {n10}, rule only correct {n01}, p={p:.2e}")

lines.append("\nPer-category against the human reference set (precision / recall / n):\n")
for col, name in [("PROMPT_STRICT_FIXED", "judge"), ("regex_current", "rule")]:
    rep = classification_report(human, g[col], labels=A, output_dict=True, zero_division=0)
    lines.append(f"- {name}: " + ", ".join(
        f"{lab} {rep[lab]['precision']:.2f}/{rep[lab]['recall']:.2f} (n={int(rep[lab]['support'])})" for lab in A))

# ---------- definition sensitivity across the corpus
prev = pd.read_csv(OUT / "llm_labels_v3/llm_labels_full.csv", usecols=KEYS + ["llm_action_type", "llm_error"])
prev = prev.rename(columns={"llm_action_type": "prev_label", "llm_error": "prev_err"})
m = c.merge(prev, on=KEYS, how="left")
m = m[m["prev_label"].notna() & m["prev_err"].isna()]
a, b, mrepos = m["prev_label"].to_numpy(), m["llm_action_type"].to_numpy(), m["full_name"].to_numpy()
changed = float((a != b).mean())
lo, hi = cluster_boot(lambda i: float((a[i] != b[i]).mean()), mrepos)
kk = cohen_kappa_score(a, b, labels=A)
klo, khi = cluster_boot(lambda i: cohen_kappa_score(a[i], b[i], labels=A), mrepos)
lines.append("\n## One definition text vs another, same model and inputs\n")
lines.append(f"- labels changed: {changed:.3f} [{lo:.3f}, {hi:.3f}] of {len(m)} turns")
lines.append(f"- kappa between the two label sets: {kk:.3f} [{klo:.3f}, {khi:.3f}]")

(HERE / "INTERVALS.md").write_text("# Intervals for the label-validity numbers\n\n"
                                   "Cluster bootstrap over repositories, 2,000 resamples.\n\n"
                                   + "\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
