"""Regenerate the paper's figures directly from the result CSVs.

Every number plotted here is read from a table on disk -- nothing is typed in
by hand -- so the figures cannot drift from the tables they summarize.

Sources:
  LLM labels  : outputs/ml_action_final/tables/{cv_macro_f1,action_type_results}.csv
  Rule labels : outputs/ml_action_rule_labels_cv/tables/cv_macro_f1.csv
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
LLM_CV = ROOT / "outputs/ml_action_final/tables/cv_macro_f1.csv"
LLM_SINGLE = ROOT / "outputs/ml_action_final/tables/action_type_results.csv"
RULE_CV = ROOT / "outputs/ml_action_rule_labels_cv/tables/cv_macro_f1.csv"

MODEL = "logistic_regression"


def cv_value(path: Path, representation: str) -> tuple[float, float]:
    df = pd.read_csv(path)
    row = df[(df["representation"] == representation) & (df["model"] == MODEL)]
    if row.empty:
        raise KeyError(f"{representation} / {MODEL} not in {path}")
    return float(row["macro_f1_mean"].iloc[0]), float(row["macro_f1_std"].iloc[0])


def single_value(representation: str, model: str) -> float:
    df = pd.read_csv(LLM_SINGLE)
    row = df[(df["representation"] == representation) & (df["model"] == model)]
    if row.empty:
        raise KeyError(f"{representation} / {model} not in {LLM_SINGLE}")
    return float(row["macro_f1"].iloc[0])


def fig1() -> None:
    """Macro-F1 by representation under the final (LLM) labels.

    CV bars carry error bars; single-split-only bars are hatched, so the two
    evaluation protocols are never silently mixed as they were in the previous
    version of this figure.
    """
    cv_bars = [
        ("SBERT\nprompt-only", *cv_value(LLM_CV, "sbert_prompt_only"), "#4C72B0"),
        ("+ prev action\nonly", *cv_value(LLM_CV, "sbert+prev_action_only"), "#8172B2"),
        ("+ full\nhistory", *cv_value(LLM_CV, "sbert+prompt_metadata+history"), "#55A868"),
        ("+ repo\ncontext", *cv_value(LLM_CV, "sbert+prompt_metadata+history+repo_context"), "#55A868"),
    ]
    single_bars = [
        ("Majority\nclass", single_value("none", "majority_class"), "#9E9E9E"),
        ("Keyword rule\n(prompt only)", single_value("prompt_keyword_rule", "rule_based"), "#9E9E9E"),
        ("+ prior-turn\ntext (N=1)", single_value("context_text_n1", MODEL), "#C44E52"),
    ]

    labels = [single_bars[0][0], single_bars[1][0], cv_bars[0][0], single_bars[2][0]] + [b[0] for b in cv_bars[1:]]
    values = [single_bars[0][1], single_bars[1][1], cv_bars[0][1], single_bars[2][1]] + [b[1] for b in cv_bars[1:]]
    errors = [0, 0, cv_bars[0][2], 0] + [b[2] for b in cv_bars[1:]]
    colors = [single_bars[0][2], single_bars[1][2], cv_bars[0][3], single_bars[2][2]] + [b[3] for b in cv_bars[1:]]
    hatches = ["//", "//", "", "//", "", "", ""]

    fig, ax = plt.subplots(figsize=(7.4, 2.8))
    bars = ax.bar(labels, values, yerr=errors, color=colors, hatch=hatches,
                  edgecolor="white", capsize=4, linewidth=0.8)
    for bar, val, err in zip(bars, values, errors):
        ax.text(bar.get_x() + bar.get_width() / 2, val + err + 0.010, f"{val:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.axhline(values[0], ls=":", color="grey", lw=1)
    ax.set_ylabel("Macro-F1 (logistic reg.)", fontsize=9.5)
    ax.set_ylim(0, 0.44)
    ax.set_title("Repository-held-out macro-F1 by representation (final LLM labels)\n"
                 "Solid bars: 5-fold CV mean $\\pm$ std.  Hatched bars: single split only.",
                 fontsize=9)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_headline_macrof1.png", dpi=300)
    plt.close(fig)


def fig2() -> None:
    """The same experiment under two label sources, identical protocol.

    Both conditions are 5-fold repository-held-out CV with logistic regression
    on the same 251-repository corpus -- the previous version compared a
    single-split rule-label number against a cross-validated LLM-label number.
    """
    rule_p, rule_p_sd = cv_value(RULE_CV, "sbert_prompt_only")
    rule_h, rule_h_sd = cv_value(RULE_CV, "sbert+prompt_metadata+history")
    llm_p, llm_p_sd = cv_value(LLM_CV, "sbert_prompt_only")
    llm_h, llm_h_sd = cv_value(LLM_CV, "sbert+prompt_metadata+history")

    rule_rel = (rule_h - rule_p) / rule_p * 100
    llm_rel = (llm_h - llm_p) / llm_p * 100

    # Canvas sized to the final rendered width (0.68 x 5.5in NeurIPS text block
    # = 3.74in), so the labels are only lightly downscaled in the PDF instead of
    # shrinking to ~5.5pt as they would from a 6.2in canvas.
    fig, ax = plt.subplots(figsize=(4.4, 2.1))
    x = [0, 1]
    w = 0.36
    b1 = ax.bar([i - w / 2 for i in x], [rule_p, llm_p], w, yerr=[rule_p_sd, llm_p_sd],
                label="Prompt-only", color="#4C72B0", capsize=4)
    b2 = ax.bar([i + w / 2 for i in x], [rule_h, llm_h], w, yerr=[rule_h_sd, llm_h_sd],
                label="+ prior-turn history", color="#55A868", capsize=4)
    for bar, val, err in list(zip(b1, [rule_p, llm_p], [rule_p_sd, llm_p_sd])) + \
                         list(zip(b2, [rule_h, llm_h], [rule_h_sd, llm_h_sd])):
        ax.text(bar.get_x() + bar.get_width() / 2, val + err + 0.008, f"{val:.3f}",
                ha="center", va="bottom", fontsize=9)
    # Headroom so the "+N% rel." annotations clear the tallest value label
    # (0.364 + err + offset reaches ~0.42 in data coordinates).
    for xi, rel in zip(x, [rule_rel, llm_rel]):
        ax.text(xi, 0.468, f"+{rel:.0f}% rel.", ha="center", fontweight="bold", fontsize=10.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["Rule-based label", "LLM-assisted label\n(this work)"])
    ax.set_ylabel("Macro-F1 (logistic reg.)", fontsize=9.5)
    ax.set_ylim(0, 0.53)
    ax.set_title("The size of the “history helps” effect depends on the label\n"
                 "Identical protocol: 5-fold repository-held-out CV, same corpus",
                 fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.legend(frameon=False, loc="lower center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / "fig2_label_quality_effect.png", dpi=300)
    plt.close(fig)

    print(f"rule: {rule_p:.4f} -> {rule_h:.4f} (+{rule_rel:.1f}%)")
    print(f"llm : {llm_p:.4f} -> {llm_h:.4f} (+{llm_rel:.1f}%)")


if __name__ == "__main__":
    fig1()
    fig2()
    print(f"figures written to {OUT}")
