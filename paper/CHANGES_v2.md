# `agent_research_v2.tex` — what changed from AGENT_RESEARCH.pdf, and why

Every change below is a correction to a number or a claim, traced to an
artifact. Nothing was added that isn't in the data. Items marked **ASK** need
your decision.

## A. Numbers that were wrong

| # | Was | Now | Evidence |
|---|---|---|---|
| A1 | κ = 0.760 vs 0.614 | **κ = 0.512 vs 0.163** (98 adjudicated turns, no imputation) | The 0.760 figure scores the *pre-refinement* LLM labels, which were replaced before any modeling. Recomputed by joining `human_review_sheet_SAFE.csv` to `llm_labels_full.csv` (212/212 match after normalizing the spreadsheet-safe `prompt_text`). |
| A2 | "56% → 7%" | **"+45.6% → +6.7%"**, identical protocol | The old pair compared single-split (rule) against 5-fold CV (LLM). Both numbers now come from 5-fold repo-held-out CV, logistic regression, same corpus: rule `0.2186 → 0.3183`, LLM `0.3411 → 0.3640`. |
| A3 | "∼19% of responses blank" | **48.4% → 29.8%** blank after the fix | Measured directly over both raw extractions. 19 was the *percentage-point reduction*, not a blank rate. ~30% remain legitimately blank. |
| A4 | "∼35% non-English" | **31%** | 2,604 / 8,404 rows with `was_translated`. |
| A5 | "top-5 repo share 34% → 3%" | Replaced with **Gini 0.299, top-10 = 5.9%** | The 34%→3% figure is a project-record claim I could not recompute; the Gini/top-10 figures come from `analysis/diversity_metrics.csv`. |
| A6 | prev-action "0.355", history "0.364" | **0.361 vs 0.364**, both 5-fold CV | 0.355 was single-split, 0.364 was CV — another protocol mix. |

## B. Claims that outran the evidence

| # | Was | Now | Why |
|---|---|---|---|
| B1 | "substantially more accurate than the rule" | "agrees with human judgment roughly twice as often" | Accuracy presumes the human is ground truth. Agreement is what was measured. |
| B2 | "as ground truth for the rest of this paper" | "as our **working** label" | With κ = 0.512 the label disagrees with a careful reader on ~4 turns in 10. |
| B3 | "statistically indistinguishable from 0.364" | "no detectable difference (paired *t* = −2.12, *p* = 0.102; Wilcoxon *p* = 0.1875) — an absence of evidence at n=5, not evidence of equivalence" | The test now exists (`cv_paired_significance.csv`) and it does not support "indistinguishable". |
| B4 | "a real, cross-validated effect" (no test shown) | Full test reported: positive on all 5 folds, paired *t* = 10.40, *p* = 0.00048; Wilcoxon *p* = 0.0625 with a note that this is the floor at n=5 | The test exists — it just wasn't in the draft. This is now a *stronger* claim, properly backed. |
| B5 | "The signal ... is categorical (which action), not lexical (which words)" | "a result about representation, not about information ... does not establish that prior-turn text carries no lexical signal" | The experiment only shows *your* encoder didn't extract it. |
| B6 | Label quality "determines" the magnitude | "the measured effect size differs sharply by label, not that label noise *causes* a specific inflation" + the mechanism offered as "the plausible account" | Observational comparison of two pipelines, not a randomized manipulation. |
| B7 | Repo-held-out framed as ruling out memorization | Explicitly labelled an evaluation-design choice, "not itself a measurement of cross-repository generalization, which we do not attempt" | This is the RQ3 removal you asked for. |

## C. Table 1, rebuilt

The old table's per-class numbers (implement 61→87%, explain 60→90%, tool_only
79→88%, review precision 39→100%) are recall/precision for the **pre-refinement**
LLM labels computed **with imputation** — and the imputation is circular for that
column, because the imputed "truth" on the 114 unadjudicated rows *is* the
pre-refinement LLM label.

The new table scores all three labelings against **real human labels only**
(n = 98). It also surfaces something the old table hid:

> The refinement raised `debug` from 0.33 to 0.83 but cost `explain`
> (0.80 → 0.53) and `tool_only` (0.58 → 0.08), netting +4 points overall.

## D. Things added (all from existing artifacts)

- **Per-class prediction F1** (`debug` 0.72 … `tool_only` 0.04) tied to the
  agreement table — the classes the model can predict are the ones the labelers
  agree on, and `tool_only` is worst on both. Recomputed with
  `classification_report` on the same split; macro-F1 reproduces the published
  0.3796 exactly.
- **Model-family consistency under CV** (LR 0.341→0.364, MLP 0.338→0.364,
  XGBoost 0.288→0.312, RF flat at 0.215→0.216). Answers "why show only logistic
  regression?" Note: RF is reported as the exception, not folded into "all four
  agree" — its matched-representation delta is +0.001.
- **Both limits on the κ estimate**: the 98 turns were selected as
  disagreements (harder than average), and the adjudicator was not blind.

## E. Provenance note on the rule-label CV

The like-for-like comparison in A2 uses
`archive/superseded_pipeline_runs/ml_action_v4_dryrun/`. Despite the name, its
report header confirms a full run: `action_type` labels, 8,404 rows, 251 repos,
200/51 repo split, zero overlap, same embedding model, 5-fold CV. **Recommend
moving it out of `archive/` and renaming** — the paper now depends on it, and
"dryrun" in an archive folder reads as throwaway.
**Done 2026-09-17:** now at `outputs/ml_action_rule_labels_cv/`.

Its CV table covers logistic regression only, so there is no paired test on the
rule side. The paper says so.

## F. Open items — need you

1. **ASK: the three references are still unverified.** Carried over from your
   draft. Check the arXiv IDs, author lists and titles, or drop them.
2. **Page count — superseded by §G.** The earlier fix (shrinking the figure
   canvases in `make_figures.py`) is retained, but the length problem is now
   solved structurally: the CFP allows an **unlimited appendix**, so overflow
   content was moved there rather than compressed away. See §G.3.
3. **ASK: citation style.** I kept your `[Ong et al., 2025]` inline style with a
   plain reference list. If you switch to `neurips_2026.sty` with natbib, these
   should become `\citep{}`.
4. **Not done: `RESEARCH_PROPOSAL.md` and `outputs/README.md` still carry the
   old κ = 0.760/0.614 and say the human pass hasn't happened.** Worth fixing so
   the repo doesn't contradict the paper.

---

## G. Venue compliance (IAB @ NeurIPS 2026)

Requirements read off the CFP at `https://iab-agents.github.io/`
(Call for Papers → Submission Format / Review Process / Ethics).

| Requirement | Status |
|---|---|
| Short paper: **up to 4 pages + references + unlimited appendix** | Body restructured to fit 4; overflow moved to appendix (§G.3) |
| **NeurIPS-style formatting** | Preamble now loads `neurips_2026.sty` (§G.1) |
| **Double-blind review** | Fixed — the draft was de-anonymizing itself (§G.2) |
| Submission site: OpenReview | — |
| Non-archival | No action needed |
| NeurIPS policies on agents/LLMs + research ethics, IRB where applicable | Appendix C added (§G.4) |

### G.1 Formatting: real NeurIPS style, and no margin tricks

The v2 preamble was an `article`-class *approximation* of NeurIPS. It now loads
the actual style file:

```latex
\documentclass{article}
\usepackage{neurips_2026}   % NO [final] option — that is camera-ready mode
```

`[final]` prints author names. Adding it would break double-blind.

**Removed as format manipulation** (NeurIPS prohibits altering the template to
gain space, and all of these did):

- `\vspace{-2em}` after `\title` and after `\maketitle`
- `\usepackage[letterpaper,margin=1in]{geometry}` — overrode the NeurIPS text block
- `\titlespacing*{\section}{...}` — compressed section spacing
- `\setlength{\parskip}{0pt}`, `\setlength{\textfloatsep}{8pt}`
- `\captionsetup{skip=4pt}`

Removing these makes the paper *longer*, which is why the appendix restructure
below matters.

### G.2 Double-blind: what was leaking

The draft carried `\url{https://github.com/medhab123/Prompt_Research}` in the
Artifacts paragraph. That names the author. It is **removed** and replaced with a
red bracketed placeholder for an anonymized mirror (e.g.
`anonymous.4open.science`). Verified by grep: no author name, handle, email, or
public repo URL remains anywhere in the `.tex`.

Note the two contemporaneous works are cited in third person already, so there
is no self-citation leak.

### G.3 Fitting 4 pages honestly

Moved to the appendix (allowed to be unlimited), not deleted:

| Moved | To | Recovered |
|---|---|---|
| Figure 1 (representation comparison) | Appendix A | ~0.30 page |
| Parser-failure detail (~150 words) | Appendix B, condensed to 5 lines in body | ~0.15 page |
| Full CV table, all reps × 4 model families | Appendix A, Table 2 | new content |
| Per-fold deltas + paired tests | Appendix A, Table 3 | new content |

Body now carries **Table 1 (label agreement) and Figure 2 (the headline)** only.
Every number in the new appendix tables was cross-checked cell-by-cell against
`outputs/ml_action_final/tables/cv_macro_f1.csv` — all 20 CV cells match.

### G.4 Ethics and LLM usage (Appendix C, new)

The CFP requires adherence to NeurIPS policies on agents/LLMs and research
ethics, "including obtaining appropriate IRB/institutional approval for studies
involving human participants." Appendix C states, factually:

- all transcripts were already public, retrieved via GitHub's APIs;
- no participants were recruited and no demographics collected, so this is not
  human-subjects research — the only human annotation is by an author on the
  authors' own data;
- a residual privacy caveat: developer prompts are free text and can incidentally
  contain identifying detail, so the corpus is PII-adjacent, not anonymized;
- the LLM's role as a *labeling instrument* is described.

**ASK:** Appendix C ends with a red bracket for you to state any use of LLMs in
**writing** the manuscript. I have not filled this in — only you know what to
declare, and it must be accurate.

### G.5 Deadline

Per the CFP: the no-reviews deadline is **September 5, 2026 (AoE)** — the site's
countdown showed under two days when checked. A second route,
**"Submission with NeurIPS reviews," closes September 25, 2026**, if this paper
was reviewed at NeurIPS 2026 and you would rather submit it with the reviews.

### G.6 Cosmetic note

`neurips_2026.sty` in submission mode stamps the footer *"Submitted to 40th
Conference on Neural Information Processing Systems (NeurIPS 2026). Do not
distribute."* That is the main-track boilerplate, not workshop-specific. It is a
normal artifact of the required style and needs no action.

---

## H. Compression pass to a hard 4 pages

Body was 4.75 pages. **435 words removed (~0.7–0.8 page).** No spacing hacks —
NeurIPS prohibits them — and no caveat weakened.

| Change | Words | Content lost? |
|---|---|---|
| Abstract 285 → 205 words | −80 | No — cut the six class names (defined in §1) and redundant phrasing |
| §1: related-work para folded into para 1 | −45 | No |
| §4 Threats → 4-sentence summary in body, **full text moved to Appendix D** | −180 | No — appendix is unlimited |
| §3 setup and per-class prose tightened | −55 | No |
| §3 model-family numbers → pointer to Appendix Table 2 | −30 | No — the numbers are in the appendix table |
| Repo-metadata negative result compressed to one clause | −25 | No |
| §2 corpus + disagreement prose tightened | −45 | No |
| Table 1 caption: dropped the sentence the body already states | −25 | No |

All 7 `\ref` targets verified against their `\label`s; environment nesting
balanced; anonymity and no-format-manipulation checks re-run clean.

**Estimated landing: 3.96–4.05 pages**, so it may still sit a line or two over.
If so, use these in order — each is safe:

1. Figure 2 `width=0.78` → `0.68` (≈3 lines)
2. Move the parser paragraph in §2 wholly to Appendix B, leaving its first
   sentence and the 48.4%→29.8% figure (≈4 lines)
3. Move the "Negative result: raw-text" paragraph's final clause on repository
   metadata to Appendix A (≈2 lines)

Do **not** reintroduce `\vspace`, `geometry`, or `\titlespacing` to fit.
