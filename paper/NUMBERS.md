# Provenance of every number in `main.tex`

Every claim maps to an artifact in this repo. Recency was checked by file
mtime, not by directory name — see "Recency audit" at the bottom.

## Corpus (Section 2)

| Claim | Source |
|---|---|
| 8,404 turns / 251 repos / 3,191 session logs | `outputs/full_run_v3_analysis/specstory_prompts_cleaned.csv` (recomputed) |
| 68,943 raw turn pairs before cleaning | `outputs/full_run_v3_analysis/analysis/RESEARCH_REPORT.md` §2 |
| Six discovery queries | `src/github_repo_miner/queries.py` |
| Cleaning filters | `src/github_repo_miner/data_cleaning.py` `_NOISE_PATTERNS` |
| Per-repo cap of 50, stratified | `data_cleaning.py` `_cap_repo_group`; max per repo in corpus = 50 |
| Gini 0.299, top-10 share 5.9% | `outputs/full_run_v3_analysis/analysis/diversity_metrics.csv` |
| Median 15 words; 56.7% code; 23.1% tool use; 31% translated | recomputed from `llm_labels_full.csv` |
| Median 11 turns, max 207 | recomputed (`session_turn_count`, recorded pre-cap) |
| Median 0 stars, median age 14.5 mo; TS 82 / Py 50 / JS 28 | `outputs/full_run_v3_analysis/repo_context.csv` |
| One repo contributed 30,000+ raw turns | `RESEARCH_PROPOSAL.md` §4.1 — **project record, not recomputed here** |

## Labels (Section 3)

| Claim | Source |
|---|---|
| Rule definition | `src/github_repo_miner/action_labeling.py` |
| LLM judge = `gemini-3.1-flash-lite`, 4K/20K truncation, few-shot | `ml_research/llm_labeling.py` |
| Sharpened `debug` / `review` definitions | `llm_labeling.py` `ACTION_DEFINITIONS` (+ `PROMPT_VERSION = "v2-taxonomy-fix"`) |
| 8,398 of 8,404 labeled; mean confidence 0.95 | recomputed (`llm_error`, `llm_confidence`) |
| Class distributions (Table 1) | recomputed from `llm_labels_full.csv` |
| Raw agreement 42.5%, κ = 0.223 | recomputed, n = 8,398 |
| 1,448 rule-`implement` → LLM-`debug`; 646 rule-`tool_only` → LLM-`implement` | recomputed crosstab |
| **37.5% of labels flipped after the taxonomy refinement; κ(pre,post) = 0.540** | `outputs/provenance/llm_labels_pre_refinement/llm_labels_sample.csv` (300-row pre-fix sample, 2026-08-12) vs `llm_labels_full.csv`; 299 rows labeled in both, 187 unchanged |
| `debug` 13.0% → 29.8%, `tool_only` 15.7% → 6.0% on that sample | same comparison (sample is class-stratified — flagged in a footnote) |
| 212-turn sheet, 98 adjudicated | `outputs/llm_labels_v3/human_review_sheet_SAFE.csv` |
| **Final LLM κ = 0.512 / 61.2%; final rule κ = 0.163 / 31.6%** | recomputed: sheet joined to `llm_labels_full.csv` on `full_name` + normalized prompt prefix (212/212 match) |
| Pre-refinement LLM scored κ = 0.482 on the same turns | recomputed from the sheet's own `llm_action_type` column |
| **Imputed variant: LLM κ = 0.727, rule κ = 0.532, n = 174** | 98 adjudicated + 76 unadjudicated turns where final rule == final LLM |
| 27 of the 98 now agree under the final labels | recomputed |

## Modeling (Section 4)

All from `outputs/ml_action_final/` (run of 2026-09-04, log `ml_action_final_log2.txt`).

| Claim | Source |
|---|---|
| 8,398 rows, 200/51 repos, 6,665/1,733 turns, 0 overlap | `tables/train_test_split.md` |
| Baselines 0.086 / 0.320 macro-F1 | `tables/baselines.md` (single split) |
| All CV rows in Table 2 | `tables/cv_macro_f1.md` |
| Fold deltas, t = 10.4, p = 0.0005, Wilcoxon p = 0.0625 | `tables/cv_paired_significance.md` |
| prev-action-only vs. full history: p = 0.10 | same file, row 2 |
| repo context changes macro-F1 by −0.002 | `cv_macro_f1.md`: 0.3616 vs 0.3640 (LR) |
| Prior-turn text N=1/2/3 = 0.314 / 0.301 / 0.309 vs 0.357 | `tables/action_type_results.md` (MLP, single split) |
| Per-class F1 (debug .72 … tool_only .04) | recomputed with `classification_report` on the same split/features; macro-F1 reproduces the published 0.3796 exactly |
| No response-side feature used as input | `ml_research/action_prediction.py` `PROMPT_METADATA_COLS` / `HISTORY_NUMERIC_COLS` |

## Corrections (Section 5)

| Stage | Source |
|---|---|
| Before parser fix: 0.237 → 0.412 | `outputs/provenance/ml_action_v1_before_parser_fix/tables/action_type_results.md` (best per representation: MLP 0.2367 / MLP 0.4117) |
| After parser fix: 0.223 → 0.348 | `outputs/provenance/ml_action_v3_after_parser_fix/tables/action_type_results.md` (LR 0.2226 / LR 0.3480) |
| After relabeling: 0.341 → 0.364 (CV) | `outputs/ml_action_final/tables/cv_macro_f1.md` |
| Parser bug description | `RESEARCH_PROPOSAL.md` §4.1 note; `src/github_repo_miner/specstory_parser.py` |

## Recency audit (2026-09-04)

> **2026-09-17:** `archive/` was moved out of the repository to
> `../prompt_research_archive/archive/`. The archive files this paper cites were
> copied first into `outputs/provenance/` and `outputs/ml_action_rule_labels_cv/`
> (paths in the tables above are already updated). References to `archive/` below
> describe the audit as it was run.

Newest artifact of each kind, confirmed by mtime across the whole repo
including `archive/`:

- **Modeling:** `outputs/ml_action_final/` — regenerated 2026-09-04 01:37–02:21.
  All tables come from that single run; `ACTION_PREDICTION_REPORT.md` matches.
  Nothing newer exists anywhere.
- **Labels:** `outputs/llm_labels_v3/llm_labels_full.csv` — 2026-09-01.
  `archive/.../llm_labels_v3_history/llm_labels_checkpoint.csv` is the checkpoint
  of that same run, not a newer one. `llm_labels_v1/` is the superseded
  300-row era.
- **Corpus:** `outputs/full_run_v3_analysis/specstory_prompts_cleaned.csv` —
  2026-08-11, the post-parser-fix v3 corpus, and the file
  `llm_labels_full.csv` is built from (both 8,404 rows).
- **Repo metadata:** `repo_context.csv` — 2026-09-01, 251 rows.
- Everything under `archive/` is older than its `outputs/` counterpart.

## Known inconsistencies in the repo (worth fixing)

1. **`outputs/README.md` quotes κ = 0.760 / 0.614 as the headline validation
   number.** Those figures are computed against the *superseded* LLM labeling
   that `human_review_sheet_SAFE.csv` was adjudicated on (2026-08-26), and use
   all 212 rows as the imputation base. The equivalent figures for the label set
   the models actually use are **κ = 0.727 / 0.532 (n = 174)**, and the clean
   adjudicated-only figures are **0.512 / 0.163**. The paper uses the corrected
   numbers.
2. **`RESEARCH_PROPOSAL.md` §5 and §9 still say the human adjudication pass
   "has not yet been performed."** It has — 98 rows in the SAFE sheet.
3. `RESEARCH_PROPOSAL.md` §7's results (κ ≈ 0.32 label source, 0.223 → 0.348)
   are the rule-label era and are superseded by `outputs/ml_action_final/`.

## Not verified — check before submitting

- **All three bibliography entries.** Carried over from `RESEARCH_PROPOSAL.md`
  §8's related-work addendum; not checked against arXiv. Verify or drop.
- Author name/affiliation on the title page is a placeholder.
