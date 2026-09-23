# Numbers used in the MSR draft

## Corpus
- turns: 8404
- repositories: 251
- sessions (history files): 3197
- turns per session: median / mean / max: 1 / 2.63 / 50
- turns per repo: median / max: 44 / 50
- repo Gini: 0.299
- top-10 repos share of turns: 5.9%
- session dates found in file names: 7679 turns; range 2024-12-21 to 2026-08-09
- translated prompts: 2604 (31.0%)
- has_tool_use: 4388 (52.2%)
- has_generated_code: 4762 (56.7%)
- empty agent response: 528 (6.3%)
- prompt words: median: 15
- response chars: median / 90th pct: 2022 / 15795

## Labels (rule = current rule labeler; LLM = corpus LLM labels)
- LLM-labeled turns: 8404
- LLM calls failed: 0
- agreement: 52.1%
- kappa: 0.349
- implement: rule / LLM prevalence; both/either: 52.9% / 34.2%; 2340/4980 = 47.0%
- debug: rule / LLM prevalence; both/either: 15.2% / 30.2%; 995/2815 = 35.3%
- explain: rule / LLM prevalence; both/either: 16.9% / 17.5%; 517/2372 = 21.8%
- review: rule / LLM prevalence; both/either: 4.3% / 4.1%; 46/664 = 6.9%
- tool_only: rule / LLM prevalence; both/either: 7.2% / 5.7%; 198/882 = 22.4%
- other: rule / LLM prevalence; both/either: 3.5% / 8.4%; 285/714 = 39.9%

Confusion (rows = rule, cols = LLM):

| action_type   |   implement |   debug |   explain |   review |   tool_only |   other |
|:--------------|------------:|--------:|----------:|---------:|------------:|--------:|
| implement     |        2340 |    1197 |       675 |      115 |         110 |      12 |
| debug         |         128 |     995 |        63 |       28 |          51 |      11 |
| explain       |         215 |     228 |       517 |       99 |          79 |     283 |
| review        |          72 |      29 |        72 |       46 |          38 |     106 |
| tool_only     |         116 |      85 |       135 |       59 |         198 |       9 |
| other         |           0 |       0 |         6 |        0 |           2 |     285 |

## Blind random reference set (100 sampled, 2 excluded as used in drafting)
- graded turns scored: 98
- grader label counts: {'debug': 37, 'implement': 35, 'explain': 13, 'review': 8, 'tool_only': 3, 'other': 2}
- confidence: {'high': 77, 'medium': 19, 'low': 2}
- rule labeler: 45.9%, kappa 0.251; high-confidence rows 53.2% (n=77)
- earlier definition text: 84.7%, kappa 0.786; high-confidence rows 90.9% (n=77)
- detailed codebook: 85.7%, kappa 0.801; high-confidence rows 89.6% (n=77)
- operational codebook, before fix: 82.7%, kappa 0.760; high-confidence rows 88.3% (n=77)
- operational codebook (production): 86.7%, kappa 0.814; high-confidence rows 92.2% (n=77)
- McNemar llm_v2_legacy vs PROMPT_BOOK: 3 / 4, p=1.000
- McNemar llm_v2_legacy vs PROMPT_STRICT: 6 / 4, p=0.754
- McNemar llm_v2_legacy vs PROMPT_STRICT_FIXED: 2 / 4, p=0.688
- McNemar PROMPT_BOOK vs PROMPT_STRICT: 6 / 3, p=0.508
- McNemar PROMPT_BOOK vs PROMPT_STRICT_FIXED: 2 / 3, p=1.000
- McNemar PROMPT_STRICT vs PROMPT_STRICT_FIXED: 1 / 5, p=0.219
- rule labeler per-category recall: {'implement': 0.71, 'debug': 0.35, 'explain': 0.54, 'review': 0.0, 'tool_only': 0.0, 'other': 0.0}

## Definition sensitivity across the whole corpus (same model and inputs, two definition texts)
- turns labeled by both texts: 8398
- same label: 84.2%, kappa 0.788
- labels changed: 1330
- largest changes: {'implement -> debug': 316, 'implement -> explain': 188, 'review -> explain': 162, 'tool_only -> explain': 84, 'implement -> tool_only': 50, 'explain -> debug': 49}
- prevalence under the earlier text: {'implement': 39.5, 'debug': 26.6, 'explain': 13.5, 'other': 7.7, 'review': 6.8, 'tool_only': 5.8}

## Disagreement clusters
- disagreements: 4023
- C0_empty_response: 254 (6.3%)
- C1_bugfix_missed_by_regex: 1148 (28.5%)
- C2_code_shown_not_applied: 870 (21.6%)
- C3_tool_turn_outcome_unread: 371 (9.2%)
- C4_work_invisible_to_regex: 372 (9.2%)
- C5_broken_or_nonhuman_turn: 360 (8.9%)
- C6_prompt_wording_vs_response_intent: 398 (9.9%)
- C7_problem_word_not_problem_turn: 250 (6.2%)

## Adjudicated contested turns
- rows matched: 96
- human label counts: {'debug': 28, 'review': 16, 'explain': 15, 'implement': 13, 'tool_only': 12, 'other': 12}
- LLM (corpus): 60.4%, kappa 0.505
- rule (current): 36.5%, kappa 0.214
- LLM codebook A: 53.1%, kappa 0.415
- LLM codebook B: 47.9%, kappa 0.347

## Definition sensitivity (200 turns, 40 from each of C1, C2, C4, C6, C7)
- codebook A vs codebook B (full length): n=200, same label 80.0%, kappa 0.735, changed 40
- B0 (applied-change test not tightened) vs B (3k chars): n=200, same label 86.0%, kappa 0.812, changed 28
- B0 original vs paraphrase: n=200, same label 88.0%, kappa 0.838, changed 24
- B original vs paraphrase: n=200, same label 89.0%, kappa 0.852, changed 22
- A: 3,000 vs 20,000 chars: n=200, same label 92.0%, kappa 0.894, changed 16
- B: 3,000 vs 20,000 chars: n=200, same label 92.0%, kappa 0.892, changed 16
- corpus LLM label vs codebook A (full length): n=200, same label 76.0%, kappa 0.683, changed 48
- G_v3_full label counts: {'debug': 62, 'explain': 54, 'implement': 52, 'review': 19, 'tool_only': 12, 'other': 1}
- G_v41_full label counts: {'debug': 67, 'explain': 67, 'implement': 30, 'review': 18, 'tool_only': 11, 'other': 7}
- A: temp 0 vs 0.7 (50 turns): same 98.0%, kappa 0.973
- B0: temp 0 vs 0.7 (50 turns): same 98.0%, kappa 0.973

## Review sub-clusters (silhouette, best over k=2..8)
- all-MiniLM-L6-v2 | prompt | review (either labeler) (n=580): 0.056 at k=2
- all-MiniLM-L6-v2 | prompt | implement (both labelers), same n (n=580): 0.066 at k=2
- all-MiniLM-L6-v2 | prompt | all turns, same n (n=580): 0.054 at k=2
- all-MiniLM-L6-v2 | response | review (either labeler) (n=580): 0.074 at k=4
- all-MiniLM-L6-v2 | response | implement (both labelers), same n (n=580): 0.112 at k=2
- all-MiniLM-L6-v2 | response | all turns, same n (n=580): 0.083 at k=4
- nomic-ai_CodeRankEmbed | prompt | review (either labeler) (n=580): 0.039 at k=2
- nomic-ai_CodeRankEmbed | prompt | implement (both labelers), same n (n=580): 0.038 at k=2
- nomic-ai_CodeRankEmbed | prompt | all turns, same n (n=580): 0.032 at k=2
- nomic-ai_CodeRankEmbed | response | review (either labeler) (n=580): 0.065 at k=2
- nomic-ai_CodeRankEmbed | response | implement (both labelers), same n (n=580): 0.069 at k=2
- nomic-ai_CodeRankEmbed | response | all turns, same n (n=580): 0.066 at k=2
- review turns (non-empty) by source: {'LLM only': 294, 'regex only': 240, 'both': 46}

## Extraction defects
- prompts whose stored translation was an error page: 225
- has_tool_use before / after legacy-format detection: 1938 / 4388
