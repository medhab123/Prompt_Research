# Findings, and where each number comes from

Regenerate the label-validity numbers with `python paper/msr2027/compute_numbers.py`
(writes `paper/msr2027/NUMBERS_draft.md`). Prediction numbers are in
`outputs/ml_action_v4_llm/` and `outputs/ml_action_v4_rule/`.

## Terms

| Name | What it is |
|---|---|
| RULE | keyword labeler, `src/github_repo_miner/action_labeling.py` (current version) |
| JUDGE | LLM labeling with the production prompt, `prompts/action_type_strict.md`, `gemini-3.1-flash-lite`, temperature 0 |
| earlier prompt | the definition text that produced `outputs/llm_labels_v3/`; kept for the definition-sensitivity comparison |
| HUMAN | 100 randomly sampled turns labeled blind, `outputs/gold_v2/`; 98 usable (two were read while drafting the definitions) |

## For the label-validity paper (MSR)

**1. The judge is accurate; the keyword labeler is not a cheap substitute.**
Against HUMAN: judge 86.7% / κ 0.81 (92.2% on the 77 high-confidence turns);
RULE 45.9% / κ 0.25. RULE recovers none of the `review`, `tool_only` or `other`
turns and a third of `debug`. Source: `outputs/gold_v2/SCORES.md`.

**2. Three candidate judge prompts are statistically tied.** 86.7% / 85.7% /
84.7%; exact McNemar p ≥ 0.22 for every pair. Accuracy alone does not choose a
definition text. Source: same.

**3. The two constructions disagree on half the corpus.** 52.1% agreement,
κ 0.35 over 8,404 turns. Prevalence: `implement` 52.9% (RULE) vs 34.2% (JUDGE);
`debug` 15.2% vs 30.2%. Of turns either calls `review`, both do for 6.9%.

**4. Rewording the definitions relabels one turn in six.** Same model, inputs
and temperature, different definition text: 15.8% of 8,398 turns changed
(κ 0.79); `implement` → `debug` 316, `implement` → `explain` 188,
`review` → `explain` 162. Controls on contested turns: temperature 0 → 0.7
changes 2%, truncating responses at 3,000 characters changes 8%, paraphrasing a
definition text changes 11%.

**5. The disagreements are mechanical.** All 4,023 fall into eight mechanisms;
four of them (66.3%) come from RULE reading surface features: bug report with no
keyword 28.5%, code shown but never applied 21.6%, prompt wording vs response
function 9.9%, problem word with no problem 6.2%.
Source: `outputs/disagreements/rule_vs_llm_disagreements.csv`.

**6. `review` is not hidden sub-behaviors.** Of 580 review turns (non-empty
responses, either labeler), 294 are the judge's alone, 240 RULE's alone, 46
shared. Best silhouette 0.039–0.074 under two encoders, never more than 0.01
above the `implement` or random baseline. Source: `outputs/review_subclusters/`.

**7. Comparing constructions surfaced three extraction defects** no test caught:
the response-truncating parser bug (48.4% → 29.8% blank responses), tool calls in
a legacy format missed on 2,450 turns, and 225 prompts whose stored translation
was an error page.

## For the prediction paper (FORGE)

**8. Trained classifiers predict the next action weakly.** 5 repeats × 5
repository-held-out folds, logistic regression: macro-F1 0.346 against JUDGE
labels, 0.338 against RULE labels; prompt-only 0.334 and 0.285. Bootstrap CIs
about ±0.013. Source: `outputs/ml_action_v4_*/stats/STATS.md`.

**9. The apparent value of conversation history is mostly the labeler's own
autocorrelation.** History adds +0.011 macro-F1 [0.008, 0.015] against JUDGE
labels but +0.050 [0.045, 0.056] against RULE labels — 4.5× larger, with
non-overlapping intervals, and RULE wins in all 25 splits. RULE labels depend on
response-side surface features that persist between neighboring turns.

**10. A zero-shot LLM beats every trained pipeline, and that is not
self-agreement.** On the same held-out split (1,734 turns): against JUDGE labels
zero-shot 0.494 vs 0.366 trained (difference CI [0.084, 0.162]); against RULE
labels the two tie (0.315 vs 0.330, CI spans zero). The tie suggested the
advantage might be the judge's own idiosyncrasies, so we tested against HUMAN
labels, which no labeler produced: zero-shot 0.584 macro-F1 / 77.6% accuracy vs
0.367 / 58.2% for the best trained model, with models trained on repositories
that contribute none of the hand-labeled turns. The next action is substantially
predictable from the prompt; embedding-plus-classifier pipelines do not capture
it. Sources: `outputs/ml_action_v4_*/zeroshot/`, `outputs/gold_v2/PREDICT_HUMAN.md`.

**11. Predicting is harder than labeling.** The judge, which sees the response,
matches HUMAN on 86.7% of turns; the same model predicting from the prompt alone
reaches 77.6%. The gap is what the response itself contributes.

## Open items

- Inter-annotator agreement (DONE): the second annotator re-labeled 40 turns
  independently. Raw agreement 92.3%, Cohen's κ 0.897 on the 39 in common; the
  three disagreements are all on the `explain` boundary. Candidate prompts rank
  identically against either annotator, so the production-prompt choice does not
  depend on whose labels are the reference. Caveat: that annotator took part in
  the consensus pass, so this is nearer a test–retest check and bounds
  reliability from above; no third annotator will be added (decided), so the
  papers present the reference labels as a two-person consensus with an
  optimistic reliability estimate. Source: `outputs/gold_v2/INTERANNOTATOR.md`.
- Mechanism validation: `outputs/disagreements/mechanism_validation_sheet.xlsx`
  (60 turns) is waiting for hand-coding, to check the automatic assignment.
- Multi-model zero-shot (DONE): three models, two families. Against human
  labels gemini-3.1-flash-lite 0.584 / 77.6%, gemini-3.5-flash 0.544 / 77.6%,
  gpt-oss-120b 0.422 / 63.3% (0.397 at low reasoning effort), vs 0.367 / 58.2%
  for the best supervised pipeline. On the 280 held-out turns all three covered,
  only gemini-3.1-flash-lite is clearly ahead of the supervised pipeline. So the
  direction generalizes and the margin is model-dependent. gemini-3.5-flash
  against rule labels could not be completed (1,423 of 1,734 calls failed on
  quota) and is not reported. Sources: `outputs/ml_action_v4_*/zeroshot*/`.
- n = 98 for HUMAN, so those CIs are wide (finding 10 spans 0.42–0.73).
- Replication package and the generative-AI disclosure.
