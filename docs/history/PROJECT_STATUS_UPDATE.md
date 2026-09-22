# Project Status Update — Predicting Coding Agent Behavior from Developer–AI Interaction Histories

**Prepared for:** PI update
**Covers:** full working session, chronological
**Repo:** `medhab123/Prompt_Research`, branch `cursor/local-prompt-analysis-and-cleaning`

---

## TL;DR

- Rebuilt the data pipeline (mining → cleaning → labeling) with several real bugs found and fixed, most importantly a parser bug that was silently corrupting ~19–30% of agent responses.
- Retargeted the ML pipeline from an old, circular label (`primary_intent`) to the actual research-spec target (`action_type`), added a repo-held-out evaluation and a prompt-only vs. conversation-history comparison.
- **Headline finding (now validated on corrected data):** prompt text alone is weak signal (barely beats a keyword rule), but adding the previous turn's context improves macro-F1 by ~56% (0.223 → 0.348). This holds after fixing the data bug, though the effect size is smaller and more credible than the first (buggy-data) measurement of ~74%.
- Built an LLM-assisted relabeling pipeline (Gemini/Groq) to validate the rule-based labels; current agreement (Cohen's κ) is 0.323 — "fair," with one class (`review`) showing a real, unresolved definitional ambiguity.
- **Blocking on:** a human review pass (you/me) to adjudicate ~115 disagreement cases, which determines whether we trust the rule-based labels, the LLM labels, or need to redefine a category.
- Found two directly relevant papers using the **same SpecStory + GitHub-mining methodology** — worth reading before finalizing the proposal's framing (see Related Work below).

---

## Chronological Log

### 1. Initial data quality pass
Started from an existing SpecStory-mining Colab notebook and a ~3,415-row dataset with visible problems: 50%+ of rows from 5 repositories, and prompts that were pure conversational filler ("ok", "continue", "sounds good") carrying no predictive signal.

**Fixes:**
- Replaced flat/no-cap sampling with a **stratified per-repository cap**: repos over the cap are downsampled, but stratified by label/signal so a repo's *mix* of behaviors is preserved rather than randomly gutted. Reduced top-5-repo share from ~34–50% to ~3%.
- Broadened the confirmation/continuation-noise filter using real corpus examples (not just a naive keyword list, which was over-broad and would have deleted substantive messages that merely *start* with "ok" or "yes").

### 2. Local pipeline execution (moved off Colab)
Set up the pipeline to run directly from this environment instead of Google Colab: `.env`-based token loading (`GITHUB_TOKEN`), Windows console UTF-8 fixes (the CLI was crashing on any emoji/non-ASCII content), and verified end-to-end mining + extraction runs.

### 3. Multi-language handling
~35% of the corpus was non-English (Chinese, Russian, French, German, Spanish, Turkish, Arabic). Rather than drop this content, wired machine translation into the cleaning step — non-English prompts are translated for modeling, with the original text preserved in `prompt_text_original`.

### 4. Excel/mojibake bug
User reported garbled text (`è‡ªå®šä¹‰...`) when opening exported CSVs. Root cause: CSVs were written as plain UTF-8 without a byte-order mark, so Excel guessed Windows-1252 encoding. Fixed by switching every CSV writer in the codebase (~20 call sites) to `utf-8-sig`.

### 5. Large-scale mining run
Discovered 454 candidate repositories via GitHub Search API (repo-search + code-search signals), extracted SpecStory turn pairs from 263 of them with usable content — 67,570 raw turn pairs.

### 6. Methodology upgrade — retargeting the ML pipeline
Discovered the existing `ml_research/` code predicted `primary_intent`, an older regex-derived label already known (from prior internal notes) to be circular — TF-IDF trivially "wins" because it memorizes the same keyword patterns used to generate the label. Built a new module (`action_prediction.py`) that:
- Targets `action_type` (the actual spec label: implement/debug/explain/review/tool_only/other) instead.
- Excludes response-side signals (`has_generated_code`, `code_block_count`, etc.) from model inputs, since those partly *define* the label — using them as features would be circular.
- Adds a **repo-held-out** split (`GroupShuffleSplit`, zero repo overlap between train/test).
- Adds the **prompt-only vs. prompt+history** comparison the spec calls for, using leakage-safe prior-turn features (`prev_action_type`, `prev_turn_had_code`, session code-rate — all `shift(1)`'d so nothing from the current turn leaks in).
- Adds two real baselines (majority-class, prompt-only keyword rule) so every trained model has an honest bar to clear.

**First result (later found to be on buggy data):** prompt-only macro-F1 0.237, history-augmented macro-F1 0.412 — an apparent ~74% relative improvement from adding context.

### 7. Self-critique
At user request, ran a harsh methodological review of the above. Surfaced several real issues:
- **Possible label circularity**: the rule-based label is partly generated from the same keyword matching used as the "prompt-only baseline," so beating that baseline isn't fully independent evidence.
- **No human validation** of any labels yet.
- **No statistical rigor**: single train/test split, no cross-validation, no confidence intervals, small minority classes (`debug` n≈90 in test).
- **No ablation** isolating which history feature does the work.
- Found and fixed **3 real data-corruption bugs** during the session itself (see below), which is itself evidence more may exist.

### 8. LLM-assisted labeling infrastructure
Built a provider-agnostic (Gemini/Groq) batch labeler (`llm_labeling.py`) with: structured JSON output, exemplar-grounded few-shot prompting (real corpus examples, not generic ones), content-hash caching, checkpointing, and Cohen's κ scoring against the rule-based labels — directly implementing the "LLM-assisted weak supervision + human validation" methodology your spec already calls for.

Along the way, fixed a subtler version of the same caching bug pattern found earlier: a malformed/failed API response was being cached as if it were a valid `"other"` label, silently poisoning results. Fixed so only genuinely successful, parsed labels are ever cached.

### 9. First kappa run and the parser bug discovery
First LLM-labeling validation sample (Groq): **κ = 0.384**, but two classes (`review`, `tool_only`) showed only 8% agreement between the LLM and the rule. Investigating *why* — not just accepting a low number — led to the real finding:

**~18.7% of the entire corpus had a blank agent response.** Tracing one blank-but-`has_tool_use=True` example back to the raw SpecStory markdown showed the actual response was substantial (multiple file edits, a build step, a closing summary) — the data was there, but the parser was silently dropping it.

**Root cause, `specstory_parser.py`:** SpecStory logs split one logical agent turn into several consecutive marker blocks (one per tool call, plus prose). The parser only read the *first* block after a user turn, discarding the rest — including the block with the actual final answer. Compounded by a second bug: a block that was purely a tool call (no prose) was being sanitized down to an empty string entirely, rather than preserved as a signal that something happened.

**Fix:** merge all consecutive assistant blocks into one logical turn; replace tool-call blocks with a short `[tool: Edit]`-style marker instead of stripping them to nothing.

### 10. Re-extraction and validated results
Re-ran extraction on the corrected parser across all 454 repos (with a resumable/checkpointed, concurrent implementation — see below), producing a corrected 68,943-row raw corpus → 8,404-row final corpus across 251 repos after cleaning.

**Effect of the fix on label distribution** (this is the strongest direct evidence the bug was real and mattered):

| Class | Before fix | After fix |
|---|---|---|
| `other` | 8.6% | 3.5% |
| `review` | 7.5% | 4.2% |
| `tool_only` | 8.1% | **16.9%** |
| `implement` | 46.5% | 53.0% |

`other`/`review` (populated by the blank-response fallback) shrank; `tool_only` (which needs real response content to detect) more than doubled.

**Corrected headline ML result:**

| | Accuracy | Macro-F1 |
|---|---|---|
| Majority-class baseline | 43.0% | 0.100 |
| Prompt-only keyword rule | 24.8% | 0.205 |
| Best prompt-only trained model | 27.9% | **0.223** |
| Best prompt+history model | 43.2% | **0.348** |

The core finding survives — history still helps substantially (~56% relative improvement, down from the inflated ~74%) — but every number is honestly lower now that the artifact is gone, and the model that wins changed from a complex one (MLP/XGBoost) to plain logistic regression, which is arguably a healthier sign this is a real, modest signal rather than something only a complex model could exploit.

### 11. Engineering: concurrency
Both extraction and LLM-labeling were fully sequential (one repo / one API call at a time). Added thread-pool concurrency to both:
- **Extraction** (GitHub API, generous 5000/hr limit): straightforward `max_workers` concurrency, clear speedup.
- **LLM labeling** (Groq free tier, tight token-per-minute limits): naive per-worker pacing *backfired* — 4 workers each independently pacing at the same interval multiplied the effective send rate and caused more rate-limit backoff, not less. Fixed with a **shared rate limiter** (a single pacer across all threads, decoupling "how many requests in flight" from "how fast they're sent"). Even so, empirically found this specific API+model tolerates little concurrency (settled on `max_workers=2`) — a real constraint of the free tier, not an engineering gap.

### 12. Repo hygiene
Everything above was local-only and uncommitted for most of the session (the Colab notebook was cloning a stale GitHub version and failing). Committed 44 files (~6,400 lines) and pushed to the feature branch (not `main`, which is untouched) after review. Rebuilt the Colab notebook to call the actual package functions directly (not duplicate logic inline, which is exactly how the parser bug went unnoticed for a long time in the old notebook).

### 13. Current LLM-labeling validation status
Re-ran the labeling sample on the corrected data: **κ = 0.323**, and critically, **`tool_only` agreement jumped from 8% to 66%** — direct confirmation the parser fix improved data quality exactly where predicted. `review` is still stuck at 8% agreement, unchanged by the fix — this is a genuine taxonomy ambiguity, not a data bug.

A prioritized human-review sheet (115 disagreement rows, `review`-involving ones first) is ready and **awaiting your/my manual adjudication** — this is the current blocking step.

---

## Where Things Stand Right Now

| Component | Status |
|---|---|
| Mining + extraction (fixed parser) | Done, 251 repos, 8,404 rows |
| Cleaning (stratified cap, translation) | Done |
| `action_type` prediction pipeline | Done, results above |
| LLM-assisted relabeling (300-row sample) | Done (258/300 labeled; 42 pending a Groq quota reset) |
| **Human review of disagreements** | **Not started — next action** |
| Full-corpus LLM relabeling | Not started, pending human-review outcome |
| Statistical rigor (CV, confidence intervals) | Not done |
| Ablation (which history feature matters) | Not done |
| Fine-tuned transformer / LLM zero-shot upper bound | Not done (optional stretch goal) |

---

## Related Work Found (verified, real citations — not fabricated)

Two papers are **directly relevant** — same data source and mining methodology as this project, from what looks like the same active research group:

1. **Tang, Chen, Fang, Xu, Dhakal, Shi, McMillan, Huang, Li. "Programming by Chat: A Large-Scale Behavioral Analysis of 11,579 Real-World AI-Assisted IDE Sessions."** arXiv:2604.00436 (April 2026). 74,998 messages, 1,300 repos, Cursor + Copilot. Builds a *behavioral* taxonomy (progressive specification, cognitive work redistribution, active collaboration management) — a different axis than our `action_type` (what the agent *did*) but directly comparable as prior taxonomy work. **Read this before finalizing your label definitions** — citing it as precedent (or contrast) strengthens Section 5 of the proposal.
2. **Fang, Zhang, Tang, McMillan, Li, Huang. "From Conversation to Contribution: Characterizing Coding Agent in Open-Source Software."** arXiv:2607.05677 (July 2026). Same SpecStory + GitHub Code Search methodology, 79,172 messages, 1,356 repos — but their research question is about *project-level* collaboration/OSS impact, not per-turn action prediction. Good evidence the mining methodology itself is publication-grade; **our angle (predictive modeling of next action, repo-held-out ML evaluation) appears to be a genuinely distinct contribution**, not a duplicate — but this must be cited and explicitly differentiated.
3. **Ong et al. "RouteLLM: Learning to Route LLMs with Preference Data."** arXiv:2406.18665. Concrete, citable evidence for the model-routing motivation in your Introduction — a trained router achieved >2x cost reduction with minimal quality loss by classifying query difficulty, which is the same underlying idea (classify the request, then route) your project applies to *action type* instead of *difficulty*.
4. General areas still worth a proper search pass (not yet verified with specific papers): dialogue-act/speech-act classification literature (ISO standard dialogue act tagging — arXiv:1806.04327 came up and may be worth citing for taxonomy methodology precedent), and the broader LLM-routing survey **"Doing More with Less: A Survey on Routing Strategies for Resource Optimisation in LLM-Based Systems"** (arXiv:2502.00409) for a comprehensive reference list to mine further citations from.

**Recommendation:** read papers #1 and #2 in full before the next proposal revision — at minimum they need a citation and a one-paragraph "how this differs" note, and #1 in particular may sharpen how you define `action_type` vs. their behavioral categories.

---

## Next Steps, In Order

1. **Human review** — adjudicate the 115 disagreement rows (`outputs/llm_labels_v3/human_review_sheet_PRIORITIZED.csv`), `review`-class ones first. Determines whether rule-based or LLM labels are more trustworthy, or whether `review` needs redefining.
2. Based on (1): decide whether to relabel the full 8,404-row corpus with the LLM and rerun the `action_type` prediction results on that higher-quality label set.
3. Read the two directly-related papers above; update the proposal's Related Work section (currently a placeholder) with real citations and an explicit differentiation statement.
4. Add statistical rigor: repeated repo-held-out splits or `GroupKFold`, report confidence intervals instead of point estimates.
5. Ablation: isolate whether `prev_action_type` alone explains the history-helps finding, or whether the other features (session code-rate, turn position) meaningfully contribute too.
6. Optional, time-permitting: try the code-aware embedding model already available in the codebase (`nomic-ai/CodeRankEmbed`, never actually tested against `action_type`); fine-tune a small transformer as a stronger upper-tier comparison; report LLM zero-shot performance as a reference point for the model-routing framing.
