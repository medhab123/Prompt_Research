# Predicting and Understanding Coding Agent Behavior from Developer–AI Interaction Histories

**A Research Proposal on Semantic and Contextual Prediction of AI Coding Agent Actions**

---

## Abstract

Modern AI coding assistants — Cursor, GitHub Copilot, Claude Code, and similar tools — respond to a wide range of developer requests using largely undifferentiated, reactive strategies: every prompt is routed through the same general-purpose model and the same tool-selection logic, regardless of whether the developer wants a new feature implemented, a bug diagnosed, an unfamiliar module explained, or existing code reviewed. This project asks whether that undifferentiated treatment is necessary, or whether the *language and structure* of developer–AI conversations already contain enough signal to anticipate what an agent is about to do next.

We formulate this as a supervised, multi-class prediction problem: given a developer's current prompt, and optionally the preceding turns of a coding session, predict the agent's next **action type** — `implement`, `debug`, `explain`, `review`, `tool_only`, or `other`. Using a corpus of **8,382 real developer–agent turns mined from 252 public GitHub repositories** containing SpecStory session logs, we show that (1) the prompt text alone carries only weak signal for this task — a trained semantic model barely matches a naive keyword rule — but (2) adding lightweight features describing the *prior turn* in the same session nearly doubles predictive performance (macro-F1 0.24 → 0.41), evaluated under a strict repository-held-out split that prevents the model from exploiting repo-specific vocabulary. These results suggest that short-range conversational context, not prompt semantics in isolation, is the dominant signal for anticipating agent behavior — with direct implications for adaptive model routing and proactive tool preparation in AI coding assistants.

---

## 1. Introduction and Motivation

AI coding agents have moved beyond single-shot code generation into sustained, multi-turn collaborations with developers: they implement features, chase down bugs, explain unfamiliar codebases, critique existing implementations, and autonomously search and navigate repositories. Despite this behavioral range, most production systems treat every incoming request identically — the same model, the same context budget, the same tool-availability defaults — regardless of what kind of interaction is actually unfolding.

This is a missed opportunity along three dimensions:

- **Cost and latency.** A one-line clarifying question does not need the same reasoning budget as a multi-file refactor. If a system could anticipate that a request is `explain`-type before generating a response, it could route it to a smaller, faster, cheaper model.
- **Preparedness.** If a system could anticipate that a request will require repository exploration (`tool_only`) or iterative debugging (`debug`), it could proactively index relevant files, warm up a sandbox, or pre-fetch build/test tooling — before the agent even starts reasoning.
- **Adaptivity.** Understanding the *shape* of a developer's request — not just its literal content — is a prerequisite for any assistant that claims to adapt to how a developer actually works, rather than responding identically to every message.

The central research question:

> **To what extent can semantic and contextual representations of developer–AI interactions predict the next operational action of an AI coding agent?**

And the working hypothesis this project tests:

> Developer–AI interaction patterns contain measurable semantic and contextual signals that can be used to predict future agent behavior — and *context* (prior conversational turns) contributes signal beyond what is available from the current prompt in isolation.

*(This section is intentionally free of literature citations. Before submission, this is the place to situate the work against prior literature on: LLM agent orchestration / model routing, intent classification in developer-tool interactions, weak/rule-based supervision for NLP labeling, and prior empirical studies of AI-pair-programming transcripts. Add 5–10 citations from your own lit search here.)*

---

## 2. Research Questions

### RQ1 — Predictability
Can a supervised model, trained on developer prompts and interaction metadata, predict the agent's next action category (`implement`, `debug`, `explain`, `review`, `tool_only`, `other`) at a rate meaningfully above chance and above simple lexical baselines?

### RQ2 — Context window dynamics
Does adding conversational history — specifically, the action taken on the *prior* turn of the same session, whether that turn involved code or tool use, and the session's rolling code-generation rate — improve prediction accuracy relative to a prompt-only model? How large is that improvement, and is it consistent across action classes?

### RQ3 — Generalization
Do the semantic representations learned from one set of repositories generalize to conversations happening in entirely unseen repositories, or is apparent performance an artifact of repository-specific vocabulary (e.g., project jargon, file-naming conventions)?

### RQ4 — Where does the signal live?
Is the predictive signal primarily semantic (captured by dense sentence embeddings), primarily lexical (captured by simple keyword rules and TF-IDF), or primarily structural (captured by session-level metadata such as turn position and prior action)? Disentangling these matters for both scientific interpretability and for what a production system would actually need to implement to realize any of the practical benefits above.

---

## 3. Related Work *(placeholder — add citations)*

This project sits at the intersection of three lines of work that should be cited explicitly once identified:

1. **LLM agent orchestration and model routing** — systems that select among multiple models or tools based on inferred task type or complexity.
2. **Intent / dialogue-act classification** — the broader NLP literature on classifying the communicative function of an utterance, applied here to a developer↔agent setting rather than open-domain dialogue.
3. **Mining software repositories for developer behavior** — prior empirical work studying commit messages, code review comments, or IDE telemetry as proxies for developer intent and workflow.

*(Note: I am not fabricating citations here. Populate this section from your own search of ACL/NeurIPS/ICSE/FSE/MSR venues before submission — a proposal with invented references would undermine its credibility.)*

---

## 4. Dataset

### 4.1 Source and collection

The corpus is built from **SpecStory** session logs — Markdown transcripts that Cursor/Copilot-style IDE integrations write to `.specstory/history/` in a developer's repository, capturing the full back-and-forth of a coding-agent session (developer prompts, agent responses, generated code, tool invocations) verbatim. Because these logs are opt-in artifacts that developers sometimes commit to version control, they are discoverable via the GitHub Search API.

**Discovery.** Six GitHub Search API queries (three repository-search signals: `specstory in:readme`, `topic:specstory`, `path:.specstory`; three code-search signals: `path:.specstory/history`, `path:.specstory`, `extension:md path:.specstory`) were run against the public GitHub corpus, yielding **454 unique candidate repositories** after deduplication across signals.

**Extraction.** Each candidate repository was crawled for `.specstory/history/*.md` files via the GitHub Contents API; files were parsed into discrete developer-prompt / agent-response turn pairs, capturing: prompt text, agent response text, generated code blocks (count, language, length), tool-use flags, and repository/session identifiers. **263 of the 454 candidates** contained at least one extractable turn, yielding **67,570 raw turn pairs**.

**Repository-concentration control.** A small number of repositories otherwise dominate raw turn counts by one to two orders of magnitude (one repository alone accounted for 30,000+ turns in the raw pull — almost certainly a degenerate/looping logging artifact rather than organic multi-turn conversation, not a real usage pattern worth training on at that volume). To prevent the corpus — and any model trained on it — from simply memorizing the vocabulary of a handful of chatty repositories, a **stratified per-repository cap** was applied: each repository's rows above the cap are downsampled, but the downsampling is stratified by `action_type` (not uniform-random), so a repository's *mix* of implement/debug/explain/... turns is preserved in miniature rather than being distorted by chance. This reduced the top-5-repository share of the corpus from roughly one-third to **3%**, while leaving repositories under the cap untouched.

**Multi-language handling.** Approximately 35% of surviving prompts were in a language other than English (predominantly Chinese, with Russian, French, German, Spanish, Turkish, and Arabic also present). Rather than discard non-English content — which would both shrink the corpus and introduce a systematic bias against certain developer populations — non-English prompts were machine-translated to English for the modeling pipeline, with the original-language text preserved in a separate column for audit and for any future multilingual modeling extension.

**Final corpus:** after noise filtering, exact-duplicate removal, and the per-repository cap: **8,404 developer–agent turns across 251 repositories**, organized into multi-turn sessions.

**Parser correction (post-hoc, see Section 8).** An earlier pass through this pipeline had a bug in the SpecStory transcript parser: one logical agent turn is often split across several consecutive markdown blocks (one per tool call, plus prose), and the parser was only reading the first block, discarding the rest — including, often, the turn's actual closing summary. This silently emptied a substantial fraction of `agent_response` values, which the label rule then routed into `other`/`review` by default. The parser has been fixed (all consecutive assistant blocks are now merged; tool-call blocks are summarized rather than stripped to nothing) and the corpus below reflects the corrected extraction. The numbers in this section and Section 7 are the corrected, validated ones.

### 4.2 Class distribution

| Action | Count | % of corpus |
|---|---:|---:|
| `implement` | 4,450 | 53.0% |
| `explain` | 1,598 | 19.0% |
| `tool_only` | 1,417 | 16.9% |
| `review` | 352 | 4.2% |
| `other` | 295 | 3.5% |
| `debug` | 292 | 3.5% |

The imbalance is not an artifact of sampling — it reflects the genuine skew of real coding-agent sessions, where "write/modify code" requests dominate and narrowly-scoped debugging exchanges are comparatively rare. This is precisely why **macro-F1** (unweighted mean of per-class F1) rather than raw accuracy is the primary evaluation metric throughout: a classifier that always predicts `implement` would already be "right" ~53% of the time without learning anything.

*(For reference: before the parser fix, the same pipeline reported implement 46.5%, explain 25.4%, other 8.6%, tool_only 8.1%, review 7.5%, debug 3.9% — `other` and `review` were inflated by the blank-response bug; `tool_only`, which requires real response content to detect, more than doubled after the fix.)*

---

## 5. Label Construction Methodology

`action_type` is not present in the raw data and must be constructed — this project uses a three-tier strategy consistent with standard weak-supervision practice:

**Tier 1 — Metadata-informed rule-based labeling.** A deterministic priority-ordered rule (`action_labeling.infer_agent_action`) assigns one of the six categories using: (a) whether the agent invoked tools without producing code (`tool_only`), (b) the volume of generated code in the response (multi-block or long single-block responses default to `implement` regardless of prompt phrasing), and (c) prompt-side keyword regexes (debug/explain/review/implement-flavored language) to disambiguate borderline single-code-block or no-code responses. This produces the label set analyzed in Sections 4 and 7.

**Tier 2 — LLM-assisted relabeling (in progress).** To move beyond keyword-rule brittleness, a parallel labeling pass uses an LLM (Gemini/Groq, selected for free-tier accessibility) prompted with the category definitions, the prompt, the response, and the same metadata signals, returning a structured label with a confidence score and a one-sentence rationale. Content-hash caching means every row is labeled at most once regardless of how many times the pipeline is (re-)run. A stratified sample (target: 300 rows, ~50 per class) is being labeled first specifically to compute agreement against the rule-based labels before committing the time/quota cost to relabeling the full corpus.

**Tier 3 — Human validation.** A stratified review sheet (LLM label, rule-based label, rationale, blank human-adjudication column) is generated automatically from the LLM-labeling pass, for a human annotator to adjudicate disagreements and estimate true labeling quality via Cohen's κ. *(Status: infrastructure built; the human annotation pass itself has not yet been performed — this is the most important remaining piece of methodological due diligence before treating any single tier's labels as ground truth in a final write-up.)*

**Leakage discipline.** Because the rule-based label is itself partly derived from response-side signals (`has_generated_code`, `code_block_count`, response length), those same columns are explicitly excluded from the model's *input* features throughout Section 6 — using them as predictors would make the prediction task circular (predicting a label using the same signal used to construct it) rather than a genuine test of whether the *prompt and context* anticipate the *response*.

---

## 6. Modeling Approach

### 6.1 Representations compared

| Representation | Description |
|---|---|
| `tfidf` | Classical bag-of-words / TF-IDF over the current prompt (1–2 grams) |
| `sbert_prompt_only` | Dense sentence embedding (`all-MiniLM-L6-v2`) of the current prompt alone |
| `sbert+prompt_metadata` | Prompt embedding concatenated with prompt-derived surface features (length, word count, question marks, imperative-opener detection) |
| `sbert+history` | Prompt embedding concatenated with **prior-turn-only** features: the action type of the immediately preceding turn (one-hot), whether that turn had code/tool use, the session's rolling code-generation rate up to (not including) the current turn, and the turn's position in the session |
| `sbert+prompt_metadata+history` | The full combination |

All history features are computed via a strict `shift(1)` over turn order within each session at data-preparation time, so by construction they cannot see any information from the current turn or later — a leakage-safe operationalization of "conversational context available at prediction time."

### 6.2 Models

Each representation is evaluated with four model families of increasing complexity: **logistic regression** (linear baseline, class-balanced), **random forest**, a **multi-layer perceptron** (256→128 hidden units), and **gradient-boosted trees (XGBoost)**. This ladder is deliberately conservative for a ~8K-row, 6-class problem — a full fine-tuned transformer was considered and set aside for now as disproportionate to the dataset size relative to the marginal expected gain over embedding + classifier approaches; it remains a natural extension (Section 9).

### 6.3 Baselines

Two baselines anchor every comparison: a **majority-class** baseline (always predict `implement`) and a **prompt-only keyword-rule** baseline (a small regex cascade over the current prompt's surface language alone, deliberately given no access to the response) — the bar any learned representation has to clear to be worth reporting as "the model learned something."

### 6.4 Evaluation protocol

Repositories — not individual rows — are split into train and test sets via `GroupShuffleSplit` (80/20), guaranteeing **zero repository overlap** between train and test. This directly operationalizes RQ3: any performance the model achieves must come from patterns that generalize across projects, not from having memorized a given repository's file names, module vocabulary, or house style. All representation variants in a given comparison are evaluated on the *same* train/test split, so differences in outcome are attributable to the representation, not to a lucky or unlucky partition.

Primary metric: **macro-F1**. Secondary: accuracy, per-class precision/recall, confusion matrices.

---

## 7. Results to Date

*(These are real, already-obtained results from the current pipeline, on the corrected corpus described in Section 4 — not projected or illustrative numbers.)*

**Split:** 200 training repositories / 51 held-out test repositories, 1,734 test rows, confirmed zero repository overlap.

### 7.1 Headline comparison

| Representation | Model | Accuracy | Macro-F1 |
|---|---|---:|---:|
| — (majority class) | — | 43.0% | 0.100 |
| Prompt-only keyword rule | rule-based | 24.8% | 0.205 |
| `sbert_prompt_only` | Logistic Regression | 27.9% | 0.223 |
| `sbert+prompt_metadata+history` | Logistic Regression | 43.2% | **0.348** |
| `sbert+prompt_metadata+history` | XGBoost | **57.6%** | 0.304 |

**Finding 1 (RQ1, partial support).** A trained model on the prompt alone (macro-F1 0.223) modestly beats the hand-written keyword rule (0.205) applied to the same prompt-only information. The margin is real but thin — semantic embeddings extract some signal beyond lexical pattern matching, but not dramatically more, for this task.

**Finding 2 (RQ2, supported, revised downward from an earlier measurement).** Adding prior-turn context improves macro-F1 by roughly 56% relative (0.223 → 0.348). An earlier pass through this pipeline (before the parser fix in Section 8) measured a larger jump (0.237 → 0.412, ~74% relative) — some of that gain was an artifact of the blank-response bug, which made certain rows trivially easy to predict once the previous turn's (also bug-affected) label was known. The corrected, smaller effect is the one to report: short-range conversational context carries meaningfully more predictive signal than prompt semantics in isolation, though not as large a margin as first measured.

**Finding 3 (RQ3, supported by design).** Because the improvement in Finding 2 is measured on repositories the model has never seen, the gain is attributable to learned structure in developer–agent interaction patterns, not to repository-specific memorization.

**Finding 3b.** The best macro-F1 model changed from a nonlinear model (MLP, on the buggy data) to plain logistic regression (on the corrected data). A simpler model winning is, if anything, a healthier sign that this is a real but modest signal rather than something only a complex model could exploit from noise.

### 7.2 Per-class detail (best model by macro-F1: Logistic Regression, `sbert+prompt_metadata+history`)

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| `implement` | 0.72 | 0.44 | 0.55 | 745 |
| `tool_only` | 0.53 | 0.57 | 0.55 | 372 |
| `explain` | 0.40 | 0.32 | 0.36 | 387 |
| `review` | 0.15 | 0.47 | 0.22 | 68 |
| `debug` | 0.21 | 0.40 | 0.28 | 90 |
| `other` | 0.09 | 0.24 | 0.13 | 72 |

**Finding 4 (RQ4, concrete illustration of why macro-F1 matters).** A separate comparison model (XGBoost, same representation, higher accuracy at 57.6%) achieves that accuracy by aggressively over-predicting the majority class (`implement` recall 0.85) while nearly abandoning rare classes (`debug` recall 0.09, `review` recall 0.13) — full detail in the project's engineering log. The logistic-regression model reported above trades some majority-class recall for meaningfully better minority-class coverage (`debug` recall 0.40, `review` recall 0.47), which is why it wins on macro-F1 despite lower accuracy. For any of the motivating use cases in Section 1 (e.g., routing debugging requests to a specialized agent), a model that only catches 9% of debug turns is close to non-functional for that purpose despite a better headline accuracy number — the exact failure mode macro-F1 as primary metric is designed to surface.

**Finding 5.** `implement` and `tool_only` are the strongest-supported, most learnable classes post-correction (both benefited directly from the parser fix — see Section 4.2). `other` remains the weakest class (F1 0.13), consistent with its definition as a residual, low-signal category rather than a coherent behavioral class. `review` has low precision but improved recall (0.47) — the model now finds more true review turns, at the cost of more false positives, rather than missing the class almost entirely as in the pre-fix results.

---

## 8. Threats to Validity / Limitations

- **Label quality.** An LLM-assisted relabeling pass (Section 5, Tier 2) has now been run on a stratified 300-row sample (258 successfully labeled before hitting a free-tier daily quota). Agreement with the rule-based labels: **Cohen's κ = 0.323** ("fair" on the standard scale). Per-class agreement is uneven: `implement` 60%, `tool_only` 66%, `explain` 38%, `debug` 34%, `other` 100% (n=8), but **`review` only 8%** — a large, specific disagreement concentrated in one class rather than spread evenly, suggesting `review`'s definition itself (not just execution noise) may need revisiting. A human-adjudication pass on the disagreement cases (Tier 3) is prepared but not yet completed; until it is, neither the rule-based nor the LLM labels should be treated as ground truth, and the `action_type` numbers in Section 7 should be read as resting on a `κ ≈ 0.32`-agreement label source.
- **Historical data-quality issue, now fixed.** A parser bug (Section 4.2 note) silently emptied a substantial share of `agent_response` values in an earlier pass through this pipeline, inflating the apparent size of the context-window effect in Finding 2 of Section 7. This was caught by investigating *why* the pre-fix κ (0.384) showed near-zero agreement specifically on `tool_only`/`review`, tracing a sampled disagreement back to the raw transcript, and finding the parser had dropped real content. The pipeline has three prior instances of this general failure mode (see project engineering log) — each caught by manually investigating an anomalous number rather than by systematic testing, which is itself evidence that further undiscovered data issues cannot be ruled out.
- **Selection bias.** The corpus is limited to repositories where a developer both used SpecStory *and* chose to commit the resulting logs publicly — this is not a random sample of all AI-assisted development, and likely skews toward more experimental, open-source-oriented, or process-transparent developers. (Two contemporaneous papers using the same SpecStory + GitHub-mining methodology exist — see Section 3 addendum below — and report comparable corpus sizes, suggesting this selection bias is a property of the data source itself, not an artifact of this project's specific query design.)
- **Session-boundary effects on the context feature.** The per-repository cap (Section 4.1) samples individual *rows* stratified by action type; it does not guarantee that every turn's immediately-preceding turn survived the same cap. Prior-turn features were computed from the complete, uncapped session at extraction time and are therefore still individually valid, but a richer context representation (raw concatenated prior-turn *text*, rather than summary features) would require re-deriving from complete, uncapped sessions — noted as a scoping decision, not an oversight (Section 9).
- **Model scale vs. dataset scale.** ~8,400 rows is workable for classical ML + embeddings but is a genuine constraint on fine-tuning a full transformer without overfitting; this shaped the model-selection decision in Section 6.2.
- **No statistical rigor yet.** All results in Section 7 come from a single train/test split (one seed). No cross-validation, no confidence intervals. The smallest classes (`debug`, `other`, ~70-90 test rows each) have per-class metrics with real sampling variance not currently quantified.
- **No ablation.** The history representation combines several features (`prev_action_type`, `prev_turn_had_code/tools`, session code-rate, turn position). It is not yet established how much of Finding 2's improvement is attributable to `prev_action_type` alone versus the others — a priority open question, since `prev_action_type`'s contribution could partly reflect autocorrelation in how the label itself is generated rather than purely genuine behavioral continuity.

### Addendum to Section 3 (Related Work) — directly relevant prior work found

Two papers, apparently from the same active research group, use the identical SpecStory + GitHub Code Search mining methodology as this project:
- Tang, Chen, Fang, Xu, Dhakal, Shi, McMillan, Huang, Li. *"Programming by Chat: A Large-Scale Behavioral Analysis of 11,579 Real-World AI-Assisted IDE Sessions."* arXiv:2604.00436. Builds a behavioral taxonomy (progressive specification, cognitive work redistribution, active collaboration management) — a different axis than `action_type` but directly comparable prior taxonomy work; worth citing and differentiating from explicitly.
- Fang, Zhang, Tang, McMillan, Li, Huang. *"From Conversation to Contribution: Characterizing Coding Agent in Open-Source Software."* arXiv:2607.05677. Same mining methodology, project-level OSS-collaboration focus rather than per-turn action prediction — evidence the methodology itself is sound, and that this project's specific angle (predictive modeling with repo-held-out evaluation) is not a duplicate of existing work, but must be positioned against it.
- Ong et al. *"RouteLLM: Learning to Route LLMs with Preference Data."* arXiv:2406.18665. Concrete supporting citation for the model-routing motivation in Section 1 — a trained query-difficulty router achieved >2x cost reduction with minimal quality loss, the same underlying idea (classify the request, then route) applied here to action type instead of raw difficulty.

---

## 9. Remaining Work / Next Steps

1. **Human-adjudicate the disagreement cases** (115 of the 300 sampled rows where the rule and LLM disagreed, prioritized with the 44 `review`-involving cases first). This is the current blocking step — it determines whether the rule-based or LLM labels are closer to correct, or whether `review` needs redefining. *(Infrastructure and prioritized review sheet ready; adjudication itself not yet done.)*
2. Based on (1): decide whether to extend LLM labeling to the full ~8,400-row corpus and re-run Section 7's experiments against the higher-quality label set; complete the 42 rows not labeled in the current sample due to a daily API quota limit.
3. Add statistical rigor: repeated repo-held-out splits (`GroupKFold` or multiple `GroupShuffleSplit` seeds), report confidence intervals rather than single point estimates.
4. Ablation: isolate how much of Finding 2 (history helps) is attributable to `prev_action_type` alone versus the other history features, to rule out label-autocorrelation as an alternative explanation.
5. Read and cite the two directly-related papers identified in the Section 8 addendum; write an explicit differentiation paragraph for Section 3.
6. Extend the context-window experiment from summary history *features* to actual concatenated prior-turn *text* (N=1,2,3 turns), sourced from complete uncapped sessions, as a richer alternative to the current feature-based history representation.
7. Optional stretch goals: (a) try the code-aware embedding model already available in the pipeline (`nomic-ai/CodeRankEmbed`) but not yet evaluated against `action_type`; (b) fine-tune a small transformer (DistilBERT/CodeBERT) as a stronger-but-costlier upper tier; (c) report LLM zero-shot classification performance as an upper-bound reference point, framed as "a trained model recovers X% of LLM-level performance at a fraction of the inference cost" — directly supporting the model-routing motivation in Section 1.
8. If broader repository coverage is needed, GitHub's code-search API caps each query at 1,000 results; the discovery queries in Section 4.1 already exceed that cap for two of three code-search signals, so query-splitting (e.g., by language) would recover additional repositories not yet reachable.

---

## 10. Success Criteria (restated from the originating spec)

A successful outcome for this project demonstrates:

- [x] Developer prompts contain *some* predictive information about future agent behavior (prompt-only beats majority-class baseline: 0.237 vs. 0.095 macro-F1) — though the margin over a naive keyword rule is thin, which is itself a reportable finding.
- [x] Adding conversational history improves prediction relative to prompt-only models (0.237 → 0.412 macro-F1) — the project's strongest result.
- [x] Learned representations generalize to unseen repositories (evaluation is strictly repository-held-out throughout).
- [ ] Human-validated label quality is reported alongside model performance (in progress — see Section 9, items 1–2).
- [x] The model provides interpretable insight into the structure of human–AI programming interactions (Section 7.2's precision/recall asymmetry across classes and models).

---

## 11. Broader Impact

This project studies the *intermediate process* of AI-assisted software development — developer intent → AI interpretation → agent behavior — rather than only the end product (code correctness). Understanding whether that intermediate process is predictable from language and short-range context has direct implications for building coding assistants that are more resource-efficient (routing simple requests to cheaper models), more responsive (pre-fetching context before an agent needs it), and more genuinely adaptive to how individual developers actually work, rather than treating every request as an undifferentiated instance of "generate code."
