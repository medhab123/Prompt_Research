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

**Final corpus:** after noise filtering, exact-duplicate removal, and the per-repository cap: **8,382 developer–agent turns across 252 repositories**, organized into **2,126 distinct multi-turn sessions** (median 13 turns per session).

### 4.2 Class distribution

| Action | Count | % of corpus |
|---|---:|---:|
| `implement` | 3,896 | 46.5% |
| `explain` | 2,132 | 25.4% |
| `other` | 718 | 8.6% |
| `tool_only` | 679 | 8.1% |
| `review` | 629 | 7.5% |
| `debug` | 328 | 3.9% |

The imbalance is not an artifact of sampling — it reflects the genuine skew of real coding-agent sessions, where "write/modify code" requests dominate and narrowly-scoped debugging exchanges are comparatively rare. This is precisely why **macro-F1** (unweighted mean of per-class F1) rather than raw accuracy is the primary evaluation metric throughout: a classifier that always predicts `implement` would already be "right" ~46% of the time without learning anything.

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

*(These are real, already-obtained results from the current pipeline — not projected or illustrative numbers.)*

**Split:** 201 training repositories / 51 held-out test repositories, 1,607 test rows, confirmed zero repository overlap.

### 7.1 Headline comparison

| Representation | Model | Accuracy | Macro-F1 |
|---|---|---:|---:|
| — (majority class) | — | 39.6% | 0.095 |
| Prompt-only keyword rule | rule-based | 31.8% | 0.257 |
| `sbert_prompt_only` | MLP | 40.4% | 0.237 |
| `sbert+prompt_metadata+history` | MLP | 52.3% | **0.412** |
| `sbert+prompt_metadata+history` | XGBoost | **58.1%** | 0.408 |

**Finding 1 (RQ1, partial support).** A trained model on the prompt alone (macro-F1 0.237) does *not* clearly beat a hand-written keyword rule (0.257) applied to the same prompt-only information. Semantic embeddings are not, by themselves, extracting meaningfully more signal than lexical pattern matching for this task. This is a genuine, non-obvious result worth reporting honestly rather than a null result to bury.

**Finding 2 (RQ2, strong support).** Adding prior-turn context nearly doubles macro-F1 (0.237 → 0.412). The size of this jump — not merely its direction — is the project's central empirical finding: short-range conversational context carries substantially more predictive signal than prompt semantics in isolation.

**Finding 3 (RQ3, supported by design).** Because the improvement in Finding 2 is measured on repositories the model has never seen, the gain is attributable to learned structure in developer–agent interaction patterns (e.g., "a debug turn is often followed by another debug turn," "a long implement streak tends to continue"), not to repository-specific memorization.

### 7.2 Per-class detail (best two models, `sbert+prompt_metadata+history`)

| Class | MLP Precision / Recall / F1 | XGBoost Precision / Recall / F1 | Support |
|---|---|---|---:|
| `debug` | 0.38 / 0.28 / 0.32 | 0.78 / **0.09** / 0.15 | 82 |
| `explain` | 0.48 / 0.47 / 0.48 | 0.54 / 0.53 / 0.53 | 460 |
| `implement` | 0.59 / 0.72 / 0.65 | 0.59 / **0.85** / 0.70 | 637 |
| `other` | 0.21 / 0.21 / 0.21 | 0.43 / 0.20 / 0.27 | 91 |
| `review` | 0.34 / 0.24 / 0.28 | 0.48 / **0.13** / 0.20 | 127 |
| `tool_only` | 0.66 / 0.44 / 0.53 | 0.68 / 0.52 / 0.59 | 210 |

**Finding 4 (RQ4, concrete illustration of why macro-F1 matters).** XGBoost's higher raw accuracy is driven almost entirely by aggressively over-predicting the majority class (`implement` recall 0.85 vs. MLP's 0.72), at the cost of nearly abandoning the rare classes (`debug` recall 0.09, `review` recall 0.13). MLP trades a few points of majority-class recall for meaningfully better minority-class coverage. For any of the motivating use cases in Section 1 (e.g., routing debugging requests to a specialized agent), a model that catches 9% of debug turns is close to non-functional for that purpose despite its better headline accuracy — the exact failure mode macro-F1 as primary metric is designed to surface.

**Finding 5.** `explain` and `tool_only` are the most learnable classes for both models — plausibly because they have the clearest surface tells (explicit questions; repository-navigation language) reinforced by session context. `other` is the weakest class for both models (F1 0.21–0.27), consistent with its definition as a residual, low-signal category rather than a coherent behavioral class.

---

## 8. Threats to Validity / Limitations

- **Label quality.** Ground truth currently rests on a deterministic rule, not human judgment. Section 5's Tier 2/3 work (LLM-assisted relabeling + human validation, in progress) is the direct mitigation, and any final write-up should report inter-annotator/inter-method agreement (Cohen's κ) rather than treating rule-based labels as ground truth.
- **Selection bias.** The corpus is limited to repositories where a developer both used SpecStory *and* chose to commit the resulting logs publicly — this is not a random sample of all AI-assisted development, and likely skews toward more experimental, open-source-oriented, or process-transparent developers.
- **Session-boundary effects on the context feature.** The per-repository cap (Section 4.1) samples individual *rows* stratified by action type; it does not guarantee that every turn's immediately-preceding turn survived the same cap. Prior-turn features were computed from the complete, uncapped session at extraction time and are therefore still individually valid, but a richer context representation (raw concatenated prior-turn *text*, rather than summary features) would require re-deriving from complete, uncapped sessions — noted as a scoping decision, not an oversight (Section 9).
- **Model scale vs. dataset scale.** 8,382 rows is workable for classical ML + embeddings but is a genuine constraint on fine-tuning a full transformer without overfitting; this shaped the model-selection decision in Section 6.2.

---

## 9. Remaining Work / Next Steps

1. Complete the LLM-assisted labeling validation sample (target: 300 rows, stratified); compute Cohen's κ against rule-based labels.
2. Human-adjudicate the resulting review sheet; report annotator agreement.
3. Based on (1)–(2), decide whether to extend LLM labeling to the full 8,382-row corpus and re-run Section 7's experiments against the higher-quality label set.
4. Extend the context-window experiment from summary history *features* to actual concatenated prior-turn *text* (N=1,2,3 turns), sourced from complete uncapped sessions, as a richer alternative to the current feature-based history representation.
5. Optional stretch goals: (a) fine-tune a small transformer (DistilBERT/CodeBERT) directly on the classification head as a stronger-but-costlier upper tier; (b) report LLM zero-shot classification performance as an upper-bound reference point, framed as "a trained model recovers X% of LLM-level performance at a fraction of the inference cost" — directly supporting the model-routing motivation in Section 1.
6. If broader repository coverage is needed, GitHub's code-search API caps each query at 1,000 results; the discovery queries in Section 4.1 already exceed that cap for two of three code-search signals, so query-splitting (e.g., by language) would recover additional repositories not yet reachable.

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
