# SpecStory Agent Behavior Prediction: Critical Audit & Research Redesign

This document answers the five-part research redesign request. It is grounded in code inspection, label semantics, and **new robustness experiments** (`run_robustness_experiments.py`).

**Artifacts:** `outputs/ml/robustness/tables/robustness_comparison.csv`, `label_audit_summary.csv`, `ROBUSTNESS_REPORT.md`

---

## Part 1: Brutal Audit of the Current Pipeline

### 1.1 What is genuinely interesting?

| Component | Why it matters |
|-----------|----------------|
| **Real SpecStory turn-pair corpus** | 3,673 developer prompts paired with actual agent responses across 65 repos — rare observational data on production agent use |
| **Repo-held-out evaluation** | Correct evaluation design: no repository appears in both train and test |
| **Multi-task framing** | Code generation, tool use, session length, intent — maps to real agent decision points |
| **Turn-pair parser** (`specstory_parser.py`) | Extracts structured behavior from markdown transcripts |
| **Session/turn metadata** | `turn_index`, prompt length, question marks — weak but legitimate contextual signals |

The **dataset and evaluation protocol** are the real assets. The current headline metrics are not.

### 1.2 What is likely trivial?

| Claim | Reality |
|-------|---------|
| "AUC 0.88 for `has_generated_code`" | **Inflated by leakage.** `repo_code_rate` was computed on the **full corpus** before split. When recomputed on **train only** and evaluated on held-out repos, repo-only AUC collapses to **0.50** (chance). |
| "Implement → generate code" keyword story | **Wrong for this dataset.** Keyword heuristic accuracy = **43.5%** (below majority 64.4%). Questions have *lower* code rate (36%) than imperatives (57%). Short prompts (≤5 words) have *higher* code rate (66%) than long ones (47%). |
| Intent classification (macro-F1 0.83) | **Circular.** `primary_intent` is assigned by regex on the same text being predicted (`prompt_analysis.py` → `_add_intents()`). TF-IDF wins because it memorizes the rule patterns. |
| TF-IDF high F1 (0.75) on code prediction | **Misleading.** High F1 comes from predicting the majority positive class aggressively; AUC is only **0.60** — modest separation. |
| `has_tool_use` prediction | **Near chance** from prompt alone (AUC ≈ 0.50). Class imbalance (10% positive) makes accuracy look fine while F1 ≈ 0. |

### 1.3 What a reviewer would attack

1. **Label semantics:** `has_generated_code` = "response contains a fenced code block" — not "agent chose to implement." See Part 5.
2. **Feature leakage:** Global `repo_code_rate` / `repo_tool_rate` before split (now fixed in robustness module).
3. **Circular intent task:** Predicting regex-derived labels from the same text.
4. **Weak baselines:** Original pipeline lacked majority-class and keyword heuristics.
5. **No ablation:** Could not tell if signal was lexical, semantic, or repository prior.
6. **Single-turn prediction:** Ignores conversation history — turn 5 behaves differently from turn 1.
7. **Selection bias:** Only repos with SpecStory installed; not representative of all GitHub development.

### 1.4 Experiments needed to prove meaningful learning

| Experiment | Status | Finding |
|------------|--------|---------|
| Majority-class baseline | ✅ Done | 64.4% acc, AUC 0.50 — minimum bar |
| Keyword heuristic baseline | ✅ Done | 43.5% acc, AUC 0.47 — **worse than majority** |
| Keyword-stripped prompts (Dataset B) | ✅ Done | AUC **unchanged** (0.636 vs 0.625) — model not keyword-dependent |
| SBERT-only (Dataset C) | ✅ Done | AUC 0.625 — weak semantic signal |
| Train-only repo features | ✅ Done | AUC 0.50 alone — **does not generalize to new repos** |
| Label quality audit | ✅ Done | 13.1% misleading positive labels |
| Session-history features | ❌ Needed | Prior turn actions as input |
| Finer action taxonomy | ❌ Needed | implement vs explain-with-snippet vs tool-call |
| Human-validated label subset | ❌ Needed | Gold standard for 200–500 rows |

**Bottom line:** The current binary code/tool prediction task, evaluated honestly, is **weakly learnable from prompt text alone** (AUC ~0.63). The earlier strong results were largely an artifact of leaky repository aggregates.

---

## Part 2: Redefine the Research Question

### Is the current question too weak?

**Yes.** "Can we predict whether an LLM coding agent generates code or uses tools?" has three problems:

1. **Trivial surface correlation** — partially debunked (keywords don't work), but labels are noisy.
2. **Binary outcome** — collapses rich agent behavior into one bit.
3. **Post-hoc label** — code-block presence ≠ action choice.

### Stronger alternative framings

| Framing | Strength | Feasibility with current data |
|---------|----------|-------------------------------|
| Predict trajectories from developer intent | High — models agent as sequential decision process | Medium — need turn-sequence labels |
| Pre-execution behavior modeling | High — actionable for routing/budgeting | Medium — needs causal framing |
| Repository context influences decisions | High — novel for SE | **Weak generalization** in our tests (repo features don't transfer) |
| Action selection in real-world workflows | **Strongest** — concrete, evaluable | High — relabel existing data |

### Recommended research question (ML/AI audience)

> **Given a developer prompt and session context, can we forecast the agent's next action type before execution — and does that forecast generalize across repositories?**

Refined into testable sub-questions:

1. **Action taxonomy:** Can we predict *action class* (implement, debug-explain, refactor, run-tests, read-files) rather than code-block presence?
2. **Generalization:** Does signal come from prompt semantics or repository-specific priors? (Our answer: priors don't transfer; semantics are weak but real.)
3. **Trajectory:** Does adding conversation history improve forecasting beyond single-turn prompts?
4. **Calibration:** Can pre-execution forecasts support agent routing (e.g., skip expensive tool loops for explanation-only turns)?

This framing is stronger because it targets **decision forecasting under distribution shift** (new repos), which ML reviewers care about.

---

## Part 3: Stronger Baselines (Implemented)

Run: `python run_robustness_experiments.py`

### Baseline 1: Majority Class

Always predict the most common training label.

| Target | Accuracy | F1 | ROC-AUC |
|--------|----------|-----|---------|
| `has_generated_code` | 64.4% | 0.783 | 0.500 |
| `has_tool_use` | 81.0% | 0.000 | 0.500 |

**Why it matters:** Any model claiming "87% accuracy" for code prediction is only ~23 points above majority. For imbalanced `has_tool_use`, high accuracy with F1 = 0 is worthless.

### Baseline 2: Keyword Heuristic

Rules (`baselines.py`):
- Code keywords (implement, create, add, write, function, class, …) → predict code
- No-code keywords (why, explain, error, crash, bug, …) → predict no code
- Abstain otherwise (defaults to 0)

| Target | Accuracy | F1 | ROC-AUC | Coverage |
|--------|----------|-----|---------|----------|
| `has_generated_code` | 43.5% | 0.441 | 0.472 | 43.9% |
| `has_tool_use` | 67.3% | 0.079 | 0.444 | 16.5% |

**Does ML beat keyword matching?** For `has_generated_code`, yes — XGBoost+SBERT (AUC 0.636) beats keyword heuristic (AUC 0.472). But the margin is small, and **neither is a strong classifier**. The naive "implement → code" story is empirically false on this corpus.

---

## Part 4: Robustness Experiments — Shortcut Removal

### Comparison table (`has_generated_code`, held-out repos)

| Dataset | Representation | Model | AUC | F1 |
|---------|----------------|-------|-----|-----|
| A: Full | TF-IDF | XGBoost | 0.600 | 0.755 |
| A: Full | SBERT + safe meta | XGBoost | 0.636 | 0.646 |
| A: Full | SBERT | XGBoost | 0.625 | 0.624 |
| C: Semantic only | SBERT | XGBoost | 0.625 | 0.624 |
| B: Keyword masked | SBERT | XGBoost | 0.636 | 0.651 |
| B: Keyword masked | SBERT + safe meta | XGBoost | 0.636 | 0.663 |
| A: Full | SBERT + repo (train-only) | XGBoost | 0.543 | 0.517 |
| A: Full | Repo train-only | XGBoost | 0.500 | 0.000 |
| — | Majority class | rule | 0.500 | 0.783 |
| — | Keyword heuristic | rule | 0.472 | 0.441 |

### Comparison table (`has_tool_use`)

| Dataset | Representation | Model | AUC | F1 |
|---------|----------------|-------|-----|-----|
| A: Full | SBERT + repo (train-only) | XGBoost | 0.604 | 0.110 |
| A: Full | TF-IDF | XGBoost | 0.584 | 0.154 |
| A: Full | SBERT | XGBoost | 0.501 | 0.016 |
| — | Majority class | rule | 0.500 | 0.000 |
| — | Keyword heuristic | rule | 0.444 | 0.079 |

### Interpretation

1. **Keyword removal does not hurt** → current models are not exploiting obvious lexical shortcuts; signal is subtler (length, structure, session position).
2. **SBERT ≈ TF-IDF in AUC** but TF-IDF inflates F1 by favoring the positive class.
3. **Repo features without leakage do not generalize** → prior "repo context" result was a leakage artifact.
4. **Semantic-only matches full prompt** → most usable signal is in embedding space, not raw keywords.

---

## Part 5: Label Quality Analysis

### What `has_generated_code` actually measures

From `specstory_parser.py`:

```160:161:src/github_repo_miner/specstory_parser.py
                has_generated_code=len(blocks) > 0,
                has_tool_use=has_tool_use,
```

It means: **the assistant response contains at least one fenced code block** (` ``` `). It does **not** mean:
- the agent chose an implementation action
- the code is new (vs. quoting existing code)
- the code is executable or intended for the user to copy

### Audit results (full corpus, n = 3,673)

| Category | Count | % |
|----------|-------|---|
| Consistent | 2,989 | 81.4% |
| Code despite non-implementation prompt | 482 | **13.1%** |
| No code despite implementation prompt | 202 | 5.5% |

**Misleading positive examples** (debug/explain prompt → code in response):

| Prompt pattern | Example | Label | Problem |
|----------------|---------|-------|---------|
| Error stack paste | `Uncaught Error: Maximum update depth exceeded...` | `has_generated_code=True` | User pasted error; agent may quote code to explain |
| Question + context | `PluginSystem.ts:360 Uncaught ReferenceError...` | True | Diagnostic, not implementation request |
| Review/discussion | `could we update function get_queues so that...` | False | Implementation request but agent replied in prose |

**User's example confirmed:**

> User: "Why does this crash?"  
> Assistant: "The problem is here: \`\`\`python x=None \`\`\`"

This would be labeled `has_generated_code=True` even though the agent action is **explanation**, not **implementation**.

### Recommended label improvements

| Current | Proposed |
|---------|----------|
| `has_generated_code` | `response_contains_code_block` (honest name) |
| — | `action_type`: {implement, explain, debug, refactor, test, other} |
| — | `code_is_novel`: heuristic or human label |
| — | `primary_outcome`: human-annotated subset for gold evaluation |

---

## Recommended Next Steps

### Immediate (strengthens paper)

1. **Stop reporting leaky `repo_code_rate` results** — use `add_train_only_repo_stats()` only.
2. **Report baselines** (majority, keyword) in every table.
3. **Rename labels** to match semantics.
4. **Drop or reframe intent Task 1** — it's circular.

### Medium-term (real contribution)

1. **Session-aware models:** feed prior turns' actions as features.
2. **Finer action taxonomy:** rule + LLM-assisted labeling on 500 examples.
3. **Human validation set:** 200–500 gold labels for serious claims.
4. **Trajectory prediction:** predict next 3 actions, not single binary outcome.

### Strongest publishable claim (given current evidence)

> Developer prompts carry a **weak but non-trivial** signal (AUC ~0.63) for whether an agent response will contain code blocks, but this signal is **not reducible to keyword rules**, **does not transfer via repository priors**, and current labels conflate explanation-with-snippet and true implementation. Pre-execution agent action forecasting requires richer action labels and session context.

---

## How to Reproduce

```bash
# Robustness baselines + datasets A/B/C + label audit
python run_robustness_experiments.py

# Original ML pipeline (note: uses leaky repo features unless updated)
python run_ml_research.py
```

**New modules:**
- `src/github_repo_miner/ml_research/baselines.py`
- `src/github_repo_miner/ml_research/label_audit.py`
- `src/github_repo_miner/ml_research/robustness.py`
- `run_robustness_experiments.py`
