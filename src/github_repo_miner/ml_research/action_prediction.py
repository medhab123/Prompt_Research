"""Predict `action_type` — the actual research target — from prompt text and context.

The rest of this package (representations.py, models.py, splits.py) was built for
an older `primary_intent` regex label. That label is circular (TF-IDF memorizes
the same keyword patterns used to generate it) and isn't the label the current
research spec asks about. This module reuses the same representation/model/split
infrastructure but targets `action_type` directly, and adds the context-window
experiment (prompt-only vs. prompt + prior-turn history) the spec calls for.

Leakage guard: `action_type` was itself derived partly from response-side signals
(has_generated_code, code_block_count, response length — see action_labeling.py).
Those columns, and anything else describing the agent's response, must never be
used as model inputs here — only the current prompt and *prior*-turn signals
(already shift(1)'d at extraction time, so they can't see the current turn).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

from .evaluate import results_to_dataframe, save_confusion_matrix, save_results_table
from .models import train_classifiers
from .report import plot_class_distribution, plot_model_comparison
from .representations import build_tfidf, hstack_features, load_or_compute_embeddings, metadata_matrix
from .splits import repo_group_kfold, repo_train_test_split, split_overlap_report

ACTION_TYPES = ["implement", "debug", "explain", "review", "tool_only", "other"]

# Prompt-derived only — safe. Never add has_generated_code/has_tool_use/
# code_block_count/agent_response*/generated_code* here: those describe the
# agent's response and are part of how action_type was labeled in the first
# place (see action_labeling.infer_agent_action).
PROMPT_METADATA_COLS = [
    "char_len",
    "word_len",
    "line_count",
    "has_question",
    "has_code_mention",
    "starts_imperative",
    "prompt_length",
    "prompt_word_count",
]

# Prior-turn signals only, already shift(1)'d during extraction
# (session_features.add_session_features) so they cannot see the current turn.
HISTORY_NUMERIC_COLS = [
    "turns_before",
    "prev_turn_had_code",
    "prev_turn_had_tools",
    "session_code_rate_before",
]
PREV_ACTION_COL = "prev_action_type"


@dataclass
class ActionPredictionConfig:
    input_csv: Path
    output_dir: Path = Path("outputs/ml_action")
    embedding_model: str = "all-MiniLM-L6-v2"
    test_size: float = 0.2
    random_state: int = 42
    batch_size: int = 64
    device: str | None = None
    tfidf_max_features: int = 2000
    # Which column to treat as ground truth. Defaults to the LLM-assisted label
    # (human-validated as the better proxy, kappa 0.760 vs 0.614 for the rule)
    # when present, falling back to the rule-based label otherwise so this
    # still runs against older/partial corpora.
    label_col: str | None = None
    # Path to repo_context.py's output (full_name, repo_language, repo_stars, ...).
    # The spec's Prediction Task section explicitly lists "repository context"
    # as a promised input; left as None this representation is simply skipped
    # rather than failing, so this script still runs on a corpus without it.
    repo_context_csv: Path | None = None
    # Repository-held-out CV folds for statistical rigor (Section 8: "no
    # statistical rigor yet" was a named limitation). A single GroupShuffleSplit
    # is still used for the headline confusion-matrix/report, but every
    # representation x model combination is also scored across folds so macro-F1
    # is reported as mean ± std rather than one point estimate.
    n_cv_folds: int = 5
    # Max number of prior turns to sweep for the context-window experiment
    # (the spec's "key experiment": performance vs. how much history is given).
    max_context_turns: int = 3


def resolve_label_col(df: pd.DataFrame, requested: str | None) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"Requested label_col={requested!r} not in dataset columns.")
        return requested
    if "llm_action_type" in df.columns and df["llm_action_type"].notna().any():
        return "llm_action_type"
    return "action_type"


def load_action_dataset(path: str | Path, *, label_col: str | None = None) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(path, low_memory=False)
    required = {"prompt_text", "full_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")
    resolved_label_col = resolve_label_col(df, label_col)
    print(f"Using label column: {resolved_label_col!r}")

    before = len(df)
    df = df[df[resolved_label_col].notna() & (df[resolved_label_col] != "")].reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows with no {resolved_label_col} label.")
    return df, resolved_label_col


def load_repo_context(df: pd.DataFrame, repo_context_csv: Path | None) -> pd.DataFrame:
    """Left-join repo-level metadata onto df by full_name. No-op if unavailable."""
    if repo_context_csv is None or not Path(repo_context_csv).exists():
        return df
    repo_df = pd.read_csv(repo_context_csv)
    repo_df = repo_df.drop_duplicates(subset="full_name", keep="last")
    missing = set(df["full_name"]) - set(repo_df["full_name"])
    if missing:
        print(f"  repo_context: {len(missing)} repos in corpus have no fetched metadata (left as unknown)")
    return df.merge(repo_df, on="full_name", how="left")


def prompt_keyword_rule_baseline(texts: list[str]) -> list[str]:
    """Prompt-only keyword-rule baseline (no response signals available).

    This is deliberately weaker than action_labeling.infer_agent_action, which
    also looks at the response — that's the point: it tells you what prompt
    text alone gets you with zero learning, the bar a real model has to clear.
    """
    from ..action_labeling import DEBUG_PROMPT, EXPLAIN_PROMPT, IMPLEMENT_PROMPT, REVIEW_PROMPT

    preds = []
    for raw in texts:
        text = str(raw)
        if DEBUG_PROMPT.search(text):
            preds.append("debug")
        elif IMPLEMENT_PROMPT.search(text):
            preds.append("implement")
        elif REVIEW_PROMPT.search(text):
            preds.append("review")
        elif EXPLAIN_PROMPT.search(text) or "?" in text:
            preds.append("explain")
        else:
            preds.append("other")
    return preds


def majority_class_baseline(y_train: np.ndarray, y_test: np.ndarray) -> dict:
    majority = pd.Series(y_train).value_counts().idxmax()
    y_pred = np.full(len(y_test), majority)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
    }


def history_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Prior-turn-only features: what happened before this turn, nothing from it."""
    numeric_cols = [c for c in HISTORY_NUMERIC_COLS if c in df.columns]
    numeric = metadata_matrix(df, numeric_cols) if numeric_cols else np.zeros((len(df), 0), dtype=np.float32)

    prev_action = df[PREV_ACTION_COL] if PREV_ACTION_COL in df.columns else pd.Series([""] * len(df))
    prev_action = prev_action.fillna("").astype(str)
    dummies = pd.get_dummies(prev_action, prefix="prev_action")
    for action in ACTION_TYPES:
        col = f"prev_action_{action}"
        if col not in dummies.columns:
            dummies[col] = 0
    dummies = dummies[[f"prev_action_{a}" for a in ACTION_TYPES] + [c for c in dummies.columns if c.replace("prev_action_", "") not in ACTION_TYPES]]

    return np.hstack([numeric, dummies.to_numpy(dtype=np.float32)])


def prev_action_only_matrix(df: pd.DataFrame) -> np.ndarray:
    """Ablation matrix (Section 9 item 4): prev_action_type alone, none of
    history_feature_matrix's other prior-turn signals — isolates how much of
    the "history helps" effect is just this one feature vs. the full bundle.
    """
    prev_action = df[PREV_ACTION_COL] if PREV_ACTION_COL in df.columns else pd.Series([""] * len(df))
    prev_action = prev_action.fillna("").astype(str)
    dummies = pd.get_dummies(prev_action, prefix="prev_action")
    for action in ACTION_TYPES:
        col = f"prev_action_{action}"
        if col not in dummies.columns:
            dummies[col] = 0
    ordered = [f"prev_action_{a}" for a in ACTION_TYPES] + [
        c for c in dummies.columns if c.replace("prev_action_", "") not in ACTION_TYPES
    ]
    return dummies[ordered].to_numpy(dtype=np.float32)


REPO_NUMERIC_COLS = ["repo_stars", "repo_forks", "repo_size_kb", "repo_open_issues", "repo_age_months"]


def top_languages(df: pd.DataFrame, k: int = 12) -> list[str]:
    if "repo_language" not in df.columns:
        return []
    return df["repo_language"].fillna("unknown").value_counts().head(k).index.tolist()


def repo_context_matrix(df: pd.DataFrame, top_languages: list[str]) -> np.ndarray:
    """Repo-level features: the "repository context" input the spec promises
    but which was entirely absent from the corpus until repo_context.py added
    it. Numeric stats + a bounded one-hot over the most common languages
    (everything else bucketed as "other") to avoid a huge sparse language
    column for a 251-repo corpus with a long tail of one-off languages.
    """
    numeric_cols = [c for c in REPO_NUMERIC_COLS if c in df.columns]
    numeric = metadata_matrix(df, numeric_cols) if numeric_cols else np.zeros((len(df), 0), dtype=np.float32)

    lang = df["repo_language"].fillna("unknown") if "repo_language" in df.columns else pd.Series(["unknown"] * len(df))
    lang_bucketed = lang.where(lang.isin(top_languages), other="other")
    lang_dummies = pd.get_dummies(lang_bucketed, prefix="repo_lang")
    for l in top_languages + ["other"]:
        col = f"repo_lang_{l}"
        if col not in lang_dummies.columns:
            lang_dummies[col] = 0
    ordered_cols = [f"repo_lang_{l}" for l in top_languages + ["other"] if f"repo_lang_{l}" in lang_dummies.columns]
    lang_dummies = lang_dummies[ordered_cols]

    return np.hstack([numeric, lang_dummies.to_numpy(dtype=np.float32)])


def build_prior_turns_text(df: pd.DataFrame, n_turns: int) -> list[str]:
    """Concatenate each row's current prompt with the *actual text* of its
    previous n_turns within the same session (not just summary stats) — the
    spec's "Sequential Conversation Representations" section asks for exactly
    this: current prompt + previous N turns, swept across N, to trace out a
    performance-vs-context-length curve.

    Does not assume df's row order matches turn order — sorts a working copy
    by (session_id, turn_index) internally and maps results back to df's
    original index, since upstream sampling/filtering (e.g. the stratified
    per-repo cap) does not preserve session-turn ordering in the CSV.
    """
    if n_turns <= 0:
        return df["prompt_text"].fillna("").astype(str).tolist()

    ordered = df[["session_id", "turn_index", "prompt_text"]].copy()
    ordered["prompt_text"] = ordered["prompt_text"].fillna("").astype(str)
    ordered = ordered.sort_values(["session_id", "turn_index"])

    result_by_index: dict = {}
    for session_id, group in ordered.groupby("session_id", sort=False):
        session_prompts = group["prompt_text"].tolist()
        session_idx = group.index.tolist()
        for pos, idx in enumerate(session_idx):
            history = session_prompts[max(0, pos - n_turns):pos]
            blocks = [f"[turn -{len(history) - i}] {_truncate_for_context(t)}" for i, t in enumerate(history)]
            blocks.append(f"[current] {_truncate_for_context(session_prompts[pos])}")
            result_by_index[idx] = "\n".join(blocks)

    return [result_by_index[idx] for idx in df.index]


def _truncate_for_context(text) -> str:
    text = "" if pd.isna(text) else str(text)
    return text[:300]


def cross_validate_representations(
    df: pd.DataFrame,
    representations: dict[str, np.ndarray],
    y_all: np.ndarray,
    *,
    n_folds: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repo-held-out GroupKFold across each representation, using
    `train_classifiers` per fold so every model family (logistic regression,
    random forest, MLP, XGBoost) gets identical treatment to the main
    single-split run — not logistic-regression-only as in the first version
    of this function.

    The same `folds` list (from a single `repo_group_kfold` call) is reused
    for every representation, so per-fold results are properly paired across
    representations — required for a paired significance test, which the
    previous version (mean/std only, no persisted per-fold values) could not
    support at all.

    Returns (summary_df, per_fold_df). `per_fold_df` is the artifact that
    actually enables a paired test: one row per (representation, model,
    fold), so e.g. `sbert_prompt_only` fold 3 and `sbert+history` fold 3 used
    the exact same train/test repository split.
    """
    folds = repo_group_kfold(df, n_splits=n_folds)
    per_fold_rows = []
    for rep_name, x_all in representations.items():
        for fold_idx, (fold_train_idx, fold_test_idx) in enumerate(folds):
            x_tr, x_te = x_all[fold_train_idx], x_all[fold_test_idx]
            y_tr, y_te = y_all[fold_train_idx], y_all[fold_test_idx]
            results = train_classifiers(x_tr, y_tr, x_te, y_te, random_state=random_state)
            for result in results:
                per_fold_rows.append({
                    "representation": rep_name,
                    "model": result.name,
                    "fold": fold_idx,
                    "macro_f1": result.metrics["macro_f1"],
                    "accuracy": result.metrics["accuracy"],
                })
    per_fold_df = pd.DataFrame(per_fold_rows)

    summary_df = (
        per_fold_df.groupby(["representation", "model"])
        .agg(
            n_folds=("fold", "count"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
        )
        .reset_index()
        .sort_values("macro_f1_mean", ascending=False)
    )
    # ddof=0 (population std over the n_folds folds actually run), matching
    # the previous version's np.std default rather than pandas' ddof=1 default.
    summary_df["macro_f1_std"] = summary_df["macro_f1_std"].fillna(0.0) * np.sqrt(
        (summary_df["n_folds"] - 1).clip(lower=1) / summary_df["n_folds"]
    )
    return summary_df, per_fold_df


def paired_significance_tests(
    per_fold_df: pd.DataFrame,
    comparisons: list[tuple[str, str]],
    *,
    model: str = "logistic_regression",
) -> pd.DataFrame:
    """Paired t-test and Wilcoxon signed-rank test on per-fold macro-F1 for
    each (rep_a, rep_b) pair, restricted to one model family so the folds are
    truly paired (same split, same model, only the representation differs).
    With n=5 folds, power is low — report both tests plainly rather than
    imply significance the sample size can't support.
    """
    from scipy import stats

    rows = []
    for rep_a, rep_b in comparisons:
        a = per_fold_df[(per_fold_df["representation"] == rep_a) & (per_fold_df["model"] == model)].sort_values("fold")
        b = per_fold_df[(per_fold_df["representation"] == rep_b) & (per_fold_df["model"] == model)].sort_values("fold")
        if len(a) != len(b) or len(a) == 0:
            rows.append({"rep_a": rep_a, "rep_b": rep_b, "model": model, "error": f"unmatched fold counts ({len(a)} vs {len(b)})"})
            continue
        vals_a = a["macro_f1"].to_numpy()
        vals_b = b["macro_f1"].to_numpy()
        diffs = vals_b - vals_a
        t_stat, t_p = stats.ttest_rel(vals_b, vals_a)
        try:
            w_stat, w_p = stats.wilcoxon(vals_b, vals_a)
        except ValueError as exc:
            w_stat, w_p = float("nan"), float("nan")
            print(f"  wilcoxon failed for {rep_a} vs {rep_b}: {exc}")
        rows.append({
            "rep_a": rep_a,
            "rep_b": rep_b,
            "model": model,
            "n_folds": len(a),
            "mean_diff_b_minus_a": float(np.mean(diffs)),
            "fold_diffs": [round(float(d), 4) for d in diffs],
            "paired_t_stat": float(t_stat),
            "paired_t_pvalue": float(t_p),
            "wilcoxon_stat": float(w_stat),
            "wilcoxon_pvalue": float(w_p),
        })
    return pd.DataFrame(rows)


def run_action_prediction(config: ActionPredictionConfig) -> Path:
    """Core task: predict action_type from the current prompt (+ optional history)."""
    output_dir = Path(config.output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    df, label_col = load_action_dataset(config.input_csv, label_col=config.label_col)
    df = load_repo_context(df, config.repo_context_csv)
    has_repo_context = "repo_language" in df.columns
    print(f"Rows: {len(df)}  |  repos: {df['full_name'].nunique()}  |  repo context: {has_repo_context}")
    print(df[label_col].value_counts().to_string())

    plot_class_distribution(
        df, label_col, figures_dir / "action_type_distribution.png", title=f"{label_col} distribution"
    )

    print("\nComputing SBERT embeddings (prompt-only)...")
    embeddings = load_or_compute_embeddings(
        df["prompt_text"].astype(str).tolist(),
        output_dir / "prompt_embeddings.npy",
        config.embedding_model,
        batch_size=config.batch_size,
        device=config.device,
    )

    train_idx, test_idx = repo_train_test_split(
        df, test_size=config.test_size, random_state=config.random_state
    )
    split_report = split_overlap_report(df, train_idx, test_idx)
    save_results_table(pd.DataFrame([split_report]), tables_dir / "train_test_split.csv")
    print("Split report:", split_report)

    le = LabelEncoder()
    le.fit(ACTION_TYPES)
    y_all = le.transform(df[label_col])
    y_train = le.transform(df.iloc[train_idx][label_col])
    y_test = le.transform(df.iloc[test_idx][label_col])

    # --- baselines: the bar any learned model has to clear ---
    baseline_rows = []
    maj = majority_class_baseline(y_train, y_test)
    baseline_rows.append({"task": label_col, "representation": "none", "model": "majority_class", **maj})

    rule_preds = prompt_keyword_rule_baseline(df.iloc[test_idx]["prompt_text"].tolist())
    rule_preds_enc = le.transform([p if p in ACTION_TYPES else "other" for p in rule_preds])
    baseline_rows.append({
        "task": label_col,
        "representation": "prompt_keyword_rule",
        "model": "rule_based",
        "accuracy": float(accuracy_score(y_test, rule_preds_enc)),
        "macro_f1": float(f1_score(y_test, rule_preds_enc, average="macro", zero_division=0)),
    })
    baseline_df = pd.DataFrame(baseline_rows)
    save_results_table(baseline_df, tables_dir / "baselines.csv")
    print("\nBaselines:")
    print(baseline_df.to_string(index=False))

    # --- representations: prompt-only text, with metadata, with history ---
    meta_cols = [c for c in PROMPT_METADATA_COLS if c in df.columns]
    meta_all = metadata_matrix(df, meta_cols)
    meta_train, meta_test = meta_all[train_idx], meta_all[test_idx]

    hist_all = history_feature_matrix(df)
    hist_train, hist_test = hist_all[train_idx], hist_all[test_idx]

    # Ablation (Section 9 item 4): prev_action_type alone vs. the full history
    # bundle (prev_action_type + prev_turn_had_code/tools + session_code_rate),
    # to see how much of the "history helps" effect is just label autocorrelation.
    prev_action_only_all = prev_action_only_matrix(df)
    prev_action_only_train = prev_action_only_all[train_idx]
    prev_action_only_test = prev_action_only_all[test_idx]

    x_tfidf_train, x_tfidf_test, _ = build_tfidf(
        df.iloc[train_idx]["prompt_text"].astype(str).tolist(),
        df.iloc[test_idx]["prompt_text"].astype(str).tolist(),
        max_features=config.tfidf_max_features,
    )

    emb_train, emb_test = embeddings[train_idx], embeddings[test_idx]

    representations: dict[str, tuple] = {
        "tfidf": (x_tfidf_train, x_tfidf_test),
        "sbert_prompt_only": (emb_train, emb_test),
        "sbert+prompt_metadata": (
            hstack_features(emb_train, meta_train),
            hstack_features(emb_test, meta_test),
        ),
        "sbert+prev_action_only": (
            hstack_features(emb_train, prev_action_only_train),
            hstack_features(emb_test, prev_action_only_test),
        ),
        "sbert+history": (
            hstack_features(emb_train, hist_train),
            hstack_features(emb_test, hist_test),
        ),
        "sbert+prompt_metadata+history": (
            hstack_features(emb_train, meta_train, hist_train),
            hstack_features(emb_test, meta_test, hist_test),
        ),
    }

    # Repository context (spec's promised "repository context when available" input).
    if has_repo_context:
        langs = top_languages(df)
        repo_all = repo_context_matrix(df, langs)
        repo_train, repo_test = repo_all[train_idx], repo_all[test_idx]
        representations["sbert+prompt_metadata+history+repo_context"] = (
            hstack_features(emb_train, meta_train, hist_train, repo_train),
            hstack_features(emb_test, meta_test, hist_test, repo_test),
        )

    # Context-window sweep (the spec's literal "key experiment": performance vs.
    # how much conversational history is given, using actual prior-turn *text*
    # rather than summary features).
    for n in range(1, config.max_context_turns + 1):
        print(f"\nComputing SBERT embeddings (context_n{n}, prior-{n}-turn text)...")
        texts_n = build_prior_turns_text(df, n)
        emb_n = load_or_compute_embeddings(
            texts_n,
            output_dir / f"context_n{n}_embeddings.npy",
            config.embedding_model,
            batch_size=config.batch_size,
            device=config.device,
        )
        representations[f"context_text_n{n}"] = (emb_n[train_idx], emb_n[test_idx])

    print("\n=== Training classifiers per representation ===")
    result_rows = [baseline_df]
    best: tuple[str, object] | None = None
    for rep_name, (x_tr, x_te) in representations.items():
        print(f"  {rep_name} ...")
        results = train_classifiers(x_tr, y_train, x_te, y_test, random_state=config.random_state)
        result_rows.append(results_to_dataframe(label_col, rep_name, results))
        top = max(results, key=lambda r: r.metrics["macro_f1"])
        if best is None or top.metrics["macro_f1"] > best[1].metrics["macro_f1"]:
            best = (rep_name, top)

    results_df = pd.concat(result_rows, ignore_index=True)
    save_results_table(results_df, tables_dir / "action_type_results.csv")

    if best is not None:
        rep_name, result = best
        save_confusion_matrix(
            y_test,
            result.y_pred,
            labels=list(range(len(le.classes_))),
            output_path=figures_dir / "action_type_confusion_matrix.png",
            title=f"{label_col} — {result.name} ({rep_name})",
        )
        print(f"\nBest: {result.name} + {rep_name}  macro_f1={result.metrics['macro_f1']:.4f}")

    plot_model_comparison(
        results_df[results_df["representation"] != "none"],
        "macro_f1",
        figures_dir / "action_type_model_comparison.png",
        title=f"{label_col} — macro-F1 by model and representation",
    )

    # --- statistical rigor: repeated repo-held-out CV (Section 8: "no
    # statistical rigor yet" was a named limitation) ---
    print(f"\n=== {config.n_cv_folds}-fold repo-held-out cross-validation (all model families) ===")
    cv_reps = {
        "sbert_prompt_only": embeddings,
        "sbert+history": hstack_features(embeddings, hist_all),
        "sbert+prompt_metadata+history": hstack_features(embeddings, meta_all, hist_all),
        "sbert+prev_action_only": hstack_features(embeddings, prev_action_only_all),
    }
    if has_repo_context:
        cv_reps["sbert+prompt_metadata+history+repo_context"] = hstack_features(
            embeddings, meta_all, hist_all, repo_context_matrix(df, top_languages(df))
        )
    cv_df, cv_per_fold_df = cross_validate_representations(
        df, cv_reps, y_all, n_folds=config.n_cv_folds, random_state=config.random_state
    )
    save_results_table(cv_df, tables_dir / "cv_macro_f1.csv")
    save_results_table(cv_per_fold_df, tables_dir / "cv_macro_f1_per_fold.csv")
    print(cv_df.to_string(index=False))

    # Paired significance tests (t-test + Wilcoxon; n=5 folds so power is low
    # either way — report both plainly rather than imply significance the
    # sample size can't support). Restricted to logistic regression: the
    # model whose headline numbers this paper actually reports, and the only
    # way the fold pairing is a true single-factor comparison (representation
    # only, not representation-and-model at once).
    sig_df = paired_significance_tests(
        cv_per_fold_df,
        comparisons=[
            ("sbert_prompt_only", "sbert+prompt_metadata+history"),
            ("sbert+prompt_metadata+history", "sbert+prev_action_only"),
        ],
        model="logistic_regression",
    )
    save_results_table(sig_df, tables_dir / "cv_paired_significance.csv")
    print("\nPaired significance tests (logistic regression, 5 folds):")
    print(sig_df.to_string(index=False))

    context_sweep_rows = results_df[
        results_df["representation"].str.startswith("context_text_n") | (results_df["representation"] == "sbert_prompt_only")
    ].sort_values("representation")

    report_lines = [
        "# Action Type Prediction — Results",
        "",
        f"Label column: `{label_col}`",
        f"Rows: {len(df):,}  |  Repos: {df['full_name'].nunique()}  |  Repo context available: {has_repo_context}",
        f"Train repos: {split_report['train_repos']}  |  Test repos: {split_report['test_repos']}  "
        f"|  Overlapping repos: {split_report['overlapping_repos']} (must be 0)",
        "",
        "## Class distribution",
        df[label_col].value_counts().to_frame("count").to_markdown(),
        "",
        "## Baselines",
        baseline_df.to_markdown(index=False),
        "",
        "## Prompt-only vs. history (does conversation context help?)",
        results_df[
            results_df["representation"].isin(["sbert_prompt_only", "sbert+history", "sbert+prompt_metadata+history"])
        ].sort_values("macro_f1", ascending=False).to_markdown(index=False),
        "",
        "## Ablation: prev_action_type alone vs. full history bundle",
        results_df[
            results_df["representation"].isin(["sbert+prev_action_only", "sbert+history"])
        ].sort_values("macro_f1", ascending=False).to_markdown(index=False),
        "",
        "## Context-window sweep (N previous turns of actual text, single split)",
        context_sweep_rows.to_markdown(index=False),
        "",
        f"## {config.n_cv_folds}-fold repo-held-out CV, all model families (macro-F1 mean ± std)",
        cv_df.to_markdown(index=False),
        "",
        f"## Paired significance tests on CV folds (logistic regression, n={config.n_cv_folds} folds — power is low; report plainly, do not overclaim)",
        sig_df.drop(columns=["fold_diffs"], errors="ignore").to_markdown(index=False),
        "",
        "## All representations x models (single split)",
        results_df.sort_values("macro_f1", ascending=False).to_markdown(index=False),
    ]
    report_path = output_dir / "ACTION_PREDICTION_REPORT.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nReport: {report_path}")
    return report_path
