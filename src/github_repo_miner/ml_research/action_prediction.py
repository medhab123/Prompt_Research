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
from .splits import repo_train_test_split, split_overlap_report

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


def load_action_dataset(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    required = {"prompt_text", "action_type", "full_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")
    before = len(df)
    df = df[df["action_type"].notna()].reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows with no action_type label.")
    return df


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


def run_action_prediction(config: ActionPredictionConfig) -> Path:
    """Core task: predict action_type from the current prompt (+ optional history)."""
    output_dir = Path(config.output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    df = load_action_dataset(config.input_csv)
    print(f"Rows: {len(df)}  |  repos: {df['full_name'].nunique()}")
    print(df["action_type"].value_counts().to_string())

    plot_class_distribution(
        df, "action_type", figures_dir / "action_type_distribution.png", title="action_type distribution"
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
    y_train = le.transform(df.iloc[train_idx]["action_type"])
    y_test = le.transform(df.iloc[test_idx]["action_type"])

    # --- baselines: the bar any learned model has to clear ---
    baseline_rows = []
    maj = majority_class_baseline(y_train, y_test)
    baseline_rows.append({"task": "action_type", "representation": "none", "model": "majority_class", **maj})

    rule_preds = prompt_keyword_rule_baseline(df.iloc[test_idx]["prompt_text"].tolist())
    rule_preds_enc = le.transform([p if p in ACTION_TYPES else "other" for p in rule_preds])
    baseline_rows.append({
        "task": "action_type",
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
    meta_train = metadata_matrix(df.iloc[train_idx], meta_cols)
    meta_test = metadata_matrix(df.iloc[test_idx], meta_cols)

    hist_train = history_feature_matrix(df.iloc[train_idx])
    hist_test = history_feature_matrix(df.iloc[test_idx])

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
        "sbert+history": (
            hstack_features(emb_train, hist_train),
            hstack_features(emb_test, hist_test),
        ),
        "sbert+prompt_metadata+history": (
            hstack_features(emb_train, meta_train, hist_train),
            hstack_features(emb_test, meta_test, hist_test),
        ),
    }

    print("\n=== Training classifiers per representation ===")
    result_rows = [baseline_df]
    best: tuple[str, object] | None = None
    for rep_name, (x_tr, x_te) in representations.items():
        print(f"  {rep_name} ...")
        results = train_classifiers(x_tr, y_train, x_te, y_test, random_state=config.random_state)
        result_rows.append(results_to_dataframe("action_type", rep_name, results))
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
            title=f"action_type — {result.name} ({rep_name})",
        )
        print(f"\nBest: {result.name} + {rep_name}  macro_f1={result.metrics['macro_f1']:.4f}")

    plot_model_comparison(
        results_df[results_df["representation"] != "none"],
        "macro_f1",
        figures_dir / "action_type_model_comparison.png",
        title="action_type — macro-F1 by model and representation",
    )

    report_lines = [
        "# Action Type Prediction — Results",
        "",
        f"Rows: {len(df):,}  |  Repos: {df['full_name'].nunique()}",
        f"Train repos: {split_report['train_repos']}  |  Test repos: {split_report['test_repos']}  "
        f"|  Overlapping repos: {split_report['overlapping_repos']} (must be 0)",
        "",
        "## Class distribution",
        df["action_type"].value_counts().to_frame("count").to_markdown(),
        "",
        "## Baselines",
        baseline_df.to_markdown(index=False),
        "",
        "## Prompt-only vs. history (does conversation context help?)",
        results_df[
            results_df["representation"].isin(["sbert_prompt_only", "sbert+history", "sbert+prompt_metadata+history"])
        ].sort_values("macro_f1", ascending=False).to_markdown(index=False),
        "",
        "## All representations x models",
        results_df.sort_values("macro_f1", ascending=False).to_markdown(index=False),
    ]
    report_path = output_dir / "ACTION_PREDICTION_REPORT.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nReport: {report_path}")
    return report_path
