"""End-to-end ML research pipeline orchestration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

from .config import MLResearchConfig
from .data import (
    INTENT_BINARY_COLS,
    METADATA_FEATURE_COLS,
    REPO_FEATURE_COLS,
    analyze_dataset,
    load_ml_dataset,
    save_dataset_analysis,
)
from .evaluate import results_to_dataframe, save_confusion_matrix, save_results_table
from .models import train_binary_classifiers, train_classifiers, train_regressors
from .report import (
    plot_class_distribution,
    plot_embedding_umap,
    plot_feature_importance,
    plot_model_comparison,
    write_research_report,
)
from .representations import (
    build_tfidf,
    encode_code_aware,
    hstack_features,
    load_or_compute_embeddings,
    metadata_matrix,
)
from .splits import repo_train_test_split, split_overlap_report


def _repo_split_indices(
    df: pd.DataFrame,
    *,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    return repo_train_test_split(df, test_size=test_size, random_state=random_state)


def _intent_eligible_indices(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    min_class_size: int,
) -> np.ndarray:
    """Keep rows whose intent appears at least min_class_size times in training."""
    train_intents = df.iloc[train_idx]["primary_intent"]
    counts = train_intents.value_counts()
    keep = counts[counts >= min_class_size].index
    return np.where(df["primary_intent"].isin(keep))[0]


def _representation_sets(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    embeddings: np.ndarray,
    code_embeddings: np.ndarray | None,
    config: MLResearchConfig,
    *,
    include_intent_features: bool = False,
) -> dict[str, tuple]:
    meta_cols = [c for c in METADATA_FEATURE_COLS if c in df.columns]
    if include_intent_features:
        meta_cols += [c for c in INTENT_BINARY_COLS + ["intent_count"] if c in df.columns]
    repo_cols = [c for c in REPO_FEATURE_COLS if c in df.columns]
    meta_train = metadata_matrix(df.iloc[train_idx], meta_cols)
    meta_test = metadata_matrix(df.iloc[test_idx], meta_cols)
    repo_train = metadata_matrix(df.iloc[train_idx], repo_cols)
    repo_test = metadata_matrix(df.iloc[test_idx], repo_cols)

    train_texts = df.iloc[train_idx]["prompt_text"].tolist()
    test_texts = df.iloc[test_idx]["prompt_text"].tolist()
    x_tfidf_train, x_tfidf_test, _ = build_tfidf(
        train_texts,
        test_texts,
        max_features=config.tfidf_max_features,
    )

    emb_train = embeddings[train_idx]
    emb_test = embeddings[test_idx]

    reps: dict[str, tuple] = {
        "sbert": (emb_train, emb_test),
        "sbert+metadata": (
            hstack_features(emb_train, meta_train),
            hstack_features(emb_test, meta_test),
        ),
        "tfidf": (x_tfidf_train, x_tfidf_test),
        "tfidf+metadata": (
            hstack_features(x_tfidf_train, meta_train),
            hstack_features(x_tfidf_test, meta_test),
        ),
        "metadata_only": (meta_train, meta_test),
    }

    if code_embeddings is not None:
        code_train = code_embeddings[train_idx]
        code_test = code_embeddings[test_idx]
        reps["code_embed"] = (code_train, code_test)
        reps["code_embed+metadata"] = (
            hstack_features(code_train, meta_train),
            hstack_features(code_test, meta_test),
        )

    behavior_meta_train = hstack_features(meta_train, repo_train)
    behavior_meta_test = hstack_features(meta_test, repo_test)
    reps["sbert+behavior_meta"] = (
        hstack_features(emb_train, behavior_meta_train),
        hstack_features(emb_test, behavior_meta_test),
    )
    return reps


def run_ml_research(config: MLResearchConfig) -> Path:
    """Run Phases 1–4 and write artifacts under config.output_dir."""
    output_dir = Path(config.output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=== Phase 1: Dataset Analysis ===")
    df = load_ml_dataset(config.input_csv)
    analysis = analyze_dataset(df)
    save_dataset_analysis(analysis, tables_dir)

    plot_class_distribution(
        df,
        "primary_intent",
        figures_dir / "class_distribution_intent.png",
        title="Primary Intent Distribution",
    )
    plot_class_distribution(
        df,
        "prompt_type",
        figures_dir / "class_distribution_prompt_type.png",
        title="Prompt Type Distribution",
    )

    embeddings = load_or_compute_embeddings(
        df["prompt_text"].tolist(),
        output_dir / "prompt_embeddings.npy",
        config.embedding_model,
        batch_size=config.batch_size,
        device=config.device,
    )

    code_embeddings = None
    if not config.skip_code_embeddings:
        print("Computing code-aware embeddings...")
        code_embeddings = encode_code_aware(
            df["prompt_text"].tolist(),
            config.code_embedding_model,
            batch_size=config.batch_size,
            device=config.device,
        )

    plot_embedding_umap(
        embeddings,
        df["primary_intent"],
        figures_dir / "embedding_umap_by_intent.png",
        title="SBERT Embeddings — colored by primary_intent",
    )

    train_idx, test_idx = _repo_split_indices(
        df,
        test_size=config.test_size,
        random_state=config.random_state,
    )
    split_report = split_overlap_report(df, train_idx, test_idx)
    pd.DataFrame([split_report]).to_csv(tables_dir / "train_test_split.csv", index=False, encoding="utf-8-sig")

    representations = _representation_sets(
        df, train_idx, test_idx, embeddings, code_embeddings, config
    )

    print("\n=== Phase 2: Task 1 — Intent Classification ===")
    eligible = _intent_eligible_indices(df, train_idx, config.min_intent_class_size)
    train_intent = np.intersect1d(train_idx, eligible)
    test_intent = np.intersect1d(test_idx, eligible)
    print(f"Intent task rows: train={len(train_intent)}, test={len(test_intent)}")

    le = LabelEncoder()
    y_train_intent = le.fit_transform(df.iloc[train_intent]["primary_intent"])
    y_test_intent = le.transform(df.iloc[test_intent]["primary_intent"])

    intent_reps = _representation_sets(
        df, train_intent, test_intent, embeddings, code_embeddings, config,
        include_intent_features=False,
    )

    task1_rows = []
    best_task1: tuple[str, object] | None = None
    for rep_name, (x_tr, x_te) in intent_reps.items():
        results = train_classifiers(
            x_tr, y_train_intent, x_te, y_test_intent, random_state=config.random_state
        )
        task1_rows.append(results_to_dataframe("intent_classification", rep_name, results))
        top = max(results, key=lambda r: r.metrics["macro_f1"])
        if best_task1 is None or top.metrics["macro_f1"] > best_task1[1].metrics["macro_f1"]:
            best_task1 = (rep_name, top)

    task1_df = pd.concat(task1_rows, ignore_index=True)
    save_results_table(task1_df, tables_dir / "task1_intent_classification.csv")

    if best_task1 is not None:
        rep_name, result = best_task1
        save_confusion_matrix(
            y_test_intent,
            result.y_pred,
            labels=list(range(len(le.classes_))),
            output_path=figures_dir / "task1_confusion_matrix.png",
            title=f"Intent Classification — {result.name} ({rep_name})",
        )

    print("\n=== Phase 2: Task 2 — Developer Behavior Prediction ===")
    task2_rows = []
    behavior_all = _representation_sets(
        df, train_idx, test_idx, embeddings, code_embeddings, config,
        include_intent_features=True,
    )
    behavior_keys = (
        "sbert+behavior_meta",
        "sbert+metadata",
        "metadata_only",
        "tfidf+metadata",
        "code_embed+metadata",
    )
    behavior_reps = {k: behavior_all[k] for k in behavior_keys if k in behavior_all}

    for target in ("has_generated_code", "has_tool_use"):
        y_train = df.iloc[train_idx][target].astype(int).to_numpy()
        y_test = df.iloc[test_idx][target].astype(int).to_numpy()
        for rep_name, (x_tr, x_te) in behavior_reps.items():
            results = train_binary_classifiers(
                x_tr, y_train, x_te, y_test, random_state=config.random_state
            )
            block = results_to_dataframe(f"behavior_{target}", rep_name, results)
            block["target"] = target
            task2_rows.append(block)

    task2_df = pd.concat(task2_rows, ignore_index=True)
    save_results_table(task2_df, tables_dir / "task2_behavior_prediction.csv")

    print("\n=== Phase 2: Task 3 — Interaction Complexity Regression ===")
    task3_rows = []
    reg_reps = {
        k: representations[k]
        for k in ("sbert+metadata", "sbert", "tfidf+metadata", "metadata_only")
        if k in representations
    }
    reg_targets = {
        "agent_response_length": "response_length",
        "generated_code_length": "code_length",
        "session_turn_count": "conversation_turns",
    }

    for col, alias in reg_targets.items():
        y_train = np.log1p(df.iloc[train_idx][col].astype(float).to_numpy())
        y_test = np.log1p(df.iloc[test_idx][col].astype(float).to_numpy())
        for rep_name, (x_tr, x_te) in reg_reps.items():
            results = train_regressors(
                x_tr, y_train, x_te, y_test, random_state=config.random_state
            )
            block = results_to_dataframe(f"complexity_{alias}", rep_name, results)
            block["target"] = alias
            task3_rows.append(block)

    task3_df = pd.concat(task3_rows, ignore_index=True)
    save_results_table(task3_df, tables_dir / "task3_complexity_regression.csv")

    print("\n=== Phase 3: Representation Comparison ===")
    repr_compare = (
        task1_df.groupby("representation", as_index=False)
        .agg(accuracy=("accuracy", "max"), macro_f1=("macro_f1", "max"))
        .sort_values("macro_f1", ascending=False)
    )
    save_results_table(repr_compare, tables_dir / "representation_comparison.csv")

    plot_model_comparison(
        task1_df,
        "macro_f1",
        figures_dir / "model_comparison_intent_macro_f1.png",
        title="Task 1 — Macro F1 by Model and Representation",
    )
    if "roc_auc" in task2_df.columns:
        plot_model_comparison(
            task2_df[task2_df["target"] == "has_generated_code"].dropna(subset=["roc_auc"]),
            "roc_auc",
            figures_dir / "model_comparison_code_roc_auc.png",
            title="Task 2 — ROC-AUC for has_generated_code",
        )
    plot_model_comparison(
        task3_df[task3_df["target"] == "response_length"],
        "r2",
        figures_dir / "model_comparison_response_r2.png",
        title="Task 3 — R² for log(response_length)",
    )

    meta_cols = [c for c in METADATA_FEATURE_COLS if c in df.columns]
    x_meta_tr = metadata_matrix(df.iloc[train_intent], meta_cols)
    x_meta_te = metadata_matrix(df.iloc[test_intent], meta_cols)
    rf_results = train_classifiers(
        x_meta_tr,
        y_train_intent,
        x_meta_te,
        y_test_intent,
        random_state=config.random_state,
    )
    rf = next((r for r in rf_results if r.name == "random_forest"), None)
    if rf and rf.feature_importances is not None:
        plot_feature_importance(
            rf.feature_importances,
            meta_cols,
            figures_dir / "feature_importance_metadata_rf.png",
            title="Metadata Feature Importance (Random Forest, intent)",
        )

    print("\n=== Phase 4: Research Report ===")
    report_path = output_dir / "ML_RESEARCH_REPORT.md"
    write_research_report(
        report_path,
        analysis=analysis,
        split_report=split_report,
        task1_df=task1_df,
        task2_df=task2_df,
        task3_df=task3_df,
        repr_df=repr_compare,
    )

    print(f"\nDone. Artifacts written to: {output_dir}")
    return report_path
