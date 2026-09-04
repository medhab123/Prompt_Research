"""Robustness experiments: baselines, keyword ablation, leakage-safe repo features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .baselines import (
    keyword_heuristic_baseline,
    majority_class_baseline,
    mask_keywords,
)
from .config import MLResearchConfig
from .data import METADATA_FEATURE_COLS, load_ml_dataset
from .evaluate import save_results_table
from .label_audit import save_label_audit
from .models import train_binary_classifiers
from .representations import (
    build_tfidf,
    hstack_features,
    load_or_compute_embeddings,
    metadata_matrix,
)
from .splits import repo_train_test_split, split_overlap_report

# Metadata safe for behavior prediction (no intent regex flags).
SAFE_METADATA_COLS = [
    "char_len", "word_len", "line_count", "has_question",
    "has_code_mention", "starts_imperative", "turn_index",
]

REPO_STAT_COLS = [
    "repo_prompt_count", "repo_median_words",
    "repo_code_rate_train", "repo_tool_rate_train",
]


def add_train_only_repo_stats(
    df: pd.DataFrame,
    train_idx: np.ndarray,
) -> pd.DataFrame:
    """Compute repo aggregates on TRAIN only to avoid label leakage."""
    out = df.copy()
    train = out.iloc[train_idx]
    stats = (
        train.groupby("full_name")
        .agg(
            repo_prompt_count=("prompt_text", "count"),
            repo_median_words=("word_len", "median"),
            repo_code_rate_train=("has_generated_code", "mean"),
            repo_tool_rate_train=("has_tool_use", "mean"),
        )
        .reset_index()
    )
    global_code = float(train["has_generated_code"].mean())
    global_tool = float(train["has_tool_use"].mean())
    out = out.drop(columns=[c for c in REPO_STAT_COLS if c in out.columns], errors="ignore")
    out = out.merge(stats, on="full_name", how="left")
    out["repo_code_rate_train"] = out["repo_code_rate_train"].fillna(global_code)
    out["repo_tool_rate_train"] = out["repo_tool_rate_train"].fillna(global_tool)
    out["repo_prompt_count"] = out["repo_prompt_count"].fillna(1)
    out["repo_median_words"] = out["repo_median_words"].fillna(out["word_len"].median())
    return out


def _run_ml_block(
    x_train,
    y_train,
    x_test,
    y_test,
    *,
    model_name: str = "xgboost",
    random_state: int = 42,
) -> dict:
    results = train_binary_classifiers(
        x_train, y_train, x_test, y_test, random_state=random_state
    )
    match = next((r for r in results if r.name == model_name), results[0])
    return match.metrics


def run_robustness_experiments(config: MLResearchConfig) -> Path:
    """Run baselines + datasets A/B/C with leakage-safe repo features."""
    output_dir = Path(config.output_dir) / "robustness"
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    df = load_ml_dataset(config.input_csv)
    train_idx, test_idx = repo_train_test_split(
        df, test_size=config.test_size, random_state=config.random_state
    )
    df = add_train_only_repo_stats(df, train_idx)

    emb_path = Path(config.output_dir) / "prompt_embeddings.npy"
    embeddings_full = load_or_compute_embeddings(
        df["prompt_text"].tolist(),
        emb_path,
        config.embedding_model,
        batch_size=config.batch_size,
        device=config.device,
    )

    # Dataset variants
    texts_a = df["prompt_text"].tolist()
    texts_b = df["prompt_text"].map(mask_keywords).tolist()
    emb_b = load_or_compute_embeddings(
        texts_b,
        Path(config.output_dir) / "prompt_embeddings_keyword_masked.npy",
        config.embedding_model,
        batch_size=config.batch_size,
        device=config.device,
    )

    split_report = split_overlap_report(df, train_idx, test_idx)
    pd.DataFrame([split_report]).to_csv(tables_dir / "split_report.csv", index=False, encoding="utf-8-sig")

    label_summary = save_label_audit(df, tables_dir)

    rows: list[dict] = []
    targets = ("has_generated_code", "has_tool_use")

    for target in targets:
        y_train = df.iloc[train_idx][target].astype(int).to_numpy()
        y_test = df.iloc[test_idx][target].astype(int).to_numpy()
        test_texts = df.iloc[test_idx]["prompt_text"].tolist()

        # Baselines
        maj = majority_class_baseline(y_train, y_test)
        kw = keyword_heuristic_baseline(test_texts, y_test, target=target)
        for bl in (maj, kw):
            rows.append({
                "dataset": "full",
                "representation": bl.name,
                "model": "rule",
                "target": target,
                "auc": bl.metrics.get("roc_auc"),
                "f1": bl.metrics.get("f1"),
                "accuracy": bl.metrics.get("accuracy"),
                "coverage": bl.coverage,
            })

        meta_train = metadata_matrix(df.iloc[train_idx], SAFE_METADATA_COLS)
        meta_test = metadata_matrix(df.iloc[test_idx], SAFE_METADATA_COLS)
        repo_train = metadata_matrix(df.iloc[train_idx], REPO_STAT_COLS)
        repo_test = metadata_matrix(df.iloc[test_idx], REPO_STAT_COLS)

        experiments = [
            # Dataset A: full prompt
            ("A_full", "sbert", embeddings_full[train_idx], embeddings_full[test_idx]),
            ("A_full", "sbert+safe_meta", hstack_features(embeddings_full[train_idx], meta_train),
             hstack_features(embeddings_full[test_idx], meta_test)),
            ("A_full", "repo_train_only", repo_train, repo_test),
            ("A_full", "sbert+repo_train", hstack_features(embeddings_full[train_idx], repo_train),
             hstack_features(embeddings_full[test_idx], repo_test)),
            ("A_full", "tfidf", *build_tfidf(
                [texts_a[i] for i in train_idx],
                [texts_a[i] for i in test_idx],
                max_features=config.tfidf_max_features,
            )[:2]),
            # Dataset B: keyword masked
            ("B_keyword_masked", "sbert", emb_b[train_idx], emb_b[test_idx]),
            ("B_keyword_masked", "sbert+safe_meta", hstack_features(emb_b[train_idx], meta_train),
             hstack_features(emb_b[test_idx], meta_test)),
            # Dataset C: semantic only (no TF-IDF, no repo)
            ("C_semantic_only", "sbert", embeddings_full[train_idx], embeddings_full[test_idx]),
        ]

        for dataset, rep, x_tr, x_te in experiments:
            metrics = _run_ml_block(x_tr, y_train, x_te, y_test, random_state=config.random_state)
            rows.append({
                "dataset": dataset,
                "representation": rep,
                "model": "xgboost",
                "target": target,
                "auc": metrics.get("roc_auc"),
                "f1": metrics.get("f1"),
                "accuracy": metrics.get("accuracy"),
                "coverage": 1.0,
            })

    results_df = pd.DataFrame(rows)
    save_results_table(results_df, tables_dir / "robustness_comparison.csv")

    # Ablation summary for has_generated_code
    code_df = results_df[results_df["target"] == "has_generated_code"].copy()
    ablation = code_df.pivot_table(
        index=["dataset", "representation"],
        values=["auc", "f1"],
        aggfunc="max",
    ).reset_index()
    save_results_table(ablation, tables_dir / "code_prediction_ablation.csv")

    report_lines = [
        "# Robustness Experiment Report",
        "",
        "## Label audit",
        f"- Misleading `has_generated_code` labels (explain/debug prompt + code in reply): "
        f"**{label_summary['pct_misleading_code_label']}%**",
        f"- Implementation prompt but no code: **{label_summary['pct_missed_code']}%**",
        "",
        "## Key comparisons (has_generated_code, test set)",
        "",
        "| Baseline / Model | AUC | F1 |",
        "|---|---|---|",
    ]
    for _, r in code_df.sort_values("auc", ascending=False).head(12).iterrows():
        report_lines.append(
            f"| {r['dataset']} / {r['representation']} / {r['model']} | "
            f"{r.get('auc', float('nan')):.3f} | {r.get('f1', float('nan')):.3f} |"
        )

    report_lines += [
        "",
        "## Interpretation guide",
        "",
        "- If **repo_train_only ≈ sbert+repo_train** and both beat **sbert alone**, signal is mostly repository context.",
        "- If **keyword_heuristic ≈ ML** on Dataset A, results are trivial lexical matching.",
        "- If **B_keyword_masked sbert** drops sharply vs **A_full sbert**, model relied on keywords.",
        "- **Majority class** is the minimum bar; beat it by a wide margin for a real contribution.",
        "",
        f"Split: {split_report['train_repos']} train repos, {split_report['test_repos']} test repos, "
        f"leakage_free={split_report['leakage_free']}",
    ]
    report_path = output_dir / "ROBUSTNESS_REPORT.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return report_path
