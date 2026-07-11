"""Local pipeline: clean, embed, cluster, and analyze a SpecStory prompt CSV."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import umap
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from .data_cleaning import CleaningConfig, clean_prompts_dataframe, print_cleaning_report

INTENT_RULES = {
    "debug_fix": r"\b(fix|bug|error|broken|crash|fail|debug|issue|exception|stack trace)\b",
    "implement": r"\b(add|create|implement|build|write|make|develop|scaffold)\b",
    "refactor": r"\b(refactor|clean up|reorganize|simplify|optimize|rewrite)\b",
    "explain": r"\b(explain|what does|how does|why does|understand|clarify|walk me through)\b",
    "test": r"\b(test|unit test|coverage|pytest|jest|spec\b|assert)\b",
    "documentation": r"\b(document|readme|docstring|comment|changelog)\b",
    "review": r"\b(review|check|look at|audit|inspect|evaluate)\b",
    "configure": r"\b(configure|setup|install|deploy|docker|ci/cd|pipeline|env)\b",
    "ui_frontend": r"\b(ui|css|style|layout|component|react|button|modal|frontend)\b",
    "data_sql": r"\b(sql|query|database|table|migration|schema|csv|dataframe)\b",
}

_STALE_COLS = [
    "cluster", "umap_x", "umap_y",
    "char_len", "word_len", "line_count",
    "has_question", "has_code_mention", "starts_imperative",
    "intent_debug_fix", "intent_implement", "intent_refactor",
    "intent_explain", "intent_test", "intent_documentation",
    "intent_review", "intent_configure", "intent_ui_frontend",
    "intent_data_sql", "intent_count", "primary_intent",
    "agent_len", "code_to_text_ratio",
]


@dataclass
class AnalysisConfig:
    input_csv: Path
    output_dir: Path = Path("outputs")
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    embedding_model: str = "all-MiniLM-L6-v2"
    batch_size: int = 64
    min_clusters: int = 4
    max_clusters: int = 20
    device: str | None = None
    show_plots: bool = False


@dataclass
class AnalysisResult:
    prompts_df: pd.DataFrame
    dropped_df: pd.DataFrame
    embeddings: np.ndarray
    best_k: int
    best_score: float
    output_dir: Path
    analysis_dir: Path
    clustered_csv: Path
    cleaned_csv: Path
    dropped_csv: Path | None
    report_path: Path


def _resolve_device(device: str | None) -> str:
    if device:
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _ensure_defaults(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, default in [
        ("agent_response", ""),
        ("generated_code", ""),
        ("agent_model", ""),
        ("code_languages", ""),
        ("code_block_count", 0),
        ("has_generated_code", False),
        ("has_tool_use", False),
    ]:
        if col not in out.columns:
            out[col] = default
    return out


def load_prompt_csv(path: str | Path) -> pd.DataFrame:
    """Load a mined or clustered prompt CSV and drop stale derived columns."""
    df = pd.read_csv(path)
    return _ensure_defaults(df.drop(columns=[c for c in _STALE_COLS if c in df.columns]))


def _maybe_show(show: bool) -> None:
    if show:
        plt.show()
    else:
        plt.close()


def _add_linguistics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    out = df.copy()
    out["char_len"] = out["prompt_text"].str.len()
    out["word_len"] = out["prompt_text"].str.split().str.len()
    out["line_count"] = out["prompt_text"].str.count("\n") + 1
    out["has_question"] = out["prompt_text"].str.contains(r"\?", regex=True)
    out["has_code_mention"] = out["prompt_text"].str.contains(
        r"```|\.py|\.js|\.ts|function |class |import ",
        case=False,
        regex=True,
    )
    out["starts_imperative"] = out["prompt_text"].str.match(
        r"^(add|create|fix|update|remove|delete|implement|write|build|make|refactor|explain|help|can you|please)\b",
        case=False,
    )
    ling = pd.Series({
        "median_chars": out["char_len"].median(),
        "median_words": out["word_len"].median(),
        "median_lines": out["line_count"].median(),
        "pct_questions": round(out["has_question"].mean() * 100, 1),
        "pct_imperative_open": round(out["starts_imperative"].mean() * 100, 1),
        "pct_mention_code": round(out["has_code_mention"].mean() * 100, 1),
        "pct_multiline_gt3": round((out["line_count"] > 3).mean() * 100, 1),
    })
    return out, ling


def _add_intents(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    out = df.copy()
    for intent, pattern in INTENT_RULES.items():
        out[f"intent_{intent}"] = out["prompt_text"].str.contains(pattern, case=False, regex=True)
    intent_cols = [f"intent_{k}" for k in INTENT_RULES]
    out["intent_count"] = out[intent_cols].sum(axis=1)
    out["primary_intent"] = out[intent_cols].apply(
        lambda row: row.idxmax().replace("intent_", "") if row.any() else "unclassified",
        axis=1,
    )
    intent_df = pd.Series({
        k: round(out[f"intent_{k}"].mean() * 100, 1) for k in INTENT_RULES
    }).sort_values(ascending=False)
    return out, intent_df, intent_cols


def _gini_coefficient(values) -> float:
    arr = np.sort(np.asarray(values, dtype=float))
    if len(arr) == 0 or arr.sum() == 0:
        return 0.0
    n = len(arr)
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * arr) / (n * arr.sum())) - (n + 1) / n)


def run_prompt_analysis(config: AnalysisConfig) -> AnalysisResult:
    """Clean, embed, cluster, and analyze a prompt CSV locally."""
    output_dir = Path(config.output_dir)
    analysis_dir = output_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {config.input_csv}")
    raw_df = load_prompt_csv(config.input_csv)
    print(f"Rows loaded: {len(raw_df)}")

    prompts_df, dropped_df, cleaning_report = clean_prompts_dataframe(
        raw_df,
        config=config.cleaning,
    )
    print("\n=== CLEANING REPORT ===")
    print_cleaning_report(cleaning_report)

    cleaned_csv = output_dir / "specstory_prompts_cleaned.csv"
    prompts_df.to_csv(cleaned_csv, index=False)
    print(f"Saved cleaned rows: {cleaned_csv}")

    dropped_csv: Path | None = None
    if not dropped_df.empty:
        dropped_csv = output_dir / "specstory_prompts_dropped.csv"
        dropped_df.to_csv(dropped_csv, index=False)
        print(f"Saved dropped rows: {dropped_csv}")

    texts = prompts_df["prompt_text"].tolist()
    device = _resolve_device(config.device)
    print(f"\nEmbedding {len(texts)} prompts on {device}...")
    model = SentenceTransformer(config.embedding_model, device=device)
    embeddings = model.encode(
        texts,
        batch_size=config.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    print("Embeddings shape:", embeddings.shape)

    candidate_ks = [k for k in range(config.min_clusters, config.max_clusters + 1) if k < len(texts)]
    best_k, best_score, best_labels = 1, -1.0, np.zeros(len(texts), dtype=int)
    for k in candidate_ks:
        labels = KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        print(f"k={k:2d}  silhouette={score:.4f}")
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels

    print(f"\nBest k = {best_k} (silhouette = {best_score:.4f})")
    prompts_df["cluster"] = best_labels

    n_neighbors = min(15, max(2, len(texts) - 1))
    coords = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    ).fit_transform(embeddings)
    prompts_df["umap_x"] = coords[:, 0]
    prompts_df["umap_y"] = coords[:, 1]

    plt.figure(figsize=(10, 7))
    sc = plt.scatter(coords[:, 0], coords[:, 1], c=best_labels, cmap="tab20", s=10, alpha=0.75)
    plt.title(f"SpecStory prompts — {best_k} clusters (UMAP)")
    plt.colorbar(sc, label="cluster")
    plt.tight_layout()
    umap_path = output_dir / "prompt_clusters_umap.png"
    plt.savefig(umap_path, dpi=150)
    _maybe_show(config.show_plots)

    prompts_df, ling = _add_linguistics(prompts_df)
    prompts_df, intent_df, intent_cols = _add_intents(prompts_df)

    if "agent_response" in prompts_df.columns:
        prompts_df["agent_len"] = prompts_df["agent_response"].fillna("").str.len()
        prompts_df["code_to_text_ratio"] = np.where(
            prompts_df["agent_len"] > 0,
            prompts_df["generated_code"].fillna("").str.len() / prompts_df["agent_len"],
            0,
        )
    else:
        prompts_df["agent_len"] = 0
        prompts_df["code_to_text_ratio"] = 0

    code_stats = pd.Series({
        "prompts_with_agent_response": round((prompts_df["agent_len"] > 50).mean() * 100, 1),
        "prompts_with_generated_code": round(prompts_df["has_generated_code"].mean() * 100, 1),
        "prompts_with_tool_use": round(prompts_df["has_tool_use"].mean() * 100, 1),
        "median_code_blocks": prompts_df["code_block_count"].median(),
        "median_code_to_text_ratio": round(prompts_df["code_to_text_ratio"].median(), 3),
    })

    repo_stats = (
        prompts_df.groupby("full_name")
        .agg(
            prompt_count=("prompt_text", "count"),
            with_code=("has_generated_code", "sum"),
            median_words=("word_len", "median"),
        )
        .reset_index()
        .sort_values("prompt_count", ascending=False)
    )
    repo_stats.to_csv(analysis_dir / "repo_stats.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].hist(prompts_df["word_len"], bins=40, color="#4C72B0", edgecolor="white")
    axes[0, 0].set_title("Word count distribution")
    axes[0, 1].hist(prompts_df["char_len"], bins=40, color="#55A868", edgecolor="white")
    axes[0, 1].set_title("Character count distribution")
    for ax, col, title in [
        (axes[1, 0], "has_question", "Contains ?"),
        (axes[1, 1], "starts_imperative", "Imperative opener"),
    ]:
        vc = prompts_df[col].value_counts()
        ax.bar(["no", "yes"], [vc.get(False, 0), vc.get(True, 0)], color=["#C44E52", "#55A868"])
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(analysis_dir / "03_prompt_linguistics.png", dpi=150)
    _maybe_show(config.show_plots)

    fig, ax = plt.subplots(figsize=(10, 6))
    intent_df.sort_values().plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_title("Heuristic developer intent prevalence")
    ax.set_xlabel("% of prompts")
    plt.tight_layout()
    plt.savefig(analysis_dir / "04_intent_taxonomy.png", dpi=150)
    intent_df.to_csv(analysis_dir / "intent_prevalence.csv")
    _maybe_show(config.show_plots)

    vectorizer = TfidfVectorizer(max_features=2000, stop_words="english", min_df=2)
    tfidf = vectorizer.fit_transform(prompts_df["prompt_text"])
    terms = np.array(vectorizer.get_feature_names_out())
    cluster_profiles = []
    for cluster_id in sorted(prompts_df["cluster"].unique()):
        mask = prompts_df["cluster"] == cluster_id
        sub = prompts_df[mask]
        mean_tfidf = np.asarray(tfidf[mask].mean(axis=0)).ravel()
        top_idx = mean_tfidf.argsort()[-8:][::-1]
        intent_mix = {ic.replace("intent_", ""): round(sub[ic].mean() * 100, 1) for ic in intent_cols}
        cluster_profiles.append({
            "cluster": cluster_id,
            "size": len(sub),
            "pct_of_corpus": round(len(sub) / len(prompts_df) * 100, 1),
            "top_terms": ", ".join(terms[top_idx]),
            "dominant_intent": max(intent_mix, key=intent_mix.get) if intent_mix else "n/a",
            "pct_with_code": round(sub["has_generated_code"].mean() * 100, 1),
            "median_words": sub["word_len"].median(),
        })
    cluster_df = pd.DataFrame(cluster_profiles)
    cluster_df.to_csv(analysis_dir / "cluster_profiles.csv", index=False)

    per_repo = prompts_df.groupby("full_name").size()
    per_file = prompts_df.groupby("source_file").size() if "source_file" in prompts_df.columns else per_repo
    diversity = pd.Series({
        "gini_repos": round(_gini_coefficient(per_repo.values), 3),
        "gini_files": round(_gini_coefficient(per_file.values), 3),
        "top10_repo_share_pct": round(per_repo.sort_values(ascending=False).head(10).sum() / per_repo.sum() * 100, 1),
        "top10_file_share_pct": round(per_file.sort_values(ascending=False).head(10).sum() / per_file.sum() * 100, 1),
        "herfindahl_repos": round(((per_repo / per_repo.sum()) ** 2).sum(), 4),
        "effective_n_repos": round(1 / ((per_repo / per_repo.sum()) ** 2).sum(), 1),
    })
    diversity.to_csv(analysis_dir / "diversity_metrics.csv")

    if "has_generated_code" in prompts_df.columns and intent_cols:
        cross = prompts_df.groupby("primary_intent").agg(
            n=("prompt_text", "count"),
            pct_code=("has_generated_code", "mean"),
            pct_tool=("has_tool_use", "mean"),
            median_agent_len=("agent_len", "median"),
        ).sort_values("n", ascending=False)
        cross["pct_code"] = (cross["pct_code"] * 100).round(1)
        cross["pct_tool"] = (cross["pct_tool"] * 100).round(1)
        cross.to_csv(analysis_dir / "intent_vs_agent_behavior.csv", index=False)

    report_lines = [
        "# SpecStory Corpus — Research Report",
        "",
        f"Generated: {pd.Timestamp.utcnow().isoformat()}",
        "",
        "## 1. Corpus scale",
        f"- **{len(prompts_df):,}** cleaned developer prompts",
        f"- **{prompts_df['full_name'].nunique():,}** repositories",
        f"- **{prompts_df['source_file'].nunique() if 'source_file' in prompts_df.columns else 'n/a'}** session log files",
        "",
        "## 2. Cleaning",
        f"- Input rows: {cleaning_report.rows_in:,}",
        f"- Output rows: {cleaning_report.rows_out:,}",
        f"- Dropped rows: {cleaning_report.rows_in - cleaning_report.rows_out:,}",
        "",
        "## 3. Prompt characteristics",
        f"- Median length: **{prompts_df['word_len'].median():.0f} words** / {prompts_df['char_len'].median():.0f} chars",
        f"- {ling['pct_questions']}% contain questions; {ling['pct_imperative_open']}% start with imperatives",
        f"- {ling['pct_mention_code']}% mention code/syntax explicitly",
        "",
        "## 4. Developer intent (heuristic)",
    ]
    for intent, pct in intent_df.head(5).items():
        report_lines.append(f"- **{intent}**: {pct}% of prompts")
    report_lines += [
        "",
        "## 5. Agent / code-generation behavior",
        f"- {code_stats['prompts_with_generated_code']}% of prompts paired with fenced code in agent response",
        f"- {code_stats['prompts_with_tool_use']}% involve tool-use blocks",
        f"- Median code-to-response ratio: {code_stats['median_code_to_text_ratio']}",
        "",
        "## 6. Clustering",
        f"- k={best_k} clusters (silhouette={best_score:.3f})",
    ]
    for _, row in cluster_df.head(5).iterrows():
        report_lines.append(
            f"- Cluster {row['cluster']}: n={row['size']} ({row['pct_of_corpus']}%), "
            f"intent~{row['dominant_intent']}, terms: {row['top_terms']}"
        )
    report_lines += [
        "",
        "## 7. Diversity",
        f"- Gini (repos): {diversity['gini_repos']} — top 10 repos supply {diversity['top10_repo_share_pct']}% of prompts",
        f"- Effective number of contributing repos: {diversity['effective_n_repos']}",
        "",
        "## Output files",
        f"- {output_dir / 'specstory_prompts_clustered.csv'}",
        f"- {analysis_dir}",
    ]
    report_path = analysis_dir / "RESEARCH_REPORT.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    clustered_csv = output_dir / "specstory_prompts_clustered.csv"
    prompts_df.to_csv(clustered_csv, index=False)
    np.save(output_dir / "prompt_embeddings.npy", embeddings)

    print("\n=== SAVED ARTIFACTS ===")
    print(f"Clustered dataset : {clustered_csv}")
    print(f"UMAP plot         : {umap_path}")
    print(f"Research report   : {report_path}")
    print(f"Analysis folder   : {analysis_dir}")

    return AnalysisResult(
        prompts_df=prompts_df,
        dropped_df=dropped_df,
        embeddings=embeddings,
        best_k=best_k,
        best_score=best_score,
        output_dir=output_dir,
        analysis_dir=analysis_dir,
        clustered_csv=clustered_csv,
        cleaned_csv=cleaned_csv,
        dropped_csv=dropped_csv,
        report_path=report_path,
    )
