"""Dataset loading, feature engineering, and exploratory analysis."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..prompt_analysis import INTENT_RULES, _add_intents, _add_linguistics, load_prompt_csv

METADATA_FEATURE_COLS = [
    "char_len",
    "word_len",
    "line_count",
    "has_question",
    "has_code_mention",
    "starts_imperative",
    "prompt_length",
    "prompt_word_count",
    "turn_index",
]

INTENT_METADATA_COLS = METADATA_FEATURE_COLS + ["intent_count"]

INTENT_BINARY_COLS = [f"intent_{k}" for k in INTENT_RULES]

REPO_FEATURE_COLS = [
    "repo_prompt_count",
    "repo_median_words",
    "repo_code_rate",
    "repo_tool_rate",
    "repo_stars_log",
]

PROMPT_TYPE_RULES = {
    "imperative": r"^(add|create|fix|update|remove|delete|implement|write|build|make|refactor|help)\b",
    "question": r"\?",
    "context_rich": r"(here is|below is|attached|context|given the following|as shown)",
    "debug": r"\b(error|bug|fix|crash|fail|exception|stack)\b",
    "specification": r"\b(should|must|require|ensure|accept|return|implement)\b",
}


def derive_prompt_type(text: str) -> str:
    """Weak label for prompt rhetorical style."""
    if not isinstance(text, str) or not text.strip():
        return "other"
    lower = text.lower().strip()
    scores: dict[str, int] = {}
    for label, pattern in PROMPT_TYPE_RULES.items():
        scores[label] = len(re.findall(pattern, lower, flags=re.IGNORECASE))
    if scores.get("question", 0) > 0:
        return "question"
    best = max(scores, key=scores.get) if scores else "other"
    return best if scores.get(best, 0) > 0 else "other"


def _gini(values: np.ndarray) -> float:
    arr = np.sort(np.asarray(values, dtype=float))
    if len(arr) == 0 or arr.sum() == 0:
        return 0.0
    n = len(arr)
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * arr) / (n * arr.sum())) - (n + 1) / n)


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add linguistics, intents, prompt_type, and session-level targets."""
    out, _ = _add_linguistics(df)
    out, _, _ = _add_intents(out)

    if "prompt_type" not in out.columns:
        out["prompt_type"] = out["prompt_text"].map(derive_prompt_type)

    if "agent_response_length" in out.columns:
        out["agent_response_length"] = pd.to_numeric(
            out["agent_response_length"], errors="coerce"
        ).fillna(0).astype(int)
    elif "agent_len" in out.columns:
        out["agent_response_length"] = out["agent_len"].fillna(0).astype(int)
    else:
        out["agent_response_length"] = out.get("agent_response", pd.Series([""] * len(out))).fillna("").str.len()

    if "generated_code_length" not in out.columns:
        out["generated_code_length"] = out.get("generated_code", pd.Series([""] * len(out))).fillna("").str.len()
    out["generated_code_length"] = pd.to_numeric(out["generated_code_length"], errors="coerce").fillna(0)

    if "code_to_text_ratio" not in out.columns:
        out["code_to_text_ratio"] = np.where(
            out["agent_response_length"] > 0,
            out["generated_code_length"] / out["agent_response_length"],
            0.0,
        )

    group_key = "source_file" if "source_file" in out.columns else "full_name"
    session_turns = out.groupby(group_key)["turn_index"].transform("max") + 1
    out["session_turn_count"] = session_turns.astype(int)

    repo_stats = (
        out.groupby("full_name")
        .agg(
            repo_prompt_count=("prompt_text", "count"),
            repo_median_words=("word_len", "median"),
            repo_code_rate=("has_generated_code", "mean"),
            repo_tool_rate=("has_tool_use", "mean"),
        )
        .reset_index()
    )
    out = out.merge(repo_stats, on="full_name", how="left")

    if "stargazers_count" in out.columns:
        out["repo_stars_log"] = np.log1p(pd.to_numeric(out["stargazers_count"], errors="coerce").fillna(0))
    else:
        out["repo_stars_log"] = 0.0

    return out


def load_ml_dataset(path: str | Path) -> pd.DataFrame:
    """Load clustered or raw prompt CSV and enrich with ML features."""
    df = load_prompt_csv(path)
    return add_derived_columns(df)


def analyze_dataset(df: pd.DataFrame) -> dict:
    """Phase 1: corpus statistics, missingness, imbalance, and leakage risks."""
    n = len(df)
    missing = df.isna().sum()
    missing_pct = (missing / n * 100).round(2)
    missing_table = (
        pd.DataFrame({"column": missing.index, "missing": missing.values, "pct": missing_pct.values})
        .query("missing > 0")
        .sort_values("missing", ascending=False)
    )

    label_dists = {
        "primary_intent": df["primary_intent"].value_counts().to_dict(),
        "prompt_type": df["prompt_type"].value_counts().to_dict(),
        "cluster": df["cluster"].value_counts().to_dict() if "cluster" in df.columns else {},
        "has_generated_code": df["has_generated_code"].value_counts().to_dict(),
        "has_tool_use": df["has_tool_use"].value_counts().to_dict(),
    }

    intent_counts = df["primary_intent"].value_counts()
    imbalance = {
        "majority_class": intent_counts.index[0] if len(intent_counts) else None,
        "majority_share_pct": round(intent_counts.iloc[0] / n * 100, 2) if len(intent_counts) else 0,
        "minority_classes_lt_30": intent_counts[intent_counts < 30].to_dict(),
        "intent_entropy": float(-(intent_counts / n * np.log(intent_counts / n + 1e-12)).sum()),
    }

    exact_dupes = int(df["prompt_text"].duplicated().sum())
    norm_dupes = int(df["prompt_text"].str.lower().str.strip().duplicated().sum())

    per_repo = df.groupby("full_name").size()
    per_file = df.groupby("source_file").size() if "source_file" in df.columns else per_repo
    leakage = {
        "n_repos": int(df["full_name"].nunique()),
        "n_sessions": int(df["source_file"].nunique()) if "source_file" in df.columns else None,
        "gini_repos": round(_gini(per_repo.values), 4),
        "top10_repo_share_pct": round(per_repo.sort_values(ascending=False).head(10).sum() / n * 100, 2),
        "herfindahl_repos": round(((per_repo / per_repo.sum()) ** 2).sum(), 4),
        "effective_n_repos": round(1 / ((per_repo / per_repo.sum()) ** 2).sum(), 2),
        "max_repo_prompts": int(per_repo.max()),
        "repos_with_gt_100_prompts": int((per_repo > 100).sum()),
        "same_repo_train_test_risk": (
            "HIGH — use GroupKFold by full_name; never random row split for evaluation"
        ),
        "same_session_risk": (
            "MEDIUM — multiple turns per source_file share context; prefer repo-level or session-level splits"
        ),
        "label_leakage_notes": [
            "primary_intent is derived from prompt_text regex — expect inflated in-sample accuracy",
            "cluster labels were fit on full corpus — do not use as features for same-corpus tasks",
            "repo aggregate features (repo_code_rate) leak label distribution if computed on full data",
        ],
    }

    feature_roles = {
        "ml_inputs": {
            "text_representations": ["prompt_text → SBERT / TF-IDF / code embeddings"],
            "prompt_metadata": METADATA_FEATURE_COLS,
            "intent_derived_metadata": INTENT_BINARY_COLS + ["intent_count"],
            "repository_metadata": REPO_FEATURE_COLS + ["full_name (for grouping only)"],
            "categorical": ["artifact_type", "prompt_type", "agent_model"],
        },
        "prediction_targets": {
            "classification": {
                "primary_intent": "heuristic developer intent (weak label)",
                "prompt_type": "rhetorical style (weak label)",
                "has_generated_code": "binary — agent produced fenced code",
                "has_tool_use": "binary — agent invoked tools",
            },
            "regression": {
                "agent_response_length": "agent response size in characters",
                "generated_code_length": "generated code size in characters",
                "session_turn_count": "conversation length for the session",
            },
            "unsupervised_only": ["cluster", "umap_x", "umap_y"],
            "do_not_use_as_input": [
                "agent_response",
                "generated_code",
                "agent_response_length",
                "generated_code_length",
                "has_generated_code",
                "has_tool_use",
                "code_to_text_ratio",
                "session_turn_count",
            ],
        },
    }

    return {
        "n_samples": n,
        "n_columns": len(df.columns),
        "missing_values": missing_table.to_dict(orient="records"),
        "label_distributions": label_dists,
        "class_imbalance": imbalance,
        "duplicate_prompts": {"exact": exact_dupes, "normalized": norm_dupes},
        "repository_leakage": leakage,
        "feature_roles": feature_roles,
    }


def save_dataset_analysis(analysis: dict, output_dir: Path) -> None:
    """Write Phase 1 tables and JSON summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(analysis, indent=2, default=str),
        encoding="utf-8",
    )

    rows = []
    for label, dist in analysis["label_distributions"].items():
        for cls, count in dist.items():
            rows.append({
                "label_column": label,
                "class": cls,
                "count": count,
                "pct": round(count / analysis["n_samples"] * 100, 2),
            })
    pd.DataFrame(rows).to_csv(output_dir / "label_distributions.csv", index=False, encoding="utf-8-sig")

    if analysis["missing_values"]:
        pd.DataFrame(analysis["missing_values"]).to_csv(
            output_dir / "missing_values.csv", index=False, encoding="utf-8-sig"
        )

    pd.DataFrame([analysis["class_imbalance"]]).to_csv(
        output_dir / "class_imbalance.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame([analysis["duplicate_prompts"]]).to_csv(
        output_dir / "duplicate_prompts.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame([analysis["repository_leakage"]]).to_csv(
        output_dir / "leakage_risks.csv", index=False, encoding="utf-8-sig"
    )
