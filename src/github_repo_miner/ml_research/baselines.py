"""Rule-based and trivial baselines for behavior prediction."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# High-signal lexical shortcuts reviewers will suspect.
CODE_KEYWORDS = re.compile(
    r"\b("
    r"implement|create|add|write|build|make|develop|scaffold|"
    r"function|class|method|module|component|api|endpoint|"
    r"refactor|generate|code|script|file"
    r")\b",
    re.IGNORECASE,
)

NO_CODE_KEYWORDS = re.compile(
    r"\b("
    r"why|explain|what|how|describe|clarify|understand|"
    r"error|issue|fail|failing|crash|bug|broken|exception|"
    r"help me understand|walk me through"
    r")\b",
    re.IGNORECASE,
)

TOOL_KEYWORDS = re.compile(
    r"\b("
    r"run|execute|test|grep|search|find|read|fetch|install|"
    r"deploy|build|compile|lint|terminal|shell|command|"
    r"file|directory|path"
    r")\b",
    re.IGNORECASE,
)

KEYWORDS_TO_MASK = [
    "implement", "create", "add", "write", "fix", "bug", "error",
    "function", "class", "code", "test", "run", "build", "make",
    "develop", "refactor", "explain", "why", "how", "debug",
    "issue", "fail", "crash", "exception", "generate", "script",
    "method", "module", "component", "api", "endpoint", "file",
]


@dataclass
class BaselineResult:
    name: str
    metrics: dict
    coverage: float | None = None  # fraction of test rows where rule fired


def mask_keywords(text: str) -> str:
    """Dataset B: replace high-signal behavior keywords with [MASK]."""
    out = text
    for kw in KEYWORDS_TO_MASK:
        out = re.sub(rf"\b{re.escape(kw)}\b", "[MASK]", out, flags=re.IGNORECASE)
    return out


def keyword_predict_code(text: str) -> int | None:
    """Return 1/0 if heuristic applies, else None (abstain)."""
    if CODE_KEYWORDS.search(text):
        return 1
    if NO_CODE_KEYWORDS.search(text):
        return 0
    return None


def keyword_predict_tool(text: str) -> int | None:
    if TOOL_KEYWORDS.search(text):
        return 1
    return None


def majority_class_baseline(y_train: np.ndarray, y_test: np.ndarray) -> BaselineResult:
    majority = int(np.bincount(y_train.astype(int)).argmax())
    y_pred = np.full(len(y_test), majority, dtype=int)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }
    if len(np.unique(y_test)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_pred))
        metrics["avg_precision"] = float(average_precision_score(y_test, y_pred))
    return BaselineResult(name="majority_class", metrics=metrics, coverage=1.0)


def keyword_heuristic_baseline(
    texts: list[str],
    y_test: np.ndarray,
    *,
    target: str = "has_generated_code",
) -> BaselineResult:
    predict_fn = keyword_predict_code if target == "has_generated_code" else keyword_predict_tool
    preds: list[int | None] = [predict_fn(str(t)) for t in texts]
    covered = np.array([p is not None for p in preds])
    coverage = float(covered.mean())

    if covered.sum() == 0:
        return BaselineResult(
            name="keyword_heuristic",
            metrics={"accuracy": 0.0, "f1": 0.0, "roc_auc": 0.5},
            coverage=0.0,
        )

    y_pred = np.array([p if p is not None else 0 for p in preds], dtype=int)
    y_cov = y_test[covered]
    p_cov = y_pred[covered]

    # Full-set metrics: abstentions default to 0
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "accuracy_on_covered": float(accuracy_score(y_cov, p_cov)),
        "f1_on_covered": float(f1_score(y_cov, p_cov, zero_division=0)),
    }
    if len(np.unique(y_test)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_pred))
    return BaselineResult(name="keyword_heuristic", metrics=metrics, coverage=coverage)


def always_positive_baseline(y_test: np.ndarray) -> BaselineResult:
    y_pred = np.ones(len(y_test), dtype=int)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }
    if len(np.unique(y_test)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_pred))
    return BaselineResult(name="always_positive", metrics=metrics, coverage=1.0)
