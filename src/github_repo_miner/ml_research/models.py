"""Model training for classification and regression tasks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier, XGBRegressor

    HAS_XGB = True
except ImportError:
    HAS_XGB = False


@dataclass
class ModelResult:
    name: str
    metrics: dict
    y_pred: np.ndarray
    y_proba: np.ndarray | None = None
    feature_importances: np.ndarray | None = None


def _maybe_scale_dense(x_train, x_test):
    from scipy.sparse import issparse

    if issparse(x_train):
        return x_train, x_test, None
    scaler = StandardScaler()
    return scaler.fit_transform(x_train), scaler.transform(x_test), scaler


def train_classifiers(
    x_train,
    y_train,
    x_test,
    y_test,
    *,
    random_state: int = 42,
) -> list[ModelResult]:
    from scipy.sparse import issparse

    x_tr, x_te, _ = _maybe_scale_dense(x_train, x_test)
    results: list[ModelResult] = []
    sparse_input = issparse(x_train)

    models: dict = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=150,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
    }
    if not sparse_input:
        models["mlp"] = Pipeline([
            ("scaler", StandardScaler()),
            (
                "clf",
                MLPClassifier(
                    hidden_layer_sizes=(256, 128),
                    max_iter=400,
                    early_stopping=True,
                    random_state=random_state,
                ),
            ),
        ])
    if HAS_XGB:
        models["xgboost"] = XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=-1,
        )

    from sklearn.metrics import accuracy_score, f1_score

    for name, model in models.items():
        if name == "mlp":
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            y_proba = model.predict_proba(x_test) if hasattr(model, "predict_proba") else None
            importances = None
        else:
            fit_x = x_train if name == "random_forest" and sparse_input else x_tr
            pred_x = x_test if name == "random_forest" and sparse_input else x_te
            model.fit(fit_x, y_train)
            y_pred = model.predict(pred_x)
            y_proba = model.predict_proba(pred_x) if hasattr(model, "predict_proba") else None
            importances = getattr(model, "feature_importances_", None)

        results.append(
            ModelResult(
                name=name,
                metrics={
                    "accuracy": float(accuracy_score(y_test, y_pred)),
                    "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
                },
                y_pred=y_pred,
                y_proba=y_proba,
                feature_importances=importances,
            )
        )
    return results


def train_binary_classifiers(
    x_train,
    y_train,
    x_test,
    y_test,
    *,
    random_state: int = 42,
) -> list[ModelResult]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    x_tr, x_te, _ = _maybe_scale_dense(x_train, x_test)
    results: list[ModelResult] = []

    models: dict = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=150,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
    }
    if HAS_XGB:
        models["xgboost"] = XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            scale_pos_weight=float((y_train == 0).sum()) / max((y_train == 1).sum(), 1),
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )

    for name, model in models.items():
        model.fit(x_tr, y_train)
        y_pred = model.predict(x_te)
        y_proba = model.predict_proba(x_te)[:, 1] if hasattr(model, "predict_proba") else None
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        }
        if y_proba is not None and len(np.unique(y_test)) > 1:
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
            metrics["avg_precision"] = float(average_precision_score(y_test, y_proba))
        results.append(
            ModelResult(
                name=name,
                metrics=metrics,
                y_pred=y_pred,
                y_proba=y_proba,
                feature_importances=getattr(model, "feature_importances_", None),
            )
        )
    return results


def train_regressors(
    x_train,
    y_train,
    x_test,
    y_test,
    *,
    random_state: int = 42,
) -> list[ModelResult]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    x_tr, x_te, _ = _maybe_scale_dense(x_train, x_test)
    results: list[ModelResult] = []

    models: dict = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=150,
            random_state=random_state,
            n_jobs=-1,
        ),
    }
    if HAS_XGB:
        models["xgboost"] = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            random_state=random_state,
            n_jobs=-1,
        )

    for name, model in models.items():
        model.fit(x_tr, y_train)
        y_pred = model.predict(x_te)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        results.append(
            ModelResult(
                name=name,
                metrics={
                    "mae": float(mean_absolute_error(y_test, y_pred)),
                    "rmse": rmse,
                    "r2": float(r2_score(y_test, y_pred)),
                },
                y_pred=y_pred,
                feature_importances=getattr(model, "feature_importances_", None),
            )
        )
    return results
