"""Repository-grouped train/test splits to reduce leakage."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


def repo_train_test_split(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    group_col: str = "full_name",
) -> tuple[np.ndarray, np.ndarray]:
    groups = df[group_col].to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=groups))
    return train_idx, test_idx


def repo_group_kfold(
    df: pd.DataFrame,
    n_splits: int = 5,
    *,
    group_col: str = "full_name",
) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = df[group_col].to_numpy()
    n_groups = df[group_col].nunique()
    folds = min(n_splits, n_groups)
    if folds < 2:
        raise ValueError("Need at least 2 repository groups for cross-validation.")
    gkf = GroupKFold(n_splits=folds)
    return list(gkf.split(df, groups=groups))


def split_overlap_report(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    group_col: str = "full_name",
) -> dict:
    train_groups = set(df.iloc[train_idx][group_col])
    test_groups = set(df.iloc[test_idx][group_col])
    overlap = train_groups & test_groups
    return {
        "train_rows": len(train_idx),
        "test_rows": len(test_idx),
        "train_repos": len(train_groups),
        "test_repos": len(test_groups),
        "overlapping_repos": len(overlap),
        "leakage_free": len(overlap) == 0,
    }
