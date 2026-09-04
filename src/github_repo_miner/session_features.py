"""Session-level features for multi-turn agent behavior modeling."""

from __future__ import annotations

import pandas as pd


def make_session_id(df: pd.DataFrame) -> pd.Series:
    """Stable session key: one SpecStory history file in one repo."""
    return df["full_name"].astype(str) + "|" + df["source_file"].astype(str)


def add_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add conversation-context columns for each turn.

    Features:
    - session_id
    - session_turn_count
    - turns_before (0-indexed position in session)
    - prev_turn_had_code
    - prev_turn_had_tools
    - prev_action_type (after action labeling)
  """
    out = df.copy()
    if "session_id" not in out.columns:
        out["session_id"] = make_session_id(out)

    if "turn_index" not in out.columns:
        out["turn_index"] = out.groupby("session_id").cumcount() + 1

    out = out.sort_values(["session_id", "turn_index"]).reset_index(drop=True)

    out["session_turn_count"] = out.groupby("session_id")["turn_index"].transform("max")
    out["turns_before"] = out["turn_index"] - 1

    out["prev_turn_had_code"] = (
        out.groupby("session_id")["has_generated_code"]
        .shift(1)
        .fillna(False)
        .infer_objects(copy=False)
        .astype(bool)
    )
    out["prev_turn_had_tools"] = (
        out.groupby("session_id")["has_tool_use"]
        .shift(1)
        .fillna(False)
        .infer_objects(copy=False)
        .astype(bool)
    )

    if "action_type" in out.columns:
        out["prev_action_type"] = out.groupby("session_id")["action_type"].shift(1).fillna("")

    # Rolling code rate in session so far (excluding current turn).
    out["session_code_rate_before"] = (
        out.groupby("session_id")["has_generated_code"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
        .fillna(0.0)
    )

    return out


def enrich_turn_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Add action labels and session context features."""
    from .action_labeling import add_action_labels

    labeled = add_action_labels(df)
    return add_session_features(labeled)
