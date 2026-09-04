"""Audit label quality for has_generated_code and has_tool_use."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# Cases where label is semantically misleading for "agent chose to write code"
MISLEADING_PATTERNS = [
    ("explain_with_code", re.compile(r"\b(why|explain|what|how)\b", re.I)),
    ("debug_with_snippet", re.compile(r"\b(error|crash|bug|fail|exception)\b", re.I)),
    ("review_only", re.compile(r"\b(review|look at|audit|check)\b", re.I)),
    ("question_mark", re.compile(r"\?")),
]


def audit_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Flag rows where has_generated_code may not mean 'agent chose implementation'."""
    rows = []
    for idx, row in df.iterrows():
        text = str(row.get("prompt_text", ""))
        has_code = bool(row.get("has_generated_code", False))
        flags = []
        for name, pat in MISLEADING_PATTERNS:
            if pat.search(text):
                flags.append(name)
        if has_code and flags:
            category = "code_despite_non_implementation_prompt"
        elif not has_code and re.search(
            r"\b(implement|create|add|write|build|scaffold)\b", text, re.I
        ):
            category = "no_code_despite_implementation_prompt"
        else:
            category = "consistent"
        rows.append({
            "index": idx,
            "full_name": row.get("full_name", ""),
            "primary_intent": row.get("primary_intent", ""),
            "has_generated_code": has_code,
            "has_tool_use": bool(row.get("has_tool_use", False)),
            "word_len": row.get("word_len", 0),
            "flags": ",".join(flags),
            "audit_category": category,
            "prompt_preview": text[:200].replace("\n", " "),
        })
    return pd.DataFrame(rows)


def summarize_label_audit(audit_df: pd.DataFrame) -> dict:
    total = len(audit_df)
    return {
        "n_rows": total,
        "consistent": int((audit_df["audit_category"] == "consistent").sum()),
        "code_despite_non_implementation_prompt": int(
            (audit_df["audit_category"] == "code_despite_non_implementation_prompt").sum()
        ),
        "no_code_despite_implementation_prompt": int(
            (audit_df["audit_category"] == "no_code_despite_implementation_prompt").sum()
        ),
        "pct_misleading_code_label": round(
            (audit_df["audit_category"] == "code_despite_non_implementation_prompt").mean() * 100,
            2,
        ),
        "pct_missed_code": round(
            (audit_df["audit_category"] == "no_code_despite_implementation_prompt").mean() * 100,
            2,
        ),
    }


def save_label_audit(df: pd.DataFrame, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_df = audit_labels(df)
    summary = summarize_label_audit(audit_df)
    audit_df.to_csv(output_dir / "label_audit_examples.csv", index=False, encoding="utf-8-sig")
    misleading = audit_df[audit_df["audit_category"] != "consistent"].head(50)
    misleading.to_csv(output_dir / "label_audit_misleading_sample.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([summary]).to_csv(output_dir / "label_audit_summary.csv", index=False, encoding="utf-8-sig")
    return summary
