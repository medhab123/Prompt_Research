"""Safe CSV export for prompt datasets with large multiline fields."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pandas as pd

# Excel cell limit is 32,767 chars; stay below that for spreadsheet compatibility.
DEFAULT_MAX_INLINE_FIELD_CHARS = 30_000

BLOB_TEXT_COLUMNS = ("agent_response", "generated_code")
PREVIEW_SUFFIX = "_preview"


def _preview(text: str, max_chars: int) -> str:
    text = "" if pd.isna(text) else str(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20] + "\n...[truncated]"


def _blob_id(row: pd.Series, column: str) -> str:
    key = "|".join(
        str(row.get(col, ""))
        for col in ("full_name", "source_file", "turn_index", "prompt_text", column)
    )
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def export_prompts_csv(
    df: pd.DataFrame,
    output_path: str | Path,
    *,
    blob_dir: str | Path | None = None,
    max_inline_chars: int = DEFAULT_MAX_INLINE_FIELD_CHARS,
    prompts_only: bool = False,
) -> Path:
    """Write a spreadsheet-safe CSV and optionally store oversized text in sidecar files."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    export_df = df.copy()
    if prompts_only:
        drop_cols = set(BLOB_TEXT_COLUMNS) | {
            "agent_response_length",
            "generated_code_length",
            "agent_response_blob_path",
            "generated_code_blob_path",
            "agent_response_truncated",
            "generated_code_truncated",
        }
        keep = [c for c in export_df.columns if c not in drop_cols]
        export_df = export_df[keep]
    else:
        blob_root = Path(blob_dir) if blob_dir is not None else output_path.parent / f"{output_path.stem}_blobs"
        blob_root.mkdir(parents=True, exist_ok=True)

        for column in BLOB_TEXT_COLUMNS:
            if column not in export_df.columns:
                continue

            blob_paths: list[str] = []
            previews: list[str] = []
            truncated_flags: list[bool] = []

            for _, row in export_df.iterrows():
                value = "" if pd.isna(row[column]) else str(row[column])
                if len(value) > max_inline_chars:
                    blob_id = _blob_id(row, column)
                    blob_path = blob_root / f"{blob_id}_{column}.txt"
                    blob_path.write_text(value, encoding="utf-8")
                    blob_paths.append(str(blob_path))
                    previews.append(_preview(value, max_inline_chars))
                    truncated_flags.append(True)
                else:
                    blob_paths.append("")
                    previews.append(value)
                    truncated_flags.append(False)

            export_df[column] = previews
            export_df[f"{column}_blob_path"] = blob_paths
            export_df[f"{column}_truncated"] = truncated_flags

    export_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_NONNUMERIC,
        lineterminator="\n",
    )
    return output_path
