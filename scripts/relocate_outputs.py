"""One-time helper: organize outputs/ and fix blob path references."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
DATASETS = OUTPUTS / "datasets"
ARCHIVE = ROOT / "archive"


def _move(src: Path, dest: Path) -> None:
    if not src.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    shutil.move(str(src), str(dest))
    print(f"Moved: {src.name} -> {dest.relative_to(ROOT)}")


def relocate_clustered_dataset() -> None:
    """Move clustered spreadsheet + blobs to datasets/ with clean names."""
    old_csv = OUTPUTS / "specstory_prompts_clustered (3)_spreadsheet.csv"
    old_blobs = OUTPUTS / "specstory_prompts_clustered (3)_spreadsheet_blobs"
    new_csv = DATASETS / "specstory_prompts_clustered.csv"
    new_blobs = DATASETS / "specstory_prompts_clustered_blobs"

    if new_csv.exists():
        print(f"Already relocated: {new_csv}")
        return
    if not old_csv.exists():
        print("No clustered spreadsheet to relocate.")
        return

    DATASETS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(old_csv)

    if old_blobs.exists():
        shutil.move(str(old_blobs), str(new_blobs))
        print(f"Moved blobs -> {new_blobs.relative_to(ROOT)}")

    for col in ("agent_response_blob_path", "generated_code_blob_path"):
        if col in df.columns:
            df[col] = df[col].apply(
                lambda p: str(DATASETS / "specstory_prompts_clustered_blobs" / Path(str(p)).name)
                if pd.notna(p) and str(p).strip()
                else p
            )

    df.to_csv(new_csv, index=False)
    print(f"Wrote {new_csv.relative_to(ROOT)} ({len(df)} rows)")
    _move(old_csv, ARCHIVE / "exports" / old_csv.name)


def main() -> None:
    # Archive old mining runs (keep latest 0057)
    mining = ARCHIVE / "mining"
    for f in OUTPUTS.glob("llm_repos_*"):
        if "20260626_0057" in f.name:
            continue
        _move(f, mining / f.name)

    # Keep + rename latest mining
    _move(
        OUTPUTS / "llm_repos_candidate_dataset_20260626_0057.csv",
        DATASETS / "specstory_candidates_20260626.csv",
    )
    _move(
        OUTPUTS / "llm_repos_raw_20260626_0057.csv",
        DATASETS / "specstory_candidates_raw_20260626.csv",
    )

    # Archive old extractions (keep largest 0610)
    extraction = ARCHIVE / "extraction"
    for f in OUTPUTS.glob("extracted_prompts_*"):
        if "20260626_0610" in f.name:
            _move(f, DATASETS / "specstory_prompts_extracted_20260626.csv")
        else:
            _move(f, extraction / f.name)

    # Archive intermediate cleaning outputs
    cleaning = ARCHIVE / "cleaning"
    for name in (
        "specstory_prompts_cleaned.csv",
        "specstory_prompts_cleaned_v2.csv",
        "specstory_prompts_dropped.csv",
        "specstory_prompts_dropped_v2.csv",
    ):
        _move(OUTPUTS / name, cleaning / name)

    # Archive slim export + debug
    _move(
        OUTPUTS / "specstory_prompts_clustered (3)_prompts_only.csv",
        ARCHIVE / "exports" / "specstory_prompts_clustered_prompts_only.csv",
    )
    _move(OUTPUTS / "_trace_source.md", ARCHIVE / "debug" / "_trace_source.md")
    _move(OUTPUTS / "_roundtrip_test.csv", ARCHIVE / "debug" / "_roundtrip_test.csv")

    # Remove empty folder if present
    inst = OUTPUTS / "Instruction Files"
    if inst.exists() and not any(inst.iterdir()):
        inst.rmdir()
        print("Removed empty outputs/Instruction Files/")

    relocate_clustered_dataset()
    print("\nDone. See outputs/README.md and archive/README.md")


if __name__ == "__main__":
    main()
