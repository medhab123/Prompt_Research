"""Create an English-readable view CSV (original training file unchanged)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from github_repo_miner.translate_dataset import export_english_view_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate text columns to English for personal review (separate CSV)."
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        nargs="?",
        default=Path("outputs/datasets/specstory_prompts_clustered_enriched.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path (default: <input>_english_view.csv).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        help="Only translate the first N rows (for quick tests).",
    )
    parser.add_argument(
        "--no-excel-bom",
        action="store_true",
        help="Write UTF-8 without BOM (Excel may garble non-ASCII).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_csv.exists():
        raise SystemExit(f"Input not found: {args.input_csv}")

    out = args.output
    if out is None:
        out = args.input_csv.with_name(f"{args.input_csv.stem}_english_view.csv")

    print(f"Source (unchanged): {args.input_csv}")
    print(f"Writing English view: {out}")
    if args.max_rows:
        print(f"Limit: first {args.max_rows} rows only")

    result = export_english_view_csv(
        args.input_csv.resolve(),
        out.resolve(),
        excel_friendly=not args.no_excel_bom,
        max_rows=args.max_rows,
    )

    print(f"Rows exported: {result['rows']}")
    print(f"Rows with any translation: {result['rows_with_any_translation']}")
    for col, count in result["translated_cells"].items():
        print(f"  {col}: {count} cells translated")
    print("Done.")


if __name__ == "__main__":
    main()
