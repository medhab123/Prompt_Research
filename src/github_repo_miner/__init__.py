"""GitHub repository mining pipeline."""

import sys as _sys

# GitHub content (commit messages, prompt text, READMEs) is routinely full of
# emoji/CJK/etc. On Windows, stdout/stderr default to the system codepage
# (e.g. cp1252) rather than UTF-8, so an unguarded print() of that content
# crashes any CLI entry point. Fix it once, here, for every entry point.
for _stream in (_sys.stdout, _sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from .pipeline import run_pipeline
from .prompt_analysis import AnalysisConfig, run_prompt_analysis

__all__ = ["run_pipeline", "AnalysisConfig", "run_prompt_analysis"]
