"""SpecStory discovery queries — maximize repos with `.specstory/history` logs."""

from __future__ import annotations

# Repo search: README mentions, topics, and path hints (no star floor — more hits).
SPECSTORY_REPO_QUERIES = [
    ("specstory_readme", "specstory in:readme"),
    ("topic_specstory", "topic:specstory"),
    ("specstory_path", "path:.specstory"),
]

# Code search: exact SpecStory artifact paths (requires GITHUB_TOKEN / GH_TOKEN).
SPECSTORY_CODE_QUERIES = [
    ("specstory_history", "path:.specstory/history"),
    ("specstory_root", "path:.specstory"),
    ("specstory_md", "extension:md path:.specstory"),
]

REPO_DETECTION_QUERIES = SPECSTORY_REPO_QUERIES
CODE_SEARCH_QUERIES = SPECSTORY_CODE_QUERIES

# Backwards-compatible alias for older callers.
DETECTION_QUERIES = REPO_DETECTION_QUERIES
