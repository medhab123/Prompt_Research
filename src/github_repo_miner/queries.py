"""Detection queries used to mine candidate repositories."""

from __future__ import annotations

# --- Full query sets (kept for later, not all active right now) ---

ALL_REPO_DETECTION_QUERIES = [
    # High-signal README mentions that are closer to actual prompt artifacts
    ("specstory_readme", "specstory in:readme stars:>0"),
    ("cursorrules_readme", '"cursorrules" in:readme stars:>0'),
    ("copilot_instructions_readme", '"copilot instructions" in:readme stars:>0'),
    ("cursor_rules_readme", '"cursor rules" in:readme stars:>0'),
    ("prompt_log_readme", '"prompt log" in:readme stars:>0'),

    # Community tags and broader discovery terms tied to prompt/instruction repos
    ("topic_cursor", "topic:cursor"),
    ("topic_specstory", "topic:specstory"),
    ("topic_prompt_engineering", "topic:prompt-engineering"),
    ("topic_cursorrules", "topic:cursorrules"),
]

ALL_CODE_SEARCH_QUERIES = [
    # Exact artifact names that are more likely to appear in code search than repo search
    ("specstory_history", "path:.specstory/history"),
    ("prompt_comment_py", '"# Prompt:" language:Python'),
    ("prompt_comment_js", '"// Prompt:" language:JavaScript'),
    ("copilot_instructions", "filename:copilot-instructions.md"),
    ("cursor_rules", "path:.cursor/rules"),
    ("cursorrules_legacy", "filename:.cursorrules"),
    ("github_copilot_instructions", "path:.github/copilot-instructions.md"),
    ("claude_md", "filename:CLAUDE.md"),
    ("agents_md", "filename:AGENTS.md"),
]

# --- SpecStory-only query sets (active) ---
# We are intentionally focusing on SpecStory artifacts only for now, since those
# contain real session-level developer prompts (highest research value).

SPECSTORY_REPO_QUERIES = [
    ("specstory_readme", "specstory in:readme stars:>0"),
    ("topic_specstory", "topic:specstory"),
]

SPECSTORY_CODE_QUERIES = [
    ("specstory_history", "path:.specstory/history"),
]

# Active query sets consumed by the pipeline.
REPO_DETECTION_QUERIES = SPECSTORY_REPO_QUERIES
CODE_SEARCH_QUERIES = SPECSTORY_CODE_QUERIES

# Backwards-compatible alias for older callers.
DETECTION_QUERIES = REPO_DETECTION_QUERIES
