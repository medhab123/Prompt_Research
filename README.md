# GitHub Repo Miner

A small Python project that mines GitHub repositories for signals of LLM-assisted development, deduplicates the results, applies quality filters, spot-checks commit history, and exports CSV datasets.

## What it does

The pipeline keeps the same core behavior as the original notebook export, but it now uses two discovery channels:

- searches GitHub repos with several detection queries
- optionally runs GitHub code search for exact prompt-artifact files when `GITHUB_TOKEN` or `GH_TOKEN` is set
- merges repeated matches into one row per repository
- filters for higher-signal candidates
- prints a summary and a top-candidate table
- checks recent commit messages for LLM-related keywords
- saves raw and filtered CSV outputs

## Project layout

```text
main.py
requirements.txt
README.md
src/github_repo_miner/
  __init__.py
  __main__.py
  config.py
  github_api.py
  pipeline.py
  queries.py
```

## Setup

1. Create and activate a virtual environment.
2. Install the project in editable mode:

```bash
pip install -e .
```

If you prefer a plain dependency file, `requirements.txt` contains the same runtime packages.

## Run

From the project root, run either of these:

```bash
python main.py
```

or

```bash
python -m github_repo_miner
```

You can also use the installed console script:

```bash
github-repo-miner
```

If you want the miner to search for exact artifact files such as `.specstory/history` or `copilot-instructions.md`, set a GitHub token first. On PowerShell:

```powershell
$env:GITHUB_TOKEN = "your_token_here"
```

`GH_TOKEN` works too.

## Extract Prompts

After running the miner, you can extract prompt-like text from the candidate repos into a new CSV:

```bash
python -m github_repo_miner --extract-prompts --candidate-csv outputs/llm_repos_candidate_dataset_<timestamp>.csv
```

If you omit `--candidate-csv`, the tool will use the newest `llm_repos_candidate_dataset_*.csv` in `outputs/`.

The extractor labels each row with an `artifact_type` and `research_priority`:

| Priority | `artifact_type` | Source |
| --- | --- | --- |
| 1 | `session_prompt` | SpecStory user turns in `.specstory/history/*` |
| 2 | `inline_prompt_comment` | `# Prompt:` / `// Prompt:` comments in source files |
| 3 | `instruction_file` | `CLAUDE.md`, `AGENTS.md`, `copilot-instructions.md` |
| 4 | `rules_config` | `.cursor/rules/*`, `.cursorrules` |

To keep only the highest-value artifacts:

```bash
python main.py --extract-prompts --max-priority 2
```

Or keep specific types:

```bash
python main.py --extract-prompts --artifact-types session_prompt,inline_prompt_comment
```

## Output

By default, the project writes timestamped CSV files to `outputs/`:

- `llm_repos_candidate_dataset_<timestamp>.csv`
- `llm_repos_raw_<timestamp>.csv`
- `extracted_prompts_<candidate_file>_<timestamp>.csv`

If you run the script in Google Colab, it will also attempt to trigger a download of the filtered dataset.

## Command-line options

```bash
python -m github_repo_miner --output-dir outputs --rate-limit-delay 7.5
```

- `--output-dir` changes where CSV files are written.
- `--rate-limit-delay` changes the pause between search requests.

## Notes

- Repository search works unauthenticated, but code search needs a token.
- If no token is set, the miner skips code search and still returns repo-search candidates.
- The detection queries are defined in `src/github_repo_miner/queries.py`.
