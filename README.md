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
main.py                  # Entry shim → github_repo_miner.__main__
requirements.txt
README.md
RESEARCH_PROPOSAL.md      # Current research write-up
PROJECT_STATUS_UPDATE.md  # Current project status log
docs/                      # Older planning/pipeline-explanation docs
scripts/                   # All other CLI entry points (see below)
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

After running the miner, extract prompts from discovered repos:

```bash
python -m github_repo_miner --extract-only
```

Or run discovery + extraction together (default):

```bash
python main.py
```

Modes:

| Flag | What it does |
|------|----------------|
| (default) | Discover SpecStory repos, then extract prompts |
| `--discover-only` | GitHub search only |
| `--extract-only` | Extract from latest `outputs/datasets/specstory_candidates_*.csv` or legacy `llm_repos_candidate_dataset_*.csv` in `outputs/` |
| `--analyze-csv PATH` | Clean, embed, cluster an existing prompt CSV |

Analyze or cluster an existing dataset:

```bash
python scripts/analyze_prompts.py outputs/full_run_v3_analysis/specstory_prompts_cleaned.csv
```

## Output

By default, timestamped CSV files are written to `outputs/` (or `outputs/datasets/` if you organize them):

- `specstory_candidates_<timestamp>.csv` — filtered repo list
- `specstory_candidates_raw_<timestamp>.csv` — all deduplicated discovery hits
- `specstory_prompts_<candidate_stem>_<timestamp>.csv` — extracted user prompts

Legacy filenames (`llm_repos_*`, `extracted_prompts_*`) are still supported as fallbacks.

If you run the script in Google Colab, it will also attempt to trigger a download of the filtered dataset.

## Action-Type Prediction Pipeline (current research)

The active research target is `action_type` — predicting whether a coding
agent's next turn will be `implement`/`debug`/`explain`/`review`/`tool_only`/
`other`. See `RESEARCH_PROPOSAL.md` for the full write-up. This supersedes
the older `primary_intent`-based `run_ml_research.py` pipeline (still present
in `scripts/` but not the current research target — see `archive/README.md`
for why it was superseded).

```bash
pip install -r requirements-analysis.txt

# 1. LLM-assisted relabeling of the corpus (rule-based `action_type` -> validated `llm_action_type`)
python scripts/run_llm_labeling.py outputs/full_run_v3_analysis/specstory_prompts_cleaned.csv \
    --output-dir outputs/llm_labels_v3 --full --provider gemini

# 2. Predict action_type from prompt + session history (repo-held-out evaluation)
python scripts/run_action_prediction.py outputs/llm_labels_v3/llm_labels_full.csv \
    --output-dir outputs/ml_action_final \
    --repo-context-csv outputs/full_run_v3_analysis/repo_context.csv \
    --label-col llm_action_type
```

Key options for `run_action_prediction.py`:

- `--label-col` — defaults to `llm_action_type` when present, else the rule-based `action_type`
- `--repo-context-csv` — adds repository-level metadata features (language, stars, size)
- `--cv-folds` — repository-held-out `GroupKFold` folds for statistical rigor (default 5)
- `--max-context-turns` — sweep of N previous turns for the context-window experiment

Outputs (see `outputs/README.md` for the full breakdown):

- `outputs/llm_labels_v3/llm_labels_full.csv` — the validated label set
- `outputs/ml_action_final/ACTION_PREDICTION_REPORT.md` — full results report
- `outputs/ml_action_final/tables/` — every results table, including per-fold CV data and paired significance tests

## Repository layout

```text
main.py                          # Entry shim → github_repo_miner.__main__

scripts/
  analyze_prompts.py             # Clean / embed / cluster CLI
  run_llm_labeling.py            # LLM-assisted action_type relabeling (current)
  run_action_prediction.py       # action_type prediction + evaluation (current)
  run_ml_research.py             # Older primary_intent ML pipeline (superseded)
  run_enrich_dataset.py
  run_robustness_experiments.py
  run_translate_dataset.py
  reexport_csv.py                # Spreadsheet-safe CSV re-export utility
  relocate_outputs.py            # One-time output organization helper

src/github_repo_miner/
  __main__.py                    # Official CLI (discover / extract / analyze)
  pipeline.py                    # GitHub repo discovery
  prompt_extraction.py           # Pull prompts from SpecStory logs
  specstory_parser.py            # Parse SpecStory markdown transcripts
  data_cleaning.py               # Noise filtering and deduplication
  prompt_analysis.py             # Embedding, clustering, descriptive stats
  action_labeling.py             # Rule-based action_type weak supervision
  repo_context.py                # Repository-level metadata fetcher
  csv_export.py                  # Blob offload for large CSV fields
  ml_research/                   # Supervised ML experiments, incl. action_prediction.py

outputs/                         # Current pipeline outputs — see outputs/README.md
docs/                            # Older planning/pipeline-explanation docs
archive/                         # Superseded runs kept for reference — see archive/README.md
```

## Command-line options (mining)

```bash
python -m github_repo_miner --output-dir outputs --rate-limit-delay 7.5
```

- `--output-dir` changes where CSV files are written.
- `--rate-limit-delay` changes the pause between search requests.

## Notes

- Repository search works unauthenticated, but code search needs a token.
- If no token is set, the miner skips code search and still returns repo-search candidates.
- The detection queries are defined in `src/github_repo_miner/queries.py`.
