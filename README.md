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
python analyze_prompts.py outputs/datasets/specstory_prompts_extracted_20260626.csv
```

## Output

By default, timestamped CSV files are written to `outputs/` (or `outputs/datasets/` if you organize them):

- `specstory_candidates_<timestamp>.csv` — filtered repo list
- `specstory_candidates_raw_<timestamp>.csv` — all deduplicated discovery hits
- `specstory_prompts_<candidate_stem>_<timestamp>.csv` — extracted user prompts

Legacy filenames (`llm_repos_*`, `extracted_prompts_*`) are still supported as fallbacks.

If you run the script in Google Colab, it will also attempt to trigger a download of the filtered dataset.

## ML Research Pipeline

Transform the prompt dataset into supervised learning experiments (intent classification, behavior prediction, complexity regression):

```bash
pip install -r requirements-analysis.txt
python run_ml_research.py
```

Default input: `outputs/datasets/specstory_prompts_clustered.csv`

Options:

- `--output-dir outputs/ml` — artifacts directory (tables, figures, report)
- `--skip-code-embeddings` — skip CodeRankEmbed comparison (faster)
- `--device cpu|cuda` — embedding device
- `--min-intent-class-size 30` — drop rare intent classes

Outputs:

- `outputs/ml/ML_RESEARCH_REPORT.md` — research summary
- `outputs/ml/tables/` — CSV + markdown result tables
- `outputs/ml/figures/` — UMAP, confusion matrices, model comparisons
- `outputs/ml/prompt_embeddings.npy` — cached SBERT vectors

## Repository layout

```text
main.py                          # Entry shim → github_repo_miner.__main__
analyze_prompts.py               # Clean / embed / cluster CLI
run_ml_research.py               # Supervised ML research CLI
reexport_csv.py                  # Spreadsheet-safe CSV re-export utility
scripts/relocate_outputs.py      # One-time output organization helper

src/github_repo_miner/
  __main__.py                    # Official CLI (discover / extract / analyze)
  pipeline.py                    # GitHub repo discovery
  prompt_extraction.py           # Pull prompts from SpecStory logs
  specstory_parser.py            # Parse SpecStory markdown transcripts
  data_cleaning.py               # Noise filtering and deduplication
  prompt_analysis.py             # Embedding, clustering, descriptive stats
  csv_export.py                  # Blob offload for large CSV fields
  ml_research/                   # Supervised ML experiments

outputs/
  datasets/                      # Canonical CSV datasets
  analysis/                      # Descriptive analysis figures/tables
  ml/                            # ML experiment results

archive/                         # Old runs kept for reference (not used by pipeline)
notebooks/                       # Colab notebook (parallel workflow)
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
