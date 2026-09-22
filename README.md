# Prompt Research: what do AI coding agents actually do?

Many developers use AI coding agents (Cursor, Claude Code, Copilot, ...). Some of them
commit the agent's chat history into their GitHub repositories, using a tool called
[SpecStory](https://specstory.com) that saves each conversation to `.specstory/history/`.

This project:

1. **Finds** public GitHub repos that contain those saved conversations.
2. **Extracts** every turn: the developer's message and the agent's reply.
3. **Cleans** the turns (drops noise, translates non-English prompts, caps each repo at 50 turns).
4. **Labels** what the agent did in each turn, first with keyword rules and then with an LLM (Gemini or Groq).
   There are six labels:

   | label | the agent… |
   |---|---|
   | `implement` | wrote or changed code, files or project state |
   | `debug` | worked on a problem the developer reported (error, crash, wrong behavior) |
   | `explain` | answered a question, explained something, or gave a plan |
   | `review` | checked existing code or project status and reported what it found |
   | `tool_only` | only ran tools (search, read files) and stopped before giving any result |
   | `other` | did nothing codable: greetings, empty replies, rate-limit messages |

   The full rules the LLM follows are in
   [`src/github_repo_miner/ml_research/prompts/action_type_codebook.md`](src/github_repo_miner/ml_research/prompts/action_type_codebook.md).
5. **Predicts** the agent's next action from the developer's prompt and the earlier turns,
   testing on repositories the model never saw during training.

---

## 1. What you need

| | |
|---|---|
| **Python 3.10 or newer** | Check with `python --version`. Download from [python.org](https://www.python.org/downloads/) if needed. On Windows, tick "Add python.exe to PATH" during install. |
| **Git** | To download this repo. [git-scm.com](https://git-scm.com/downloads) |
| **A GitHub token** (required) | Free. GitHub → Settings → Developer settings → Personal access tokens → *Fine-grained tokens* → Generate. No extra permissions are needed (public repos only). Without it GitHub allows only 60 requests/hour and the miner will stall. |
| **A Gemini API key** (only for step 4) | Free tier works for small runs. Get one at [aistudio.google.com/apikey](https://aistudio.google.com/apikey). A Groq key ([console.groq.com](https://console.groq.com/keys)) works instead. |
| **A few GB of free disk** | Mostly PyTorch (larger on Linux, where the default PyTorch includes GPU support). |

---

## 2. Setup (one time)

Open a terminal (on Windows: **PowerShell**), then run these one at a time.

**Download the code and go into the folder:**

```bash
git clone https://github.com/medhab123/Prompt_Research.git
```

```bash
cd Prompt_Research
```

**Create a virtual environment** (a private Python install just for this project, inside a folder called `.venv`):

```bash
python -m venv .venv
```

**Turn it on.** You need to do this every time you open a new terminal.

- Windows (PowerShell): `.venv\Scripts\Activate.ps1`
- macOS / Linux: `source .venv/bin/activate`

Your prompt should now start with `(.venv)`.

> **Windows: "running scripts is disabled on this system"?** Skip activation and type
> `.venv\Scripts\python` everywhere this README says `python`.

**Install everything** (takes a few minutes; PyTorch is the big download):

```bash
python -m pip install --upgrade pip
```

```bash
pip install -r requirements.txt
```

**Add your keys.** Copy the example settings file:

- Windows: `copy .env.example .env`
- macOS / Linux: `cp .env.example .env`

Open `.env` in any text editor and fill in your keys (no quotes, no spaces):

```
GITHUB_TOKEN=github_pat_xxxxxxxxxxxx
GEMINI_API_KEY=AIzaxxxxxxxxxxxx
GROQ_API_KEY=
```

`.env` is ignored by git, so your keys will not be committed.

**Check the install worked:**

```bash
python -m github_repo_miner --help
```

You should see a list of options. If you see `No module named github_repo_miner`, the virtual environment is not active (see above).

---

## 3. Quick test run (about 15 minutes)

This runs the whole pipeline on **8 repositories**, so you can see every step work before
committing to a full run. All results go into `outputs/test_run/`. Run the steps in order
from the project folder.

### Step 1: find repos and extract conversations (~5 min)

```bash
python -m github_repo_miner --output-dir outputs/test_run --max-results-per-query 20 --max-repos 8
```

The last line printed looks like:

```
Saved: outputs\test_run\datasets\specstory_turn_pairs_specstory_candidates_20260917_0807_20260917_0810.csv
```

**Copy that path.** Your file name will have different dates/times. You need it for step 2.

### Step 2: clean the turns (~3 min; the first run also downloads a ~90 MB model)

Replace `PASTE_PATH_FROM_STEP_1` with the path you copied:

```bash
python scripts/analyze_prompts.py PASTE_PATH_FROM_STEP_1 --output-dir outputs/test_run/corpus --max-prompts-per-repo 50
```

This writes `outputs/test_run/corpus/specstory_prompts_cleaned.csv`, the dataset every
later step uses. It already includes the rule-based label in a column called `action_type`.

### Step 3: fetch repository info (a few seconds)

```bash
python scripts/fetch_repo_context.py outputs/test_run/corpus/specstory_prompts_cleaned.csv --output outputs/test_run/corpus/repo_context.csv
```

### Step 4: label turns with an LLM (~2 min, uses ~30 API calls)

```bash
python scripts/run_llm_labeling.py outputs/test_run/corpus/specstory_prompts_cleaned.csv --output-dir outputs/test_run/llm_labels --sample-size 30
```

It prints how often the LLM agrees with the rule labels (Cohen's kappa) and writes
`outputs/test_run/llm_labels/llm_labels_sample.csv`.

### Step 5: train and evaluate the prediction models (~3 min)

```bash
python scripts/run_action_prediction.py outputs/test_run/corpus/specstory_prompts_cleaned.csv --output-dir outputs/test_run/ml_action --repo-context-csv outputs/test_run/corpus/repo_context.csv --label-col action_type
```

Open **`outputs/test_run/ml_action/ACTION_PREDICTION_REPORT.md`** to read the results.
With only 8 repos the numbers are meaningless. This run just shows that everything works.

---

## 4. Full run (hours)

Same five steps with the limits removed. Use a new output folder, e.g. `outputs/full_run`.

| Step | Change from the test run |
|---|---|
| 1 | Drop `--max-results-per-query 20 --max-repos 8`. This searches all of GitHub and downloads thousands of conversation files, which takes **several hours**. It saves a checkpoint every 20 repos. |
| 2 | Same command, with the new path. |
| 3 | Same command. |
| 4 | Replace `--sample-size 30` with `--full` to label every turn. That is one API call per turn (~8,400 for our corpus), so check your quota first. If it stops (quota, network), run the same command again: finished rows are cached in `llm_label_cache.json` and skipped. |
| 5 | Give it the **labeled** file from step 4 and use the LLM labels: `python scripts/run_action_prediction.py outputs/full_run/llm_labels/llm_labels_full.csv --output-dir outputs/full_run/ml_action --repo-context-csv outputs/full_run/corpus/repo_context.csv --label-col llm_action_type` |

GitHub search results change over time, so a new full run will not find exactly the same
repositories as ours (8,404 turns from 251 repos, mined August 2026).

---

## 5. What's in this repository

```text
README.md                  you are here
pyproject.toml             package definition + dependency list
requirements.txt           installs the package (same as pip install -e .)
.env.example               template for your API keys
main.py                    shortcut: python main.py == python -m github_repo_miner

src/github_repo_miner/     the library
  __main__.py              step 1 command line (find repos, extract turns)
  pipeline.py              GitHub repo search
  queries.py               the search queries used to find SpecStory repos
  github_api.py            GitHub API helpers
  prompt_extraction.py     download and split conversation files into turns
  specstory_parser.py      parse SpecStory markdown
  action_labeling.py       rule-based action_type labels
  session_features.py      features from earlier turns (previous action, etc.)
  data_cleaning.py         noise filters, per-repo cap
  translate_dataset.py     translation of non-English prompts
  prompt_analysis.py       step 2: cleaning, embeddings, clustering, descriptive report
  repo_context.py          step 3: repo metadata from GitHub
  csv_export.py            spreadsheet-safe CSV export
  ml_research/
    llm_labeling.py        step 4: LLM labeling (Gemini / Groq), caching, kappa
    prompts/               the labeling codebook sent to the LLM
    action_prediction.py   step 5: features, models, repo-held-out CV, significance tests
    models.py, splits.py, representations.py, evaluate.py, report.py   (used by step 5)
    pipeline.py, robustness.py, baselines.py, label_audit.py, data.py, config.py   (legacy, see below)

scripts/                   command-line entry points
  analyze_prompts.py         step 2
  fetch_repo_context.py      step 3
  run_llm_labeling.py        step 4
  run_action_prediction.py   step 5
  run_enrich_dataset.py      re-add action_type + session features to an older turn CSV
  run_translate_dataset.py   make an English-readable copy of a dataset for reading
  run_robustness_experiments.py   legacy: baselines/keyword-ablation for the old label
  run_ml_research.py         legacy: the earlier "primary_intent" pipeline
  reexport_csv.py            re-save a CSV so Excel opens it cleanly

paper/                     paper drafts (LaTeX), figure script, and number provenance
  NUMBERS.md               where every number in the paper comes from
  figures/make_figures.py  regenerates the paper figures from outputs/

docs/history/              earlier proposals and status notes (kept for the record)

outputs/                   NOT in git; everything the pipeline writes goes here
```

**Legacy code.** The project first tried to predict a label called `primary_intent`.
`run_ml_research.py`, `run_robustness_experiments.py` and the `ml_research` modules marked
legacy belong to that earlier phase. They still run but are not part of the current results.

**Data is not included.** `outputs/` is in `.gitignore`. The mined conversations are
other people's repository content, and the files are large (the labeled corpus alone is ~110 MB).
Running the pipeline recreates them.

---

## 6. Reproducing the paper's numbers

The paper's tables and figures come from these files, produced by the full run:

| File | What |
|---|---|
| `outputs/full_run_v3_analysis/specstory_prompts_cleaned.csv` | the 8,404-turn corpus (step 2) |
| `outputs/full_run_v3_analysis/repo_context.csv` | repo metadata (step 3) |
| `outputs/llm_labels_v3/llm_labels_full.csv` | LLM labels for every turn (step 4, `--full`) |
| `outputs/ml_action_final/` | prediction results on LLM labels (step 5) |
| `outputs/ml_action_rule_labels_cv/` | the same models on rule labels, for comparison |
| `outputs/provenance/` | earlier runs cited for before/after comparisons |

With those files in place:

```bash
python paper/figures/make_figures.py
```

`paper/NUMBERS.md` maps every number in the paper to the file and column it came from.

Note: the labels in `llm_labels_full.csv` were made with an earlier prompt
(`LEGACY_SYSTEM_INSTRUCTIONS` in `llm_labeling.py`, version `v2-taxonomy-fix`). New
labeling runs use the codebook in `ml_research/prompts/` (version `v3-codebook`), so a
fresh step 4 will not give exactly the same labels.

---

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `No module named github_repo_miner` | The virtual environment isn't active. Activate it (section 2) or use `.venv\Scripts\python` / `.venv/bin/python`. |
| Windows: `[WinError 206] The filename or extension is too long` during install | Your project folder path is too long. Move the project somewhere shorter (e.g. `C:\code\Prompt_Research`), delete `.venv`, and redo setup. |
| Step 1 is very slow or prints `403` / rate-limit messages | `GITHUB_TOKEN` is missing or wrong in `.env`. The miner waits and retries automatically, so a correct token fixes it. |
| Step 4: `GEMINI_API_KEY (or GOOGLE_API_KEY) not set` | Put your key in `.env` and run the command from the project folder. |
| Step 4: `429 RESOURCE_EXHAUSTED` | Gemini quota hit. Wait and re-run (finished rows are cached), add `--max-workers 2 --rate-limit-delay 6`, or use `--provider groq`. |
| Step 4: `404` model not found | Google retired the model. Pass a current one, e.g. `--model gemini-3.1-flash-lite`. |
| `Warning: You are sending unauthenticated requests to the HF Hub` | Harmless. It is only a download-speed notice for the embedding model. |
| Step 5 scores on the test run look random | Expected. 8 repos is far too little data. The test run only checks that the code works. |
