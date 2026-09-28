"""Predict the HUMAN labels from the prompt alone: the cleanest prediction target.

The zero-shot LLM beats trained classifiers when the target is labels produced
by the same model family, but not when the target is the rule labeler's labels.
Either the LLM is genuinely better at anticipating agent behavior, or it is
partly reproducing its own labeling function. The hand-labeled reference set
distinguishes the two: no labeler produced it.

Trained models are fitted on the rest of the corpus with every repository that
contributes a hand-labeled turn held out, so nothing leaks. Both predictors see
only prompt-side inputs.

Usage: python scripts/predict_human_labels.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

import github_repo_miner.ml_research.llm_labeling as ll
from github_repo_miner.ml_research.action_prediction import (
    PROMPT_METADATA_COLS,
    history_feature_matrix,
    load_action_dataset,
    prompt_keyword_rule_baseline,
)
from github_repo_miner.ml_research.representations import hstack_features, metadata_matrix

load_dotenv(ROOT / ".env")
CORPUS_LABELS = ROOT / "outputs/llm_labels_v4/llm_labels_full.csv"
EMB = ROOT / "outputs/ml_action_v4_llm/prompt_embeddings.npy"
GOLD_DIR = ROOT / "outputs/gold_v2"
EXCLUDE = {"G023", "G061"}  # read while drafting the definitions
LABELS = ["implement", "debug", "explain", "review", "tool_only", "other"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--provider", choices=["gemini", "groq"], default="gemini")
    p.add_argument("--model", default=None, help="Zero-shot model; default gemini-3.1-flash-lite.")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def zero_shot(rows: pd.DataFrame, cache: Path, provider: str, model: str | None) -> pd.Series:
    """Reuse run_zeroshot_prediction's instructions so the two runs are comparable."""
    from importlib import util
    spec = util.spec_from_file_location("zs", ROOT / "scripts/run_zeroshot_prediction.py")
    zs = util.module_from_spec(spec)
    sys.modules["zs"] = zs
    spec.loader.exec_module(zs)
    done = pd.read_csv(cache) if cache.exists() else pd.DataFrame(columns=["row", "pred"])
    todo = rows[~rows.index.isin(done["row"])]
    if len(todo):
        print(f"zero-shot: predicting {len(todo)} turns")
        if provider == "groq":  # free tier: a few calls a minute, so run serially
            preds = []
            for i, (_, r) in enumerate(todo.iterrows(), 1):
                preds.append(zs.predict_one(zs.build_prompt(r, True), provider, model))
                time.sleep(zs.GROQ_MIN_INTERVAL)
                if i % 20 == 0:
                    print(f"  {i}/{len(todo)}", flush=True)
        else:
            with ThreadPoolExecutor(6) as ex:
                preds = list(ex.map(lambda r: zs.predict_one(zs.build_prompt(r[1], True), provider, model),
                                    todo.iterrows()))
        done = pd.concat([done, pd.DataFrame({"row": todo.index, "pred": preds})], ignore_index=True)
        done.to_csv(cache, index=False)
    return done.set_index("row")["pred"]


def main() -> None:
    args = parse_args()
    df, _ = load_action_dataset(CORPUS_LABELS, label_col="llm_action_type")
    df = ll.attach_previous_turn(df)
    emb = np.load(EMB)
    meta = metadata_matrix(df, [c for c in PROMPT_METADATA_COLS if c in df.columns])
    x = hstack_features(emb, meta, history_feature_matrix(df))

    grades = pd.read_excel(GOLD_DIR / "grading_sheet.xlsx", sheet_name="Grade", usecols=["grade_id", "your_label"])
    grades["your_label"] = grades["your_label"].astype(str).str.strip().str.lower()
    grades = grades[grades["your_label"].isin(LABELS) & ~grades["grade_id"].isin(EXCLUDE)]
    idx = pd.read_csv(GOLD_DIR / "sample_index.csv").merge(grades, on="grade_id")
    gold_rows = idx["row_index"].to_numpy()
    y_gold = idx["your_label"].to_numpy()

    gold_repos = set(df.iloc[gold_rows]["full_name"])
    train_mask = ~df["full_name"].isin(gold_repos)
    train_idx = np.where(train_mask)[0]
    print(f"hand-labeled turns: {len(gold_rows)}; training rows: {len(train_idx)} "
          f"({df['full_name'].nunique() - len(gold_repos)} repositories, {len(gold_repos)} held out)")

    preds = {}
    for name, target in [("trained LR -> LLM labels", "llm_action_type"), ("trained LR -> rule labels", "action_type")]:
        y_tr = df[target].to_numpy()[train_idx]
        sc = StandardScaler(with_mean=False).fit(x[train_idx])
        clf = LogisticRegression(max_iter=1000, random_state=args.seed, n_jobs=-1).fit(sc.transform(x[train_idx]), y_tr)
        preds[name] = clf.predict(sc.transform(x[gold_rows]))
    preds["keyword rule on the prompt"] = np.array(
        prompt_keyword_rule_baseline(df.iloc[gold_rows]["prompt_text"].astype(str).tolist()))
    model_name = args.model or ("gemini-3.1-flash-lite" if args.provider == "gemini" else "openai/gpt-oss-120b")
    tag = model_name.replace("/", "_")
    if args.provider == "groq":  # keep caches for different reasoning budgets apart
        tag += "_effort-" + os.getenv("GROQ_REASONING_EFFORT", "low")
    zs = zero_shot(df.iloc[gold_rows], GOLD_DIR / f"zeroshot_human_predictions_{tag}.csv", args.provider, args.model)
    preds[f"zero-shot {model_name} (prompt only)"] = zs.reindex(gold_rows).to_numpy()

    rng = np.random.default_rng(args.seed)
    groups = df["full_name"].to_numpy()[gold_rows]
    uniq = np.unique(groups)
    by_group = {g: np.where(groups == g)[0] for g in uniq}

    def ci(fn):
        vals = []
        for _ in range(args.bootstrap):
            drawn = rng.choice(uniq, size=len(uniq), replace=True)
            i = np.concatenate([by_group[g] for g in drawn])
            vals.append(fn(i))
        return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

    table = []
    for name, p in preds.items():
        keep = pd.notna(p)
        macro = f1_score(y_gold[keep], p[keep], average="macro", zero_division=0)
        lo, hi = ci(lambda i, p=p: f1_score(y_gold[i], p[i], average="macro", zero_division=0)
                    if pd.notna(p[i]).all() else np.nan)
        table.append({"predictor": name, "n": int(keep.sum()), "macro_f1": round(macro, 4),
                      "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                      "accuracy": round(float((p[keep] == y_gold[keep]).mean()), 4)})
    summary = pd.DataFrame(table).sort_values("macro_f1", ascending=False)

    md = ["# Predicting the human labels from the prompt alone\n",
          f"{len(gold_rows)} hand-labeled turns; trained models fitted on {len(train_idx)} turns from "
          f"repositories that contribute none of them. Cluster bootstrap over repositories "
          f"({args.bootstrap} resamples).\n",
          summary.to_markdown(index=False), "",
          f"Zero-shot model: `{model_name}` ({args.provider}), temperature 0.\n",
          "Note: labeling a turn (the agent's response is visible) and predicting it (only the prompt is "
          "visible) are different tasks; these numbers are the prediction task against human labels.\n"]
    out = args.output or (GOLD_DIR / f"PREDICT_HUMAN_{tag}.md")
    out.write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
