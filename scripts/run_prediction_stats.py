"""Statistical rigor for the next-action prediction results.

Adds three things the main prediction run does not provide:
  1. Repeated repository-held-out CV (R repeats x K folds, repository->fold
     assignment reshuffled each repeat), so results do not hinge on one
     partition. GroupKFold is deterministic, so repeats are produced by
     shuffling repositories before assigning them to folds.
  2. Bootstrap confidence intervals for macro-F1, resampling *repositories*
     (a cluster bootstrap), since turns within a repository are not
     independent.
  3. Paired comparisons between representations across all repeats, reported
     as a mean difference with a bootstrap CI and the share of splits in which
     the second representation wins.

Embeddings are read from the cache written by run_action_prediction.py, so
nothing is re-encoded.

Usage:
  python scripts/run_prediction_stats.py outputs/llm_labels_v4/llm_labels_full.csv \
      --embeddings-dir outputs/ml_action_v4_llm --output-dir outputs/ml_action_v4_llm/stats \
      --label-col llm_action_type --repo-context-csv outputs/full_run_v3_analysis/repo_context.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from github_repo_miner.ml_research.action_prediction import (
    PROMPT_METADATA_COLS,
    history_feature_matrix,
    load_action_dataset,
    load_repo_context,
    prev_action_only_matrix,
    repo_context_matrix,
    top_languages,
)
from github_repo_miner.ml_research.representations import hstack_features, metadata_matrix


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input_csv", type=Path)
    p.add_argument("--embeddings-dir", type=Path, required=True, help="Directory holding prompt_embeddings.npy.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--label-col", default=None)
    p.add_argument("--repo-context-csv", type=Path, default=None)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def shuffled_group_folds(groups: np.ndarray, n_folds: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Assign repositories to folds at random, balancing fold sizes by turn count."""
    uniq, counts = np.unique(groups, return_counts=True)
    order = rng.permutation(len(uniq))
    load = np.zeros(n_folds)
    fold_of = {}
    for i in order:  # greedy: give each repository to the currently smallest fold
        f = int(np.argmin(load))
        fold_of[uniq[i]] = f
        load[f] += counts[i]
    assign = np.array([fold_of[g] for g in groups])
    return [np.where(assign == f)[0] for f in range(n_folds)]


def fit_predict(x_tr, y_tr, x_te, seed: int) -> np.ndarray:
    scaler = StandardScaler(with_mean=False).fit(x_tr)
    clf = LogisticRegression(max_iter=1000, random_state=seed, n_jobs=-1)
    clf.fit(scaler.transform(x_tr), y_tr)
    return clf.predict(scaler.transform(x_te))


def cluster_bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray,
                         n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    """Percentile CI for macro-F1, resampling repositories with replacement."""
    uniq = np.unique(groups)
    idx_by_group = {g: np.where(groups == g)[0] for g in uniq}
    scores = []
    for _ in range(n_boot):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in drawn])
        scores.append(f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df, label_col = load_action_dataset(args.input_csv, label_col=args.label_col)
    df = load_repo_context(df, args.repo_context_csv)
    y = df[label_col].to_numpy()
    groups = df["full_name"].to_numpy()
    emb = np.load(args.embeddings_dir / "prompt_embeddings.npy")
    if len(emb) != len(df):
        raise SystemExit(f"Embedding cache has {len(emb)} rows but the dataset has {len(df)}.")

    # Only the prompt-derived columns present in this file (same list the main run uses).
    meta = metadata_matrix(df, [c for c in PROMPT_METADATA_COLS if c in df.columns])
    hist = history_feature_matrix(df)
    reps = {
        "sbert_prompt_only": emb,
        "sbert+history": hstack_features(emb, hist),
        "sbert+prompt_metadata+history": hstack_features(emb, meta, hist),
        "sbert+prev_action_only": hstack_features(emb, prev_action_only_matrix(df)),
    }
    if "repo_language" in df.columns:
        reps["sbert+prompt_metadata+history+repo_context"] = hstack_features(
            emb, meta, hist, repo_context_matrix(df, top_languages(df)))

    rng = np.random.default_rng(args.seed)
    rows, preds_by_rep = [], {r: {} for r in reps}
    for repeat in range(args.repeats):
        folds = shuffled_group_folds(groups, args.folds, rng)
        for fold_i, test_idx in enumerate(folds):
            train_idx = np.setdiff1d(np.arange(len(df)), test_idx)
            for name, x in reps.items():
                pred = fit_predict(x[train_idx], y[train_idx], x[test_idx], args.seed)
                rows.append({"representation": name, "repeat": repeat, "fold": fold_i,
                             "macro_f1": f1_score(y[test_idx], pred, average="macro", zero_division=0),
                             "accuracy": float((pred == y[test_idx]).mean()), "n_test": len(test_idx)})
                if repeat == 0:  # keep repeat 0 predictions for the bootstrap (every turn appears once)
                    preds_by_rep[name][fold_i] = (test_idx, pred)
        print(f"repeat {repeat + 1}/{args.repeats} done", flush=True)

    per_fold = pd.DataFrame(rows)
    per_fold.to_csv(args.output_dir / "repeated_cv_per_fold.csv", index=False)

    summary = (per_fold.groupby("representation")
               .agg(n_splits=("macro_f1", "size"), macro_f1_mean=("macro_f1", "mean"),
                    macro_f1_std=("macro_f1", "std"), accuracy_mean=("accuracy", "mean"))
               .reset_index().sort_values("macro_f1_mean", ascending=False))
    # Bootstrap CI over out-of-fold predictions from the first repeat.
    cis = {}
    for name, folds_preds in preds_by_rep.items():
        idx = np.concatenate([t for t, _ in folds_preds.values()])
        pred = np.concatenate([p for _, p in folds_preds.values()])
        cis[name] = cluster_bootstrap_ci(y[idx], pred, groups[idx], args.bootstrap, rng)
    summary["oof_macro_f1"] = [f1_score(y[np.concatenate([t for t, _ in preds_by_rep[r].values()])],
                                        np.concatenate([p for _, p in preds_by_rep[r].values()]),
                                        average="macro", zero_division=0) for r in summary["representation"]]
    summary["bootstrap_ci_low"] = [cis[r][0] for r in summary["representation"]]
    summary["bootstrap_ci_high"] = [cis[r][1] for r in summary["representation"]]
    summary.to_csv(args.output_dir / "repeated_cv_summary.csv", index=False)

    # Paired comparisons across all repeats x folds.
    pivot = per_fold.pivot_table(index=["repeat", "fold"], columns="representation", values="macro_f1")
    comps = [("sbert_prompt_only", "sbert+history"),
             ("sbert_prompt_only", "sbert+prompt_metadata+history"),
             ("sbert+prev_action_only", "sbert+prompt_metadata+history")]
    comp_rows = []
    for a, b in comps:
        if a not in pivot or b not in pivot:
            continue
        d = (pivot[b] - pivot[a]).to_numpy()
        boot = [np.mean(rng.choice(d, size=len(d), replace=True)) for _ in range(args.bootstrap)]
        comp_rows.append({"rep_a": a, "rep_b": b, "n_splits": len(d), "mean_diff": d.mean(),
                          "ci_low": float(np.percentile(boot, 2.5)), "ci_high": float(np.percentile(boot, 97.5)),
                          "b_wins_share": float((d > 0).mean())})
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(args.output_dir / "repeated_cv_paired.csv", index=False)

    md = ["# Repeated CV and bootstrap intervals\n",
          f"Target: `{label_col}` | {args.repeats} repeats x {args.folds} repository-held-out folds "
          f"| logistic regression | cluster bootstrap over repositories ({args.bootstrap} resamples)\n",
          summary.round(4).to_markdown(index=False), "",
          "## Paired differences across all splits\n", comp.round(4).to_markdown(index=False), ""]
    (args.output_dir / "STATS.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
