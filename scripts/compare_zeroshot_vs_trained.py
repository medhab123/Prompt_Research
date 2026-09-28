"""Compare the zero-shot LLM baseline with trained classifiers on identical test rows.

The zero-shot run and the main prediction run use the same repository-held-out
split, so the two can be compared directly. This refits the trained models on
that split using the cached embeddings, scores both on exactly the rows the
zero-shot run covered, and bootstraps the difference by resampling
repositories (turns within a repository are not independent).

Usage:
  python scripts/compare_zeroshot_vs_trained.py outputs/llm_labels_v4/llm_labels_full.csv \
      --embeddings-dir outputs/ml_action_v4_llm --zeroshot outputs/ml_action_v4_llm/zeroshot/zeroshot_predictions.csv \
      --label-col llm_action_type --output outputs/ml_action_v4_llm/zeroshot/COMPARISON.md
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
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from github_repo_miner.ml_research.action_prediction import (
    PROMPT_METADATA_COLS,
    history_feature_matrix,
    load_action_dataset,
    prompt_keyword_rule_baseline,
)
from github_repo_miner.ml_research.representations import hstack_features, metadata_matrix
from github_repo_miner.ml_research.splits import repo_train_test_split


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input_csv", type=Path)
    p.add_argument("--embeddings-dir", type=Path, required=True)
    p.add_argument("--zeroshot", type=Path, required=True)
    p.add_argument("--label-col", default=None)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def boot_ci(fn, n_boot: int, rng, uniq, idx_by_group) -> tuple[float, float]:
    vals = []
    for _ in range(n_boot):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in drawn])
        vals.append(fn(idx))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> None:
    args = parse_args()
    df, label_col = load_action_dataset(args.input_csv, label_col=args.label_col)
    y = df[label_col].to_numpy()
    emb = np.load(args.embeddings_dir / "prompt_embeddings.npy")
    meta = metadata_matrix(df, [c for c in PROMPT_METADATA_COLS if c in df.columns])
    hist = history_feature_matrix(df)
    x_hist = hstack_features(emb, meta, hist)

    train_idx, test_idx = repo_train_test_split(df, test_size=args.test_size, random_state=args.seed)
    zs = pd.read_csv(args.zeroshot).dropna(subset=["pred"]).set_index("row")["pred"]
    rows = np.array([i for i in test_idx if i in zs.index])  # score everyone on identical rows
    print(f"test rows scored by both: {len(rows)} of {len(test_idx)}")

    preds = {"zero-shot LLM (prompt only)": zs.loc[rows].to_numpy()}
    for name, x in [("trained LR (prompt only)", emb), ("trained LR (prompt+metadata+history)", x_hist)]:
        sc = StandardScaler(with_mean=False).fit(x[train_idx])
        clf = LogisticRegression(max_iter=1000, random_state=args.seed, n_jobs=-1)
        clf.fit(sc.transform(x[train_idx]), y[train_idx])
        preds[name] = clf.predict(sc.transform(x[rows]))
    sc = StandardScaler(with_mean=False).fit(x_hist[train_idx])
    mlp = MLPClassifier(hidden_layer_sizes=(256,), max_iter=300, random_state=args.seed)
    mlp.fit(sc.transform(x_hist[train_idx]), y[train_idx])
    preds["trained MLP (prompt+metadata+history)"] = mlp.predict(sc.transform(x_hist[rows]))
    preds["keyword rule on the prompt"] = np.array(
        prompt_keyword_rule_baseline(df.iloc[rows]["prompt_text"].astype(str).tolist()))

    groups = df["full_name"].to_numpy()[rows]
    uniq = np.unique(groups)
    idx_by_group = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(args.seed)
    y_te = y[rows]

    table = []
    for name, p in preds.items():
        macro = f1_score(y_te, p, average="macro", zero_division=0)
        lo, hi = boot_ci(lambda i, p=p: f1_score(y_te[i], p[i], average="macro", zero_division=0),
                         args.bootstrap, rng, uniq, idx_by_group)
        table.append({"predictor": name, "macro_f1": round(macro, 4), "ci_low": round(lo, 4),
                      "ci_high": round(hi, 4), "accuracy": round(float((p == y_te).mean()), 4)})
    summary = pd.DataFrame(table).sort_values("macro_f1", ascending=False)

    diffs = []
    zs_pred = preds["zero-shot LLM (prompt only)"]
    for name, p in preds.items():
        if name.startswith("zero-shot"):
            continue
        d = f1_score(y_te, zs_pred, average="macro", zero_division=0) - f1_score(y_te, p, average="macro", zero_division=0)
        lo, hi = boot_ci(lambda i, p=p: (f1_score(y_te[i], zs_pred[i], average="macro", zero_division=0)
                                         - f1_score(y_te[i], p[i], average="macro", zero_division=0)),
                         args.bootstrap, rng, uniq, idx_by_group)
        diffs.append({"comparison": f"zero-shot minus {name}", "mean_diff": round(d, 4),
                      "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                      "zero-shot better": "yes" if lo > 0 else ("no" if hi < 0 else "inconclusive")})

    md = [f"# Zero-shot LLM vs trained classifiers (`{label_col}`)\n",
          f"Identical held-out rows ({len(rows)} turns, {len(uniq)} repositories). "
          f"Cluster bootstrap over repositories, {args.bootstrap} resamples.\n",
          summary.to_markdown(index=False), "", "## Paired differences\n",
          pd.DataFrame(diffs).to_markdown(index=False), ""]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
