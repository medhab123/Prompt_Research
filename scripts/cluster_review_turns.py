"""Is `review` one behavior? Embed every turn labeled review by the rule-based
labeler or the LLM and look for sub-clusters.

For each embedding view (the developer prompt, the agent response, or both) it
runs k-means for k=2..8 and reports the silhouette score. For scale, it also
scores the same procedure on an equally sized random sample of `implement`
turns and of all turns. It then describes the best-k clusters: size, which
labeler(s) called them review, distinctive terms and example prompts.

Usage:
    python scripts/cluster_review_turns.py --model all-MiniLM-L6-v2
    python scripts/cluster_review_turns.py --model nomic-ai/CodeRankEmbed
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "outputs/full_run_v3_analysis/specstory_prompts_cleaned.csv"
LLM_LABELS = ROOT / "outputs/llm_labels_v3/llm_labels_full.csv"
KEYS = ["full_name", "source_file", "turn_index"]
RESPONSE_CHARS = 1500
K_RANGE = range(2, 9)
# CodeRankEmbed is a retrieval model: queries (natural-language text) take this prefix.
QUERY_PREFIX = {"nomic-ai/CodeRankEmbed": "Represent this query for searching relevant code: "}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="all-MiniLM-L6-v2")
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs/review_subclusters")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_turns() -> pd.DataFrame:
    corpus = pd.read_csv(CORPUS)
    llm = pd.read_csv(LLM_LABELS, usecols=KEYS + ["llm_action_type", "llm_error"])
    df = corpus.merge(llm, on=KEYS, how="left", validate="one_to_one")
    df = df[df["llm_action_type"].notna() & df["llm_error"].isna()].copy()
    # Empty responses form a trivial cluster of their own (the regex's "asked to
    # build, no code -> review" rule), which says nothing about review behaviors.
    df = df[df["agent_response"].fillna("").astype(str).str.strip().ne("")]
    df["response_head"] = df["agent_response"].fillna("").astype(str).str.slice(0, RESPONSE_CHARS)
    df["prompt_text"] = df["prompt_text"].fillna("").astype(str)
    return df.reset_index(drop=True)


def embed(texts: list[str], model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, trust_remote_code=model_name in QUERY_PREFIX)
    model.max_seq_length = min(model.max_seq_length, 512)  # CodeRankEmbed allows 8192; too slow on CPU
    prefix = QUERY_PREFIX.get(model_name, "")
    return model.encode([prefix + t for t in texts], batch_size=32, show_progress_bar=True,
                        convert_to_numpy=True, normalize_embeddings=True)


def sweep(x: np.ndarray, seed: int) -> dict[int, tuple[float, np.ndarray]]:
    out = {}
    for k in K_RANGE:
        labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(x)
        out[k] = (float(silhouette_score(x, labels, metric="cosine")), labels)
    return out


def top_terms(texts: pd.Series, labels: np.ndarray, n: int = 10) -> dict[int, str]:
    # One document per cluster, so min_df/max_df must stay at their defaults.
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True,
                          token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
    docs = [" ".join(texts[labels == c]) for c in sorted(set(labels))]  # class-based TF-IDF
    m = vec.fit_transform(docs).toarray()
    vocab = np.array(vec.get_feature_names_out())
    return {c: ", ".join(vocab[np.argsort(-m[i])[:n]]) for i, c in enumerate(sorted(set(labels)))}


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir / re.sub(r"[^\w.-]", "_", args.model)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_turns()

    df["review_by"] = np.select(
        [(df["action_type"] == "review") & (df["llm_action_type"] == "review"),
         df["action_type"] == "review", df["llm_action_type"] == "review"],
        ["both", "regex only", "LLM only"], default="")
    rev = df[df["review_by"] != ""].reset_index(drop=True)
    n = len(rev)
    rng = np.random.default_rng(args.seed)
    impl = df[(df["action_type"] == "implement") & (df["llm_action_type"] == "implement")]
    comparisons = {
        "review (either labeler)": rev,
        "implement (both labelers), same n": impl.iloc[rng.choice(len(impl), n, replace=False)],
        "all turns, same n": df.iloc[rng.choice(len(df), n, replace=False)],
    }
    print(f"review turns: {n}", rev["review_by"].value_counts().to_dict())

    views = {
        "prompt": lambda d: d["prompt_text"].tolist(),
        "response": lambda d: d["response_head"].tolist(),
    }
    sil_rows, best = [], {}
    for view, get in views.items():
        for name, d in comparisons.items():
            x = embed(get(d), args.model)
            res = sweep(x, args.seed)
            row = {"view": view, "set": name, "n": len(d)}
            row.update({f"k={k}": round(s, 3) for k, (s, _) in res.items()})
            sil_rows.append(row)
            if name.startswith("review"):
                best[view] = (x, res)
    sil = pd.DataFrame(sil_rows)
    sil.to_csv(out_dir / "silhouette.csv", index=False)
    print(sil.to_string(index=False))

    # Describe the review clusters for each view at its best k.
    lines = [f"# Review sub-clusters — {args.model}\n", f"{n} turns labeled review by either labeler.\n",
             "## Silhouette (cosine) by k\n", sil.to_markdown(index=False), ""]
    for view, (x, res) in best.items():
        k = max(res, key=lambda kk: res[kk][0])
        labels = res[k][1]
        rev[f"cluster_{view}"] = labels
        terms = top_terms(rev["prompt_text"] if view == "prompt" else rev["response_head"], labels)
        lines += [f"## View: {view} — best k={k} (silhouette {res[k][0]:.3f})\n"]
        tab = pd.crosstab(labels, rev["review_by"])
        tab["n"] = tab.sum(axis=1)
        tab["distinctive terms"] = [terms[c] for c in tab.index]
        lines += [tab.reset_index(names="cluster").to_markdown(index=False), ""]
        for c in sorted(set(labels)):
            ex = rev[labels == c].sample(min(5, int((labels == c).sum())), random_state=args.seed)
            lines.append(f"**Cluster {c} examples**")
            lines += [f"- `{r.review_by}` | {re.sub(r'\s+', ' ', r.prompt_text)[:160]}" for r in ex.itertuples()]
            lines.append("")
    rev.drop(columns=["response_head"]).to_csv(out_dir / "review_turns_clustered.csv", index=False, encoding="utf-8-sig")
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
