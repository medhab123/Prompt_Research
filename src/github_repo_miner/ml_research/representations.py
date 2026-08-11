"""Text representation builders: SBERT, TF-IDF, and code-aware embeddings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def _resolve_device(device: str | None) -> str:
    if device:
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def encode_sbert(
    texts: list[str],
    model_name: str,
    *,
    batch_size: int = 64,
    device: str | None = None,
    normalize: bool = True,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    resolved = _resolve_device(device)
    model = SentenceTransformer(model_name, device=resolved)
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )


def build_tfidf(
    train_texts: list[str],
    test_texts: list[str] | None = None,
    *,
    max_features: int = 2000,
) -> tuple[np.ndarray, np.ndarray | None, TfidfVectorizer]:
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=2,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    x_train = vectorizer.fit_transform(train_texts)
    x_test = vectorizer.transform(test_texts) if test_texts is not None else None
    return x_train, x_test, vectorizer


def encode_code_aware(
    texts: list[str],
    model_name: str,
    *,
    batch_size: int = 32,
    device: str | None = None,
) -> np.ndarray | None:
    """Encode prompts with a code-oriented embedding model; return None if unavailable."""
    try:
        return encode_sbert(texts, model_name, batch_size=batch_size, device=device, normalize=True)
    except Exception as exc:
        print(f"Code-aware embeddings unavailable ({model_name}): {exc}")
        return None


def load_or_compute_embeddings(
    texts: list[str],
    cache_path: Path,
    model_name: str,
    *,
    batch_size: int = 64,
    device: str | None = None,
    force_recompute: bool = False,
) -> np.ndarray:
    if cache_path.exists() and not force_recompute:
        cached = np.load(cache_path)
        if cached.shape[0] == len(texts):
            print(f"Loaded cached embeddings: {cache_path}")
            return cached
        print("Cache size mismatch — recomputing embeddings.")

    print(f"Computing SBERT embeddings with {model_name}...")
    embeddings = encode_sbert(texts, model_name, batch_size=batch_size, device=device)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    return embeddings


def metadata_matrix(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Numeric metadata feature matrix with boolean coercion."""
    block = df[columns].copy()
    for col in block.columns:
        if block[col].dtype == bool:
            block[col] = block[col].astype(float)
        else:
            block[col] = pd.to_numeric(block[col], errors="coerce").fillna(0.0)
    return block.to_numpy(dtype=np.float32)


def hstack_features(*parts: np.ndarray) -> np.ndarray:
    from scipy.sparse import hstack, issparse

    sparse_parts = [p for p in parts if issparse(p)]
    dense_parts = [p for p in parts if not issparse(p)]
    if sparse_parts and dense_parts:
        from scipy.sparse import csr_matrix

        dense_sparse = [csr_matrix(d) if d.ndim == 2 else csr_matrix(d.reshape(-1, 1)) for d in dense_parts]
        return hstack(sparse_parts + dense_sparse).tocsr()
    if sparse_parts:
        return hstack(sparse_parts).tocsr()
    return np.hstack(dense_parts)
