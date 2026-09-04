"""Configuration for the ML research pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MLResearchConfig:
    """Paths, models, and split settings for supervised ML experiments."""

    input_csv: Path
    output_dir: Path = Path("outputs/ml")
    embedding_model: str = "all-MiniLM-L6-v2"
    code_embedding_model: str = "nomic-ai/CodeRankEmbed"
    test_size: float = 0.2
    random_state: int = 42
    n_cv_folds: int = 5
    batch_size: int = 64
    device: str | None = None
    skip_code_embeddings: bool = False
    min_intent_class_size: int = 30
    tfidf_max_features: int = 2000
