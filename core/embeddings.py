"""Embedding model wrapper using sentence-transformers."""

from __future__ import annotations

import numpy as np


class EmbeddingModel:
    """Wraps a sentence-transformers model for encoding text into vectors."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]
