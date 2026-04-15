"""FAISS-based vector store for document retrieval."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from core.data_models import DocumentChunk, RetrievalResult
from core.embeddings import EmbeddingModel


class VectorStore:
    """FAISS vector store wrapping chunk embeddings."""

    def __init__(self, embedding_model: EmbeddingModel):
        import faiss
        self.embedding_model = embedding_model
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: list[DocumentChunk] = []
        self._faiss = faiss

    def build(self, chunks: list[DocumentChunk], batch_size: int = 64) -> None:
        self.chunks = chunks
        texts = [c.text for c in chunks]
        embeddings = self.embedding_model.encode(texts, batch_size=batch_size)
        dim = embeddings.shape[1]
        self.index = self._faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype(np.float32))

    def search(self, query: str, top_k: int = 10, threshold: float = 0.0) -> RetrievalResult:
        if self.index is None:
            raise RuntimeError("Vector store not built. Call build() first.")

        import time
        start = time.time()
        query_vec = self.embedding_model.encode_single(query).reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(query_vec, top_k)
        latency = time.time() - start

        result_chunks = []
        result_scores = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or score < threshold:
                continue
            result_chunks.append(self.chunks[idx])
            result_scores.append(float(score))

        return RetrievalResult(
            chunks=result_chunks,
            scores=result_scores,
            retrieval_method="faiss_flat_ip",
            latency_seconds=latency,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self.index, str(path / "faiss.index"))
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        self.index = self._faiss.read_index(str(path / "faiss.index"))
        with open(path / "chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)
