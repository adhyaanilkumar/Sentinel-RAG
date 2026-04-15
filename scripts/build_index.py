"""Build the FAISS vector index from the FM corpus."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from core.config_loader import load_config
from core.document_processor import load_corpus
from core.embeddings import EmbeddingModel
from retrieval.vector_store import VectorStore


def main():
    config = load_config()

    corpus_dir = Path(__file__).resolve().parent.parent / "data" / "corpus"
    index_dir = Path(__file__).resolve().parent.parent / "data" / "index"

    print(f"Loading corpus from {corpus_dir}")
    chunks = load_corpus(
        corpus_dir,
        chunk_size=config.retrieval.chunk_size,
        chunk_overlap=config.retrieval.chunk_overlap,
    )

    print(f"\nBuilding embeddings with {config.embedding.model_name}")
    emb_model = EmbeddingModel(config.embedding.model_name)
    vs = VectorStore(emb_model)
    vs.build(chunks)

    print(f"Saving index to {index_dir}")
    vs.save(index_dir)
    print("Done.")


if __name__ == "__main__":
    main()
