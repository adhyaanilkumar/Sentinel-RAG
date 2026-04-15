"""Run vanilla and iterative RAG baselines against gold annotations."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from core.config_loader import load_config
from core.embeddings import EmbeddingModel
from data.gold_annotations import GOLD_ANNOTATIONS
from baselines.vanilla_rag import VanillaRAG
from baselines.iterative_rag import IterativeRAG
from generation.llm_client import LLMClient
from retrieval.vector_store import VectorStore


def _print(msg: str) -> None:
    print(msg, flush=True)


def main():
    config = load_config()
    index_dir = Path(__file__).resolve().parent.parent / "data" / "index"
    results_dir = Path(__file__).resolve().parent.parent / "data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    _print("Loading vector store...")
    emb_model = EmbeddingModel(config.embedding.model_name)
    vs = VectorStore(emb_model)
    vs.load(index_dir)
    _print(f"  Loaded {len(vs.chunks)} chunks")

    llm = LLMClient(provider=config.llm_provider, model=config.openai_model)

    _print("\n--- Running Vanilla RAG Baseline ---")
    vanilla = VanillaRAG(vs, llm, top_k=config.retrieval.top_k)
    vanilla_results = []

    for i, ann in enumerate(GOLD_ANNOTATIONS):
        _print(f"  [{i+1}/40] [{ann.id}] {ann.query[:70]}...")
        try:
            result = vanilla.query(ann.query)
        except Exception as e:
            _print(f"    ERROR: {e}, retrying in 5s...")
            import time; time.sleep(5)
            try:
                result = vanilla.query(ann.query)
            except Exception as e2:
                _print(f"    RETRY FAILED: {e2}, skipping")
                continue
        vanilla_results.append({
            "query_id": ann.id,
            "category": ann.category.value,
            "hop_count": ann.hop_count.value,
            "query": ann.query,
            "answer": result.answer,
            "ground_truth": ann.ground_truth_answer,
            "information_units": ann.information_units,
            "num_chunks_retrieved": len(result.retrieval_result.chunks),
            "sources_retrieved": [c.source_document for c in result.retrieval_result.chunks],
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.total_tokens,
            "latency_seconds": result.latency_seconds,
            "num_iterations": result.num_iterations,
        })

    with open(results_dir / "vanilla_rag_results.json", "w") as f:
        json.dump(vanilla_results, f, indent=2)
    _print(f"  Saved {len(vanilla_results)} results")

    _print("\n--- Running Iterative RAG Baseline ---")
    iterative = IterativeRAG(
        vs, llm, top_k=config.retrieval.top_k, max_iterations=5, coverage_threshold=0.9,
    )
    iterative_results = []

    for i, ann in enumerate(GOLD_ANNOTATIONS):
        _print(f"  [{i+1}/40] [{ann.id}] {ann.query[:70]}...")
        try:
            result = iterative.query(ann.query, information_checklist=ann.information_units)
        except Exception as e:
            _print(f"    ERROR: {e}, retrying in 5s...")
            import time; time.sleep(5)
            try:
                result = iterative.query(ann.query, information_checklist=ann.information_units)
            except Exception as e2:
                _print(f"    RETRY FAILED: {e2}, skipping")
                continue
        iterative_results.append({
            "query_id": ann.id,
            "category": ann.category.value,
            "hop_count": ann.hop_count.value,
            "query": ann.query,
            "answer": result.answer,
            "ground_truth": ann.ground_truth_answer,
            "information_units": ann.information_units,
            "num_chunks_retrieved": len(result.retrieval_result.chunks),
            "sources_retrieved": [c.source_document for c in result.retrieval_result.chunks],
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.total_tokens,
            "latency_seconds": result.latency_seconds,
            "num_iterations": result.num_iterations,
        })

    with open(results_dir / "iterative_rag_results.json", "w") as f:
        json.dump(iterative_results, f, indent=2)
    _print(f"  Saved {len(iterative_results)} results")

    _print(f"\nDone! Results saved to {results_dir}")


if __name__ == "__main__":
    main()
