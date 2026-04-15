"""Rebuild benchmark_checkpoint.pkl from saved JSON answers + fresh retrieval.

Use when a long benchmark run completed Vanilla / Iterative / Sentinel (with decay)
but crashed before ablation or evaluation. This avoids re-calling the LLM for those
phases while restoring full GenerationResult objects (including retrieved chunks)
needed for RAGAS and evidence metrics.

Iterative RAG retrieval is approximated with the same vector search as vanilla
(final answers and token counts still come from the saved JSON).
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _format_vanilla_context(retrieval) -> str:
    parts = []
    for i, chunk in enumerate(retrieval.chunks):
        parts.append(
            f"[Source {i+1}: {chunk.source_document}, {chunk.section_id}]\n{chunk.text}"
        )
    return "\n\n".join(parts)


def _format_sentinel_context(retrieval) -> str:
    parts = []
    for i, (chunk, score) in enumerate(zip(retrieval.chunks, retrieval.scores)):
        parts.append(
            f"[Source {i+1}: {chunk.source_document}, {chunk.section_id} "
            f"(relevance={score:.3f})]\n{chunk.text}"
        )
    return "\n\n".join(parts)


def _rows_by_id(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {r["query_id"]: r for r in data}


def main() -> None:
    from core.config_loader import load_config
    from core.data_models import GenerationResult
    from core.embeddings import EmbeddingModel
    from data.gold_annotations import GOLD_ANNOTATIONS
    from graph.knowledge_graph import KnowledgeGraph
    from retrieval.graph_retriever import CSSConfig, GraphRetriever
    from retrieval.vector_store import VectorStore
    from sentinel.temporal_decay import TemporalDecayEngine

    root = Path(__file__).resolve().parent.parent
    results_dir = root / "data" / "results"
    index_dir = root / "data" / "index"
    graph_dir = root / "data" / "graph"
    config = load_config()
    annotations = GOLD_ANNOTATIONS
    n = len(annotations)

    for name in ("vanilla_rag_results.json", "iterative_rag_results.json", "sentinel_rag_results.json"):
        p = results_dir / name
        if not p.exists():
            raise SystemExit(f"Missing {p}; cannot seed checkpoint.")

    vanilla_rows = _rows_by_id(results_dir / "vanilla_rag_results.json")
    iterative_rows = _rows_by_id(results_dir / "iterative_rag_results.json")
    sentinel_rows = _rows_by_id(results_dir / "sentinel_rag_results.json")

    if len(vanilla_rows) != n or len(iterative_rows) != n or len(sentinel_rows) != n:
        raise SystemExit(
            f"Expected {n} rows per JSON; got vanilla={len(vanilla_rows)}, "
            f"iterative={len(iterative_rows)}, sentinel={len(sentinel_rows)}"
        )

    print("Loading vector store and graph...")
    emb_model = EmbeddingModel(config.embedding.model_name)
    vs = VectorStore(emb_model)
    vs.load(index_dir)

    kg = KnowledgeGraph()
    with open(graph_dir / "graph.gpickle", "rb") as f:
        kg.graph = pickle.load(f)
    with open(graph_dir / "nodes.pkl", "rb") as f:
        kg.nodes = pickle.load(f)

    css_cfg = CSSConfig(
        relevance=config.css.weights.relevance,
        context_cohesion=config.css.weights.context_cohesion,
        subquery_coverage=config.css.weights.subquery_coverage,
        cross_ref_bonus=config.css.weights.cross_ref_bonus,
        entity_overlap=config.css.weights.entity_overlap,
        temporal_recency=config.css.weights.temporal_recency,
        token_budget=config.css.token_budget,
        redundancy_threshold=config.css.redundancy_threshold,
    )
    graph_retriever = GraphRetriever(
        vector_store=vs,
        knowledge_graph=kg,
        embedding_model=emb_model,
        css_config=css_cfg,
        initial_top_k=15,
        max_hops=2,
        final_top_k=config.retrieval.top_k,
    )
    temporal_engine = TemporalDecayEngine(
        decay_function=config.temporal.decay_function,
        half_life_hours=config.temporal.half_life_hours,
        stale_threshold_hours=config.temporal.stale_threshold_hours,
        flag_stale=config.temporal.flag_stale,
    )

    vanilla_gens: list[GenerationResult] = []
    iterative_gens: list[GenerationResult] = []
    sentinel_gens: list[GenerationResult] = []

    print("Rebuilding generations (retrieval only, answers from JSON)...")
    for ann in annotations:
        vr = vanilla_rows[ann.id]
        rr_v = vs.search(ann.query, top_k=config.retrieval.top_k)
        vanilla_gens.append(
            GenerationResult(
                answer=vr["answer"],
                retrieved_context=_format_vanilla_context(rr_v),
                retrieval_result=rr_v,
                prompt_tokens=vr.get("prompt_tokens", 0),
                completion_tokens=vr.get("completion_tokens", 0),
                total_tokens=vr.get("total_tokens", 0),
                latency_seconds=vr.get("latency_seconds", 0.0),
                num_iterations=vr.get("num_iterations", 1),
                model_name="",
            )
        )

        ir = iterative_rows[ann.id]
        rr_i = vs.search(ann.query, top_k=config.retrieval.top_k)
        iterative_gens.append(
            GenerationResult(
                answer=ir["answer"],
                retrieved_context=_format_vanilla_context(rr_i),
                retrieval_result=rr_i,
                prompt_tokens=ir.get("prompt_tokens", 0),
                completion_tokens=ir.get("completion_tokens", 0),
                total_tokens=ir.get("total_tokens", 0),
                latency_seconds=ir.get("latency_seconds", 0.0),
                num_iterations=ir.get("num_iterations", 1),
                model_name="",
            )
        )

        sr = sentinel_rows[ann.id]
        tw = temporal_engine.compute_weights(kg)
        rr_s = graph_retriever.retrieve(ann.query, temporal_weights=tw)
        sentinel_gens.append(
            GenerationResult(
                answer=sr["answer"],
                retrieved_context=_format_sentinel_context(rr_s),
                retrieval_result=rr_s,
                prompt_tokens=sr.get("prompt_tokens", 0),
                completion_tokens=sr.get("completion_tokens", 0),
                total_tokens=sr.get("total_tokens", 0),
                latency_seconds=sr.get("latency_seconds", 0.0),
                num_iterations=sr.get("num_iterations", 1),
                model_name="",
            )
        )

    cp = {
        "vanilla_gens": vanilla_gens,
        "iterative_gens": iterative_gens,
        "sentinel_gens": sentinel_gens,
    }
    out = results_dir / "benchmark_checkpoint.pkl"
    tmp = out.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(cp, f)
    tmp.replace(out)
    print(f"Wrote {out} (sentinel_no_decay_gens not set — run_full_benchmark will run ablation only)")


if __name__ == "__main__":
    main()
