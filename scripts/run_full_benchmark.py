"""Run the complete Sentinel-RAG benchmark suite.

Executes: Vanilla RAG -> Iterative RAG -> Sentinel-RAG -> Sentinel-RAG (no decay) -> Evaluation -> Report
Produces: benchmark_report.md, benchmark_report.json with LaTeX tables, per-FM breakdown, ablation.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _print(msg: str) -> None:
    print(msg, flush=True)


def _safe_query(system, question: str, **kwargs):
    """Run a query with exponential backoff on transient failures (e.g. connection errors)."""
    from core.data_models import GenerationResult, RetrievalResult

    delays = (3, 10, 25, 45)
    last_err: Exception | None = None
    for attempt, delay in enumerate(delays, start=1):
        try:
            return system.query(question, **kwargs)
        except Exception as e:
            last_err = e
            _print(f"    ERROR (attempt {attempt}/{len(delays)}): {e}, waiting {delay}s...")
            time.sleep(delay)
    _print(f"    ALL RETRIES FAILED: {last_err}, returning error placeholder")
    return GenerationResult(
        answer="ERROR",
        retrieved_context="",
        retrieval_result=RetrievalResult(chunks=[], scores=[]),
    )


def _load_checkpoint(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return {}


def _save_checkpoint(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(data, f)
    tmp.replace(path)


def main():
    import argparse

    from core.config_loader import load_config
    from core.data_models import GenerationResult
    from core.embeddings import EmbeddingModel
    from data.gold_annotations import GOLD_ANNOTATIONS
    from baselines.vanilla_rag import VanillaRAG
    from baselines.iterative_rag import IterativeRAG
    from evaluation.harness import EvaluationHarness
    from generation.llm_client import LLMClient
    from graph.knowledge_graph import KnowledgeGraph
    from retrieval.graph_retriever import CSSConfig, GraphRetriever
    from retrieval.vector_store import VectorStore
    from sentinel.sentinel_rag import SentinelRAG
    from sentinel.temporal_decay import TemporalDecayEngine

    parser = argparse.ArgumentParser(description="Run Sentinel-RAG benchmark suite")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore benchmark_checkpoint.pkl and re-run all query phases from scratch",
    )
    args = parser.parse_args()

    config = load_config()
    root = Path(__file__).resolve().parent.parent
    index_dir = root / "data" / "index"
    graph_dir = root / "data" / "graph"
    results_dir = root / "data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = results_dir / "benchmark_checkpoint.pkl"
    cp: dict = {} if args.fresh else _load_checkpoint(checkpoint_path)

    annotations = GOLD_ANNOTATIONS
    n_queries = len(annotations)

    _print("=" * 60)
    _print("SENTINEL-RAG FULL BENCHMARK SUITE")
    _print(f"  Queries: {n_queries}")
    if args.fresh:
        _print("  Mode: FRESH (checkpoint ignored)")
    elif cp:
        _print("  Mode: RESUME (using benchmark_checkpoint.pkl where complete)")
    _print("=" * 60)

    # ============================================================
    # 1. Load infrastructure
    # ============================================================
    _print("\n[1/8] Loading vector store...")
    emb_model = EmbeddingModel(config.embedding.model_name)
    vs = VectorStore(emb_model)
    vs.load(index_dir)
    _print(f"  Loaded {len(vs.chunks)} chunks")

    _print("\n[2/8] Loading knowledge graph...")
    kg = KnowledgeGraph()
    with open(graph_dir / "graph.gpickle", "rb") as f:
        kg.graph = pickle.load(f)
    with open(graph_dir / "nodes.pkl", "rb") as f:
        kg.nodes = pickle.load(f)
    _print(f"  Graph: {kg.num_nodes} nodes, {kg.num_edges} edges, connectivity={kg.connectivity:.2%}")

    llm = LLMClient(provider=config.llm_provider, model=config.openai_model)

    # ============================================================
    # 2. Vanilla RAG Baseline
    # ============================================================
    _print(f"\n[3/8] Running Vanilla RAG Baseline ({n_queries} queries)...")
    vanilla_gens: list[GenerationResult] = list(cp.get("vanilla_gens") or [])
    if len(vanilla_gens) > n_queries:
        vanilla_gens = vanilla_gens[:n_queries]
    if len(vanilla_gens) < n_queries:
        vanilla = VanillaRAG(vs, llm, top_k=config.retrieval.top_k)
        for i in range(len(vanilla_gens), n_queries):
            ann = annotations[i]
            _print(f"  [{i+1}/{n_queries}] {ann.id}")
            vanilla_gens.append(_safe_query(vanilla, ann.query))
            cp["vanilla_gens"] = vanilla_gens
            _save_checkpoint(checkpoint_path, cp)
    else:
        _print("  (skipped: loaded from checkpoint)")

    _save_raw_results(results_dir / "vanilla_rag_results.json", annotations, vanilla_gens, "Vanilla RAG")

    # ============================================================
    # 3. Iterative RAG Baseline
    # ============================================================
    _print(f"\n[4/8] Running Iterative RAG Baseline ({n_queries} queries)...")
    iterative_gens: list[GenerationResult] = list(cp.get("iterative_gens") or [])
    if len(iterative_gens) > n_queries:
        iterative_gens = iterative_gens[:n_queries]
    if len(iterative_gens) < n_queries:
        iterative = IterativeRAG(
            vs, llm, top_k=config.retrieval.top_k, max_iterations=5, coverage_threshold=0.9,
        )
        for i in range(len(iterative_gens), n_queries):
            ann = annotations[i]
            _print(f"  [{i+1}/{n_queries}] {ann.id}")
            iterative_gens.append(
                _safe_query(iterative, ann.query, information_checklist=ann.information_units)
            )
            cp["iterative_gens"] = iterative_gens
            _save_checkpoint(checkpoint_path, cp)
    else:
        _print("  (skipped: loaded from checkpoint)")

    _save_raw_results(results_dir / "iterative_rag_results.json", annotations, iterative_gens, "Iterative RAG")

    # ============================================================
    # 4. Sentinel-RAG (with temporal decay)
    # ============================================================
    _print(f"\n[5/8] Running Sentinel-RAG WITH temporal decay ({n_queries} queries)...")
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
        vector_store=vs, knowledge_graph=kg, embedding_model=emb_model,
        css_config=css_cfg, initial_top_k=15, max_hops=2, final_top_k=config.retrieval.top_k,
    )
    temporal_engine = TemporalDecayEngine(
        decay_function=config.temporal.decay_function,
        half_life_hours=config.temporal.half_life_hours,
        stale_threshold_hours=config.temporal.stale_threshold_hours,
        flag_stale=config.temporal.flag_stale,
    )
    sentinel = SentinelRAG(graph_retriever, kg, llm, temporal_engine, top_k=config.retrieval.top_k)

    sentinel_gens: list[GenerationResult] = list(cp.get("sentinel_gens") or [])
    if len(sentinel_gens) > n_queries:
        sentinel_gens = sentinel_gens[:n_queries]
    if len(sentinel_gens) < n_queries:
        for i in range(len(sentinel_gens), n_queries):
            ann = annotations[i]
            _print(f"  [{i+1}/{n_queries}] {ann.id}")
            sentinel_gens.append(_safe_query(sentinel, ann.query, enable_temporal=True))
            cp["sentinel_gens"] = sentinel_gens
            _save_checkpoint(checkpoint_path, cp)
    else:
        _print("  (skipped: loaded from checkpoint)")

    _save_raw_results(results_dir / "sentinel_rag_results.json", annotations, sentinel_gens, "Sentinel-RAG")

    # ============================================================
    # 5. Sentinel-RAG (WITHOUT temporal decay — ablation)
    # ============================================================
    _print(f"\n[6/8] Running Sentinel-RAG WITHOUT temporal decay (ablation) ({n_queries} queries)...")
    sentinel_no_decay_gens: list[GenerationResult] = list(cp.get("sentinel_no_decay_gens") or [])
    if len(sentinel_no_decay_gens) > n_queries:
        sentinel_no_decay_gens = sentinel_no_decay_gens[:n_queries]
    if len(sentinel_no_decay_gens) < n_queries:
        for i in range(len(sentinel_no_decay_gens), n_queries):
            ann = annotations[i]
            _print(f"  [{i+1}/{n_queries}] {ann.id}")
            sentinel_no_decay_gens.append(_safe_query(sentinel, ann.query, enable_temporal=False))
            cp["sentinel_no_decay_gens"] = sentinel_no_decay_gens
            _save_checkpoint(checkpoint_path, cp)
    else:
        _print("  (skipped: loaded from checkpoint)")

    _save_raw_results(
        results_dir / "sentinel_rag_no_decay_results.json",
        annotations, sentinel_no_decay_gens, "Sentinel-RAG (no decay)",
    )

    # ============================================================
    # 6. Evaluate all systems
    # ============================================================
    _print("\n[7/8] Evaluating all systems (includes RAGAS metrics)...")
    harness = EvaluationHarness(run_ragas=True)

    _print("  Evaluating Vanilla RAG...")
    baseline_results = harness.evaluate_batch(annotations, vanilla_gens, "Vanilla RAG")
    _print("  Evaluating Iterative RAG...")
    iterative_results = harness.evaluate_batch(annotations, iterative_gens, "Iterative RAG")
    _print("  Evaluating Sentinel-RAG (with decay)...")
    sentinel_results = harness.evaluate_batch(annotations, sentinel_gens, "Sentinel-RAG")
    _print("  Evaluating Sentinel-RAG (no decay — ablation)...")
    ablation_results = harness.evaluate_batch(annotations, sentinel_no_decay_gens, "Sentinel-RAG (no decay)")

    _print("  Running statistical comparisons...")
    comparisons = harness.compare_systems(baseline_results, sentinel_results)
    ablation_comparisons = harness.compare_systems(ablation_results, sentinel_results)
    iterative_comparisons = harness.compare_systems(baseline_results, iterative_results)

    # ============================================================
    # 7. Generate report
    # ============================================================
    _print("\n[8/8] Generating benchmark report...")
    report_path = results_dir / "benchmark_report.md"
    report = harness.generate_report(
        baseline_results, sentinel_results, comparisons, report_path,
        ablation_results=ablation_results,
        ablation_comparisons=ablation_comparisons,
    )
    _print(f"  Report saved to {report_path}")
    _print(f"  JSON data saved to {report_path.with_suffix('.json')}")

    # ============================================================
    # Summary
    # ============================================================
    _print("\n" + "=" * 60)
    _print("BENCHMARK SUMMARY")
    _print("=" * 60)

    _print("\n--- Vanilla RAG vs Sentinel-RAG ---")
    for c in comparisons:
        sig = "***" if c.corrected_significant else ("*" if c.significant else "")
        _print(f"  {c.metric_name}: Vanilla={c.system_a_mean:.3f} Sentinel={c.system_b_mean:.3f} "
               f"p={c.p_value:.4f} d={c.effect_size_d:.3f} {sig}")

    _print("\n--- Adversarial Trap Results ---")
    for label, cat_val, extractor in [
        ("Trap A Fatal Errors", "trap_a_overriding_directive",
         lambda rs: f"{sum(1 for r in rs if r.fatal_error)}/{len(rs)}"),
        ("Trap B Defn Retrieved", "trap_b_distant_definition",
         lambda rs: f"{sum(1 for r in rs if r.definition_retrieved)}/{len(rs)}"),
        ("Trap C Component Recall", "trap_c_scattered_components",
         lambda rs: f"{sum(r.component_recall for r in rs)/max(len(rs),1):.3f}"),
    ]:
        base_cat = [r for r in baseline_results if r.category.value == cat_val]
        sent_cat = [r for r in sentinel_results if r.category.value == cat_val]
        _print(f"  {label}: Vanilla={extractor(base_cat)} Sentinel={extractor(sent_cat)}")

    _print("\n--- Temporal Decay Ablation ---")
    for c in ablation_comparisons:
        sig = "*" if c.significant else ""
        _print(f"  {c.metric_name}: WithDecay={c.system_b_mean:.3f} NoDecay={c.system_a_mean:.3f} "
               f"p={c.p_value:.4f} {sig}")

    _print("\n--- Efficiency (Tokens-to-Truth) ---")
    for sname, gens in [("Vanilla", vanilla_gens), ("Iterative", iterative_gens), ("Sentinel", sentinel_gens)]:
        avg_tok = sum(g.total_tokens for g in gens) / len(gens)
        avg_lat = sum(g.latency_seconds for g in gens) / len(gens)
        avg_it = sum(g.num_iterations for g in gens) / len(gens)
        _print(f"  {sname}: tokens={avg_tok:.0f} latency={avg_lat:.1f}s iterations={avg_it:.1f}")

    # Save iterative RAG evaluation alongside others in JSON
    _save_evaluation_supplement(
        results_dir / "iterative_rag_evaluation.json",
        iterative_results, iterative_comparisons,
    )

    _print("\nDone!")
    _print("  Checkpoint retained at data/results/benchmark_checkpoint.pkl (delete or use --fresh to re-run all phases)")


def _save_evaluation_supplement(
    path: Path, results, comparisons,
) -> None:
    """Save supplementary evaluation data (e.g. iterative RAG) to JSON."""
    from dataclasses import asdict
    data = {
        "system": results[0].system_name if results else "unknown",
        "n": len(results),
        "comparisons_vs_vanilla": [asdict(c) for c in comparisons],
        "per_query": [
            {
                "query_id": r.query_id,
                "category": r.category.value,
                "information_unit_coverage": r.information_unit_coverage,
                "evidence_recall": r.evidence_recall,
                "component_recall": r.component_recall,
                "faithfulness": r.faithfulness,
                "answer_correctness": r.answer_correctness,
                "total_tokens": r.generation_result.total_tokens,
                "latency_seconds": r.generation_result.latency_seconds,
                "num_iterations": r.generation_result.num_iterations,
            }
            for r in results
        ],
    }
    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def _save_raw_results(path: Path, annotations, generations, system_name: str) -> None:
    """Save raw query results to JSON for inspection."""
    results = []
    for ann, gen in zip(annotations, generations):
        results.append({
            "query_id": ann.id,
            "system": system_name,
            "category": ann.category.value,
            "hop_count": ann.hop_count.value,
            "query": ann.query,
            "answer": gen.answer,
            "ground_truth": ann.ground_truth_answer,
            "information_units": ann.information_units,
            "num_chunks_retrieved": len(gen.retrieval_result.chunks),
            "sources_retrieved": [c.source_document for c in gen.retrieval_result.chunks],
            "prompt_tokens": gen.prompt_tokens,
            "completion_tokens": gen.completion_tokens,
            "total_tokens": gen.total_tokens,
            "latency_seconds": gen.latency_seconds,
            "num_iterations": gen.num_iterations,
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    _print(f"  Saved {len(results)} results to {path.name}")


if __name__ == "__main__":
    main()
