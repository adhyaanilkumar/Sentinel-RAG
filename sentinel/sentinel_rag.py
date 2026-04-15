"""Sentinel-RAG: Adaptive Multi-Hop Reasoning for Military Situational Awareness."""

from __future__ import annotations

import time
from typing import Optional

from core.data_models import GenerationResult, RetrievalResult
from generation.llm_client import LLMClient
from generation.prompt_templates import SENTINEL_RAG_SYSTEM, SENTINEL_RAG_USER
from graph.knowledge_graph import KnowledgeGraph
from retrieval.graph_retriever import GraphRetriever
from sentinel.temporal_decay import TemporalDecayEngine


class SentinelRAG:
    """Single-pass graph-augmented RAG with temporal awareness."""

    def __init__(
        self,
        graph_retriever: GraphRetriever,
        knowledge_graph: KnowledgeGraph,
        llm: LLMClient,
        temporal_engine: Optional[TemporalDecayEngine] = None,
        top_k: int = 10,
    ):
        self.retriever = graph_retriever
        self.kg = knowledge_graph
        self.llm = llm
        self.temporal_engine = temporal_engine
        self.top_k = top_k

    def query(self, question: str, enable_temporal: bool = True) -> GenerationResult:
        start = time.time()

        temporal_weights = None
        if enable_temporal and self.temporal_engine:
            temporal_weights = self.temporal_engine.compute_weights(self.kg)

        retrieval_result = self.retriever.retrieve(question, temporal_weights=temporal_weights)

        context = self._format_context(retrieval_result)
        cross_ref_notes = self._build_cross_ref_notes(retrieval_result)

        stale_warnings = ""
        if enable_temporal and self.temporal_engine:
            stale_warnings = self.temporal_engine.flag_stale_chunks(retrieval_result.chunks)
            if stale_warnings:
                cross_ref_notes += f"\n\nTemporal Warnings:\n{stale_warnings}"

        prompt = SENTINEL_RAG_USER.format(
            context=context, cross_ref_notes=cross_ref_notes, query=question,
        )
        llm_resp = self.llm.generate(prompt, system_prompt=SENTINEL_RAG_SYSTEM)

        total_latency = time.time() - start
        return GenerationResult(
            answer=llm_resp.text,
            retrieved_context=context,
            retrieval_result=retrieval_result,
            prompt_tokens=llm_resp.prompt_tokens,
            completion_tokens=llm_resp.completion_tokens,
            total_tokens=llm_resp.total_tokens,
            latency_seconds=total_latency,
            num_iterations=1,
            model_name=llm_resp.model,
        )

    @staticmethod
    def _format_context(retrieval: RetrievalResult) -> str:
        parts = []
        for i, (chunk, score) in enumerate(zip(retrieval.chunks, retrieval.scores)):
            parts.append(
                f"[Source {i+1}: {chunk.source_document}, {chunk.section_id} "
                f"(relevance={score:.3f})]\n{chunk.text}"
            )
        return "\n\n".join(parts)

    def _build_cross_ref_notes(self, retrieval: RetrievalResult) -> str:
        docs = set()
        sections = set()
        for chunk in retrieval.chunks:
            docs.add(chunk.source_document)
            sections.add(f"{chunk.source_document} {chunk.section_id}")

        notes = []
        if len(docs) > 1:
            notes.append(f"Cross-document context from: {', '.join(sorted(docs))}")
        if retrieval.edges_traversed:
            cross_doc_edges = [e for e in retrieval.edges_traversed if "xdoc" in e]
            if cross_doc_edges:
                notes.append(f"Cross-reference edges followed: {len(cross_doc_edges)}")
        notes.append(f"Graph nodes explored: {len(retrieval.nodes_used)}")
        return "\n".join(notes) if notes else "Single-document retrieval."
