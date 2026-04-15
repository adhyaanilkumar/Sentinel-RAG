"""Graph-augmented retrieval with CSS-style optimization for Sentinel-RAG."""

from __future__ import annotations

import re
import time
from typing import Optional

import numpy as np

from core.data_models import DocumentChunk, RetrievalResult
from core.embeddings import EmbeddingModel
from graph.knowledge_graph import KnowledgeGraph
from retrieval.vector_store import VectorStore


def _decompose_query(query: str) -> list[str]:
    """Rule-based query decomposition into sub-questions for subquery coverage scoring."""
    subqueries = [query]

    # Split compound questions joined by "and" or ";"
    parts = re.split(r"\band\b|;", query, flags=re.IGNORECASE)
    if len(parts) > 1:
        subqueries = [p.strip() for p in parts if len(p.strip()) > 10]

    # Generate definition/requirement subqueries for key doctrine terms
    terms = re.findall(
        r"\b(?:mission command|multidomain operations|convergence|relative advantage|"
        r"operational reach|decisive point|defeat mechanism|combat power|"
        r"warfighting function|area of operations|main effort|reserve)\b",
        query, re.IGNORECASE,
    )
    for term in terms:
        subqueries.append(f"What is {term}?")
        subqueries.append(f"What are the requirements for {term}?")

    # Extract FM/doctrine cross-refs and generate subqueries for them
    fm_refs = re.findall(r"(?:FM|ATP|ADP)\s+[\d\-]+", query, re.IGNORECASE)
    for ref in fm_refs:
        subqueries.append(f"What does {ref} say about this topic?")

    return list(dict.fromkeys(subqueries))  # deduplicate, preserve order


class CSSConfig:
    """Constitutional Search Space optimization weights."""

    def __init__(
        self,
        relevance: float = 2.0,
        context_cohesion: float = 0.4,
        subquery_coverage: float = 1.5,
        cross_ref_bonus: float = 1.2,
        entity_overlap: float = 0.8,
        temporal_recency: float = 1.0,
        token_budget: int = 3000,
        redundancy_threshold: float = 0.92,
    ):
        self.relevance = relevance
        self.context_cohesion = context_cohesion
        self.subquery_coverage = subquery_coverage
        self.cross_ref_bonus = cross_ref_bonus
        self.entity_overlap = entity_overlap
        self.temporal_recency = temporal_recency
        self.token_budget = token_budget
        self.redundancy_threshold = redundancy_threshold


class GraphRetriever:
    """Two-stage retriever: vector search + graph expansion + CSS optimization."""

    def __init__(
        self,
        vector_store: VectorStore,
        knowledge_graph: KnowledgeGraph,
        embedding_model: EmbeddingModel,
        css_config: Optional[CSSConfig] = None,
        initial_top_k: int = 15,
        max_hops: int = 2,
        final_top_k: int = 10,
    ):
        self.vector_store = vector_store
        self.kg = knowledge_graph
        self.embedding_model = embedding_model
        self.css = css_config or CSSConfig()
        self.initial_top_k = initial_top_k
        self.max_hops = max_hops
        self.final_top_k = final_top_k

    def retrieve(self, query: str, temporal_weights: dict[str, float] | None = None) -> RetrievalResult:
        start = time.time()

        initial = self.vector_store.search(query, top_k=self.initial_top_k)
        seed_ids = [c.id for c in initial.chunks]
        seed_scores = dict(zip(seed_ids, initial.scores))

        candidate_ids: dict[str, float] = {}
        for cid in seed_ids:
            candidate_ids[cid] = seed_scores[cid]

        nodes_used = list(seed_ids)
        edges_traversed = []

        for seed_id in seed_ids:
            neighbors = self.kg.get_neighbors(seed_id, max_hops=self.max_hops)
            for nid, graph_weight in neighbors:
                if nid not in candidate_ids:
                    nodes_used.append(nid)
                    edges_traversed.append(f"{seed_id}->{nid}")
                base_score = candidate_ids.get(nid, 0.0)
                boosted = base_score + graph_weight * self.css.context_cohesion
                candidate_ids[nid] = max(candidate_ids.get(nid, 0.0), boosted)

            cross_doc = self.kg.get_cross_document_neighbors(seed_id, max_hops=self.max_hops + 1)
            for nid, graph_weight in cross_doc[:self.final_top_k]:
                if nid not in candidate_ids:
                    nodes_used.append(nid)
                    edges_traversed.append(f"{seed_id}->xdoc->{nid}")
                base_score = candidate_ids.get(nid, 0.0)
                boosted = base_score + graph_weight * self.css.cross_ref_bonus
                candidate_ids[nid] = max(candidate_ids.get(nid, 0.0), boosted)

        scored = self._css_score(query, candidate_ids, temporal_weights)

        scored = self._remove_redundant(scored)

        scored = self._enforce_source_diversity(scored)

        final = scored[: self.final_top_k]

        chunks = []
        scores = []
        for nid, score in final:
            node = self.kg.get_node(nid)
            if node:
                chunks.append(node.chunk)
                scores.append(score)

        latency = time.time() - start
        return RetrievalResult(
            chunks=chunks,
            scores=scores,
            nodes_used=nodes_used,
            edges_traversed=edges_traversed,
            retrieval_method="sentinel_graph_css",
            latency_seconds=latency,
        )

    def _css_score(
        self,
        query: str,
        candidate_ids: dict[str, float],
        temporal_weights: dict[str, float] | None = None,
    ) -> list[tuple[str, float]]:
        """Score candidates using CSS-style multi-factor optimization."""
        from graph.entity_extractor import extract_entities
        ents, _ = extract_entities(query)
        query_entities = set(ents)

        subqueries = _decompose_query(query)

        # Resolve valid (nid, node) pairs once — avoids repeated kg lookups
        valid = [(nid, base, self.kg.get_node(nid))
                 for nid, base in candidate_ids.items()]
        valid = [(nid, base, node) for nid, base, node in valid if node]

        if not valid:
            return []

        # Batch-encode query, subqueries, and all candidate chunks in three calls
        # instead of one encode_single() per candidate (was the 368s/query bottleneck).
        all_texts = [query] + subqueries + [node.chunk.text for _, _, node in valid]
        all_vecs = self.embedding_model.encode(all_texts)

        query_vec = all_vecs[0]
        subquery_vecs = all_vecs[1: 1 + len(subqueries)]
        chunk_vecs = all_vecs[1 + len(subqueries):]   # one vector per valid candidate

        scored = []
        for i, (nid, base_score, node) in enumerate(valid):
            chunk_vec = chunk_vecs[i]

            relevance = float(np.dot(query_vec, chunk_vec))

            # Subquery coverage: avg cosine similarity across decomposed subqueries
            subquery_coverage = float(np.mean(subquery_vecs @ chunk_vec))

            shared_entities = query_entities & set(node.entities)
            union_entities = query_entities | set(node.entities)
            entity_score = len(shared_entities) / max(len(union_entities), 1)

            is_cross_ref = any(
                self.kg.graph[nid].get(neighbor, {}).get("relation", "").startswith("cross_reference")
                for neighbor in self.kg.graph.neighbors(nid)
                if self.kg.graph.has_edge(nid, neighbor)
            ) if nid in self.kg.graph else False

            temporal_w = 1.0
            if temporal_weights and nid in temporal_weights:
                temporal_w = temporal_weights[nid]

            final_score = (
                relevance * self.css.relevance
                + subquery_coverage * self.css.subquery_coverage
                + base_score * self.css.context_cohesion
                + entity_score * self.css.entity_overlap
                + (self.css.cross_ref_bonus if is_cross_ref else 0.0)
                + temporal_w * self.css.temporal_recency
            )
            scored.append((nid, final_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _enforce_source_diversity(self, scored: list[tuple[str, float]]) -> list[tuple[str, float]]:
        """Re-rank to ensure no single source document dominates the top-k results.

        Allows at most ceil(final_top_k / 2) chunks from any one FM in the top slice,
        pushing excess same-source chunks to the back so other FMs get representation.
        """
        import math
        per_source_cap = math.ceil(self.final_top_k / 2)
        source_counts: dict[str, int] = {}
        capped: list[tuple[str, float]] = []
        deferred: list[tuple[str, float]] = []

        for nid, score in scored:
            node = self.kg.get_node(nid)
            src = node.chunk.source_document if node else "__unknown__"
            if source_counts.get(src, 0) < per_source_cap:
                source_counts[src] = source_counts.get(src, 0) + 1
                capped.append((nid, score))
            else:
                deferred.append((nid, score))

        return capped + deferred

    def _remove_redundant(self, scored: list[tuple[str, float]]) -> list[tuple[str, float]]:
        """Remove near-duplicate chunks above redundancy threshold."""
        if not scored:
            return scored

        kept = [scored[0]]
        kept_texts = [self.kg.get_node(scored[0][0]).chunk.text if self.kg.get_node(scored[0][0]) else ""]

        for nid, score in scored[1:]:
            node = self.kg.get_node(nid)
            if not node:
                continue
            text = node.chunk.text
            is_redundant = False
            for kt in kept_texts:
                overlap = len(set(text.split()) & set(kt.split())) / max(len(set(text.split())), 1)
                if overlap > self.css.redundancy_threshold:
                    is_redundant = True
                    break
            if not is_redundant:
                kept.append((nid, score))
                kept_texts.append(text)

        return kept
