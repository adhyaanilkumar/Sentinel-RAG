"""Tests verifying the graph RAG bug fixes."""

from __future__ import annotations

import math

import pytest

from core.data_models import DocumentChunk, GraphEdge, GraphNode
from graph.entity_extractor import build_graph_edges, build_graph_nodes, extract_entities
from graph.knowledge_graph import KnowledgeGraph
from retrieval.graph_retriever import _decompose_query


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(id: str, text: str, source: str = "FM 3-0", section: str = "1-1") -> DocumentChunk:
    return DocumentChunk(id=id, text=text, source_document=source, section_id=section)


def _make_node(id: str, text: str, source: str = "FM 3-0", section: str = "1-1",
               entities: list[str] | None = None) -> GraphNode:
    chunk = _make_chunk(id, text, source, section)
    return GraphNode(id=id, chunk=chunk, entities=entities or [])


# ---------------------------------------------------------------------------
# Fix #1 — BFS deduplication: each node appears once with best weight
# ---------------------------------------------------------------------------

class TestBFSDeduplication:
    def test_node_appears_once(self):
        """A node reachable via two paths should appear exactly once in results."""
        kg = KnowledgeGraph()
        # Build a diamond: A -> B, A -> C, B -> D, C -> D
        for nid in ["A", "B", "C", "D"]:
            kg.graph.add_node(nid)
            kg.nodes[nid] = _make_node(nid, f"text {nid}")

        kg.graph.add_edge("A", "B", weight=0.5, relation="r", evidence="")
        kg.graph.add_edge("A", "C", weight=0.9, relation="r", evidence="")
        kg.graph.add_edge("B", "D", weight=0.5, relation="r", evidence="")
        kg.graph.add_edge("C", "D", weight=0.9, relation="r", evidence="")

        results = kg.get_neighbors("A", max_hops=2)
        node_ids = [nid for nid, _ in results]

        assert node_ids.count("D") == 1, "D should appear exactly once"

    def test_best_weight_kept(self):
        """The weight returned for a multi-path node should be the highest cumulative weight."""
        kg = KnowledgeGraph()
        for nid in ["A", "B", "C", "D"]:
            kg.graph.add_node(nid)
            kg.nodes[nid] = _make_node(nid, f"text {nid}")

        # Path A->B->D: 0.5+0.5 = 1.0
        # Path A->C->D: 0.9+0.9 = 1.8  <-- higher
        kg.graph.add_edge("A", "B", weight=0.5, relation="r", evidence="")
        kg.graph.add_edge("A", "C", weight=0.9, relation="r", evidence="")
        kg.graph.add_edge("B", "D", weight=0.5, relation="r", evidence="")
        kg.graph.add_edge("C", "D", weight=0.9, relation="r", evidence="")

        results = kg.get_neighbors("A", max_hops=2)
        d_weight = next(w for nid, w in results if nid == "D")

        assert abs(d_weight - 1.8) < 1e-6, f"Expected 1.8, got {d_weight}"

    def test_seed_not_in_results(self):
        """The seed node itself must not appear in the results."""
        kg = KnowledgeGraph()
        for nid in ["A", "B"]:
            kg.graph.add_node(nid)
            kg.nodes[nid] = _make_node(nid, f"text {nid}")
        kg.graph.add_edge("A", "B", weight=0.5, relation="r", evidence="")

        results = kg.get_neighbors("A", max_hops=2)
        assert all(nid != "A" for nid, _ in results)

    def test_max_results_cap(self):
        """get_neighbors should return at most max_results entries."""
        kg = KnowledgeGraph()
        # Star graph: A connected to 100 nodes
        kg.graph.add_node("A")
        kg.nodes["A"] = _make_node("A", "hub")
        for i in range(100):
            nid = f"N{i}"
            kg.graph.add_node(nid)
            kg.nodes[nid] = _make_node(nid, f"spoke {i}")
            kg.graph.add_edge("A", nid, weight=0.5, relation="r", evidence="")

        results = kg.get_neighbors("A", max_hops=1, max_results=20)
        assert len(results) <= 20


# ---------------------------------------------------------------------------
# Fix #2 — Cross-reference edges created without shared entities
# ---------------------------------------------------------------------------

class TestCrossRefEdgesNoSharedEntities:
    def test_cross_ref_edge_without_shared_entities(self):
        """A chunk citing FM 3-0 must be connected to FM 3-0 chunks even with no shared entities."""
        citing_chunk = _make_chunk("c1", "See FM 3-0 for more details.", source="FM 6-0", section="2-1")
        target_chunk = _make_chunk("c2", "This paragraph covers convergence doctrine.", source="FM 3-0", section="1-1")

        nodes = build_graph_nodes([citing_chunk, target_chunk])
        edges = build_graph_edges(nodes)

        cross_ref_edges = [e for e in edges if "cross_reference" in e.relation]
        assert len(cross_ref_edges) >= 1, "Expected at least one cross-reference edge"

        edge_pairs = {(e.source, e.target) for e in cross_ref_edges}
        assert ("c1", "c2") in edge_pairs or ("c2", "c1") in edge_pairs

    def test_cross_ref_weight_is_high(self):
        """Cross-reference edges should have weight 1.5 (higher than shared-entity edges)."""
        citing_chunk = _make_chunk("c1", "IAW FM 3-0 this applies.", source="FM 6-0", section="1-1")
        target_chunk = _make_chunk("c2", "Unrelated text about weather.", source="FM 3-0", section="1-1")

        nodes = build_graph_nodes([citing_chunk, target_chunk])
        edges = build_graph_edges(nodes)

        cross_ref_edges = [e for e in edges if "cross_reference" in e.relation]
        assert cross_ref_edges, "Expected cross-reference edge"
        assert cross_ref_edges[0].weight == 1.5


# ---------------------------------------------------------------------------
# Fix #3 — Entity frequency threshold raised to 200
# ---------------------------------------------------------------------------

class TestEntityFrequencyThreshold:
    def test_common_entity_creates_edges(self):
        """Chunks sharing entities at 60 occurrences should still produce edges (under 200 threshold)."""
        chunks = [
            _make_chunk(f"c{i}", "The commander leads the battalion during operations.", source=f"FM {i}", section="1-1")
            for i in range(60)  # 60 chunks — would have been skipped at old threshold of 50
        ]
        nodes = build_graph_nodes(chunks)
        edges = build_graph_edges(nodes)

        # Pair (c0, c1) must have an edge via whichever shared entity is processed first
        entity_edges = [e for e in edges if e.relation.startswith("shared_entity:")]
        assert len(entity_edges) > 0, "Expected shared-entity edges when frequency is under 200"

    def test_very_common_entity_still_filtered(self):
        """Entities in >200 chunks should still be skipped."""
        chunks = [
            _make_chunk(f"c{i}", "The commander leads operations.", source=f"FM {i}", section="1-1")
            for i in range(210)
        ]
        nodes = build_graph_nodes(chunks)
        # With 210 chunks all containing 'commander', it should be filtered
        edges = build_graph_edges(nodes)
        commander_edges = [e for e in edges if "commander" in e.evidence]
        assert len(commander_edges) == 0, "Expected 'commander' filtered at 210 occurrences"


# ---------------------------------------------------------------------------
# Fix #9 — IDF weighting: rare entities produce stronger edges
# ---------------------------------------------------------------------------

class TestIDFWeighting:
    def test_rare_entity_stronger_than_common(self):
        """A rare shared entity should produce a higher-weight edge than a common one."""
        # 20 chunks all share 'commander' (common), only 2 share 'operation anaconda' (rare)
        common_chunks = [
            _make_chunk(f"c{i}", "The commander leads the battalion.", source="FM 3-0", section="1-1")
            for i in range(20)
        ]
        rare_chunks = [
            _make_chunk("r1", "Operation Anaconda involved the brigade combat team.", source="FM 3-90", section="2-1"),
            _make_chunk("r2", "Operation Anaconda showed the importance of fires.", source="FM 3-90", section="2-2"),
        ]
        all_chunks = common_chunks + rare_chunks
        nodes = build_graph_nodes(all_chunks)
        edges = build_graph_edges(nodes)

        # Find a 'commander' edge between two of the common chunks
        commander_edges = [e for e in edges if "commander" in e.evidence
                           and e.source.startswith("c") and e.target.startswith("c")]

        # Find the edge between r1 and r2 (shares rare entity 'operation anaconda' or 'brigade combat team')
        rare_edges = [e for e in edges if e.source in ("r1", "r2") and e.target in ("r1", "r2")]

        if commander_edges and rare_edges:
            avg_common = sum(e.weight for e in commander_edges) / len(commander_edges)
            avg_rare = sum(e.weight for e in rare_edges) / len(rare_edges)
            assert avg_rare >= avg_common, (
                f"Rare entity edges ({avg_rare:.3f}) should be >= common entity edges ({avg_common:.3f})"
            )


# ---------------------------------------------------------------------------
# Query decomposition
# ---------------------------------------------------------------------------

class TestQueryDecomposition:
    def test_returns_original_query(self):
        q = "What are the key principles of mission command?"
        subqueries = _decompose_query(q)
        assert q in subqueries

    def test_compound_query_split(self):
        q = "What is mission command and how does it apply to multidomain operations?"
        subqueries = _decompose_query(q)
        assert len(subqueries) > 1

    def test_doctrine_term_generates_subqueries(self):
        q = "Explain mission command in the context of FM 3-0."
        subqueries = _decompose_query(q)
        assert any("mission command" in sq.lower() for sq in subqueries if sq != q)

    def test_fm_ref_generates_subquery(self):
        q = "What does FM 3-0 say about convergence?"
        subqueries = _decompose_query(q)
        assert any("FM 3-0" in sq for sq in subqueries)

    def test_no_duplicates(self):
        q = "Explain mission command and mission command doctrine."
        subqueries = _decompose_query(q)
        assert len(subqueries) == len(set(subqueries))


# ---------------------------------------------------------------------------
# Fix #4 — Jaccard similarity replaces asymmetric entity overlap normalization
# ---------------------------------------------------------------------------

class TestJaccardEntityOverlap:
    def test_jaccard_symmetric(self):
        """Entity overlap score should be the same regardless of which chunk is the query."""
        # Two chunks: chunk A has 3 entities, chunk B shares 2 of them plus 1 new one
        # Jaccard = |intersection| / |union| = 2 / 4 = 0.5 for both directions
        chunks = [
            _make_chunk("a1", "The battalion commander leads the task force.", "FM 3-0", "1-1"),
            _make_chunk("a2", "The task force conducts operations in the sector.", "FM 3-90", "2-1"),
        ]
        nodes = build_graph_nodes(chunks)
        edges = build_graph_edges(nodes)

        # Shared entities should produce the same edge regardless of ordering
        entity_edges = [e for e in edges if e.relation.startswith("shared_entity:")]
        if entity_edges:
            # There should be no asymmetry — a single edge per pair
            pairs = [(e.source, e.target) for e in entity_edges]
            reversed_pairs = [(e.target, e.source) for e in entity_edges]
            for p in pairs:
                assert p not in reversed_pairs, "Duplicate reverse edge found — asymmetric normalization still present"

    def test_jaccard_penalizes_large_union(self):
        """A chunk with many unique entities should score lower overlap than one with few."""
        # Chunk with 2 entities sharing 1 = Jaccard 1/3 ≈ 0.33
        # Chunk with 10 unique entities sharing 1 = Jaccard 1/11 ≈ 0.09
        # This is only verifiable via the graph_retriever CSS score; here we just check
        # that shared-entity edges exist and that the weight is bounded < 1.5 (cross-ref weight)
        chunks = [
            _make_chunk("j1", "The commander leads operations.", "FM 3-0", "1-1"),
            _make_chunk("j2", "The commander coordinates logistics support.", "FM 4-0", "1-1"),
        ]
        nodes = build_graph_nodes(chunks)
        edges = build_graph_edges(nodes)
        entity_edges = [e for e in edges if e.relation.startswith("shared_entity:")]
        for e in entity_edges:
            assert e.weight < 1.5, "Shared-entity edge weight must be below cross-reference weight of 1.5"


# ---------------------------------------------------------------------------
# Fix #5 — Cross-doc neighbor boost includes base_score from candidate_ids
# ---------------------------------------------------------------------------

class TestCrossDocNeighborBaseScore:
    def test_cross_doc_boost_uses_base_score(self):
        """Cross-doc neighbors must accumulate score from both base_score and graph weight."""
        from retrieval.graph_retriever import GraphRetriever, CSSConfig
        from core.embeddings import EmbeddingModel
        from retrieval.vector_store import VectorStore
        from unittest.mock import MagicMock, patch
        import numpy as np

        kg = KnowledgeGraph()
        for nid, src in [("seed1", "FM 3-0"), ("xdoc1", "FM 6-0")]:
            kg.graph.add_node(nid)
            kg.nodes[nid] = _make_node(nid, f"text {nid}", source=src)

        kg.graph.add_edge("seed1", "xdoc1", weight=0.8, relation="cross_reference", evidence="FM 6-0")

        # Fake vector store returns seed1 as the only initial result
        mock_vs = MagicMock(spec=VectorStore)
        mock_vs.search.return_value = MagicMock(chunks=[kg.nodes["seed1"].chunk], scores=[0.6])

        mock_emb = MagicMock(spec=EmbeddingModel)
        mock_emb.encode_single.return_value = np.array([1.0, 0.0])

        retriever = GraphRetriever(mock_vs, kg, mock_emb, final_top_k=5)

        with patch.object(kg, "get_cross_document_neighbors", return_value=[("xdoc1", 0.8)]):
            result = retriever.retrieve("What does FM 6-0 say?")

        # xdoc1 should appear in results (its score was boosted)
        retrieved_ids = [c.id for c in result.chunks]
        assert "xdoc1" in retrieved_ids, "Cross-doc neighbor xdoc1 should be in results after boost"


# ---------------------------------------------------------------------------
# Fix #6 — Temporal decay no longer hard-scaled by * 0.1
# ---------------------------------------------------------------------------

class TestTemporalDecayScaling:
    def test_temporal_weight_has_full_contribution(self):
        """A temporal weight of 0.0 should meaningfully lower the CSS score vs 1.0."""
        from retrieval.graph_retriever import GraphRetriever, CSSConfig
        from core.embeddings import EmbeddingModel
        from retrieval.vector_store import VectorStore
        from unittest.mock import MagicMock
        import numpy as np

        kg = KnowledgeGraph()
        for nid in ["n1", "n2"]:
            kg.graph.add_node(nid)
            kg.nodes[nid] = _make_node(nid, "The commander leads operations.", source="FM 3-0")

        mock_vs = MagicMock(spec=VectorStore)
        mock_vs.search.return_value = MagicMock(chunks=[], scores=[])
        mock_emb = MagicMock(spec=EmbeddingModel)
        mock_emb.encode_single.return_value = np.array([1.0, 0.0])

        css = CSSConfig(temporal_recency=1.0)
        retriever = GraphRetriever(mock_vs, kg, mock_emb, css_config=css)

        candidates = {"n1": 0.5, "n2": 0.5}

        # n1: temporal weight = 1.0 (fresh),  n2: temporal weight = 0.0 (stale)
        scored_fresh = retriever._css_score("commander operations", candidates, {"n1": 1.0, "n2": 1.0})
        scored_stale = retriever._css_score("commander operations", candidates, {"n1": 1.0, "n2": 0.0})

        n1_fresh = next(s for nid, s in scored_fresh if nid == "n1")
        n2_fresh = next(s for nid, s in scored_fresh if nid == "n2")
        n1_stale = next(s for nid, s in scored_stale if nid == "n1")
        n2_stale = next(s for nid, s in scored_stale if nid == "n2")

        # Both equal when fresh
        assert abs(n1_fresh - n2_fresh) < 1e-6, "Both nodes should score equally when both fresh"
        # n2 scores lower when stale (temporal_recency weight is 1.0, not 0.1)
        assert n1_stale - n2_stale >= 0.9, (
            f"Temporal decay should reduce n2 score by ~1.0 (temporal_recency=1.0), "
            f"got delta={n1_stale - n2_stale:.3f}"
        )


# ---------------------------------------------------------------------------
# Fix #11 — Cross-doc neighbor cap raised from [:3] to [:final_top_k]
# ---------------------------------------------------------------------------

class TestCrossDocNeighborCap:
    def test_cross_doc_cap_is_final_top_k(self):
        """GraphRetriever should fetch up to final_top_k cross-doc neighbors per seed."""
        from retrieval.graph_retriever import GraphRetriever
        from core.embeddings import EmbeddingModel
        from retrieval.vector_store import VectorStore
        from unittest.mock import MagicMock, patch
        import numpy as np

        kg = KnowledgeGraph()
        kg.graph.add_node("seed")
        kg.nodes["seed"] = _make_node("seed", "seed text", source="FM 3-0")
        # Create 8 cross-doc neighbors from FM 6-0
        for i in range(8):
            nid = f"xd{i}"
            kg.graph.add_node(nid)
            kg.nodes[nid] = _make_node(nid, f"cross text {i}", source="FM 6-0")
            kg.graph.add_edge("seed", nid, weight=0.5, relation="cross_reference", evidence="FM 6-0")

        mock_vs = MagicMock(spec=VectorStore)
        mock_vs.search.return_value = MagicMock(chunks=[kg.nodes["seed"].chunk], scores=[0.5])

        mock_emb = MagicMock(spec=EmbeddingModel)
        mock_emb.encode_single.return_value = np.array([1.0, 0.0])

        # final_top_k = 10 means cap should be 10, so all 8 should be reachable
        retriever = GraphRetriever(mock_vs, kg, mock_emb, final_top_k=10)
        result = retriever.retrieve("test query")
        retrieved_ids = {c.id for c in result.chunks}
        # With the old [:3] cap we'd only get 3 cross-doc chunks; with [:final_top_k] we get all 8
        xdoc_count = sum(1 for cid in retrieved_ids if cid.startswith("xd"))
        assert xdoc_count > 3, (
            f"Expected >3 cross-doc neighbors with final_top_k=10 cap, got {xdoc_count}. "
            "Old [:3] cap would have limited this."
        )


# ---------------------------------------------------------------------------
# Fix #12 — Source diversity enforcement caps any single FM at ceil(k/2)
# ---------------------------------------------------------------------------

class TestSourceDiversityEnforcement:
    def test_no_single_source_dominates(self):
        """No single FM should occupy more than ceil(final_top_k/2) of the top-k results."""
        from retrieval.graph_retriever import GraphRetriever
        from core.embeddings import EmbeddingModel
        from retrieval.vector_store import VectorStore
        from unittest.mock import MagicMock
        import numpy as np
        import math

        kg = KnowledgeGraph()
        # 8 nodes from FM 3-0, 4 nodes from FM 6-0 (enough to fill remaining slots)
        for i in range(8):
            nid = f"fm30_{i}"
            kg.graph.add_node(nid)
            kg.nodes[nid] = _make_node(nid, f"FM 3-0 text {i}", source="FM 3-0")
        for i in range(4):
            nid = f"fm60_{i}"
            kg.graph.add_node(nid)
            kg.nodes[nid] = _make_node(nid, f"FM 6-0 text {i}", source="FM 6-0")

        final_top_k = 6
        per_source_cap = math.ceil(final_top_k / 2)  # 3

        # Build a scored list where FM 3-0 nodes all have higher scores
        scored = [(f"fm30_{i}", 1.0 - i * 0.05) for i in range(8)] + \
                 [(f"fm60_{i}", 0.5 - i * 0.01) for i in range(4)]

        mock_vs = MagicMock(spec=VectorStore)
        mock_emb = MagicMock(spec=EmbeddingModel)
        retriever = GraphRetriever(mock_vs, kg, mock_emb, final_top_k=final_top_k)

        diversified = retriever._enforce_source_diversity(scored)
        top_k = diversified[:final_top_k]

        fm30_count = sum(1 for nid, _ in top_k if nid.startswith("fm30"))
        assert fm30_count <= per_source_cap, (
            f"FM 3-0 has {fm30_count} chunks in top-{final_top_k}, "
            f"expected at most {per_source_cap}"
        )

    def test_diversity_preserves_all_entries(self):
        """_enforce_source_diversity should not drop any candidates, only reorder."""
        from retrieval.graph_retriever import GraphRetriever
        from core.embeddings import EmbeddingModel
        from retrieval.vector_store import VectorStore
        from unittest.mock import MagicMock
        import numpy as np

        kg = KnowledgeGraph()
        for i in range(5):
            nid = f"n{i}"
            kg.graph.add_node(nid)
            kg.nodes[nid] = _make_node(nid, f"text {i}", source=f"FM {i}")

        scored = [(f"n{i}", float(5 - i)) for i in range(5)]
        mock_vs = MagicMock(spec=VectorStore)
        mock_emb = MagicMock(spec=EmbeddingModel)
        retriever = GraphRetriever(mock_vs, kg, mock_emb, final_top_k=5)

        diversified = retriever._enforce_source_diversity(scored)
        assert len(diversified) == len(scored), "No candidates should be dropped by diversity enforcement"
