"""Knowledge graph built from document chunks using NetworkX."""

from __future__ import annotations

import pickle
from pathlib import Path

import networkx as nx

from core.data_models import DocumentChunk, GraphEdge, GraphNode
from graph.entity_extractor import build_graph_edges, build_graph_nodes


class KnowledgeGraph:
    """Military doctrine knowledge graph for multi-hop retrieval."""

    def __init__(self):
        self.graph = nx.Graph()
        self.nodes: dict[str, GraphNode] = {}

    def build(self, chunks: list[DocumentChunk], graph_config: dict | None = None) -> None:
        nodes = build_graph_nodes(chunks)
        edges = build_graph_edges(nodes, graph_config)

        for node in nodes:
            self.nodes[node.id] = node
            self.graph.add_node(node.id, source=node.chunk.source_document,
                                section=node.chunk.section_id,
                                entity_count=len(node.entities))

        for edge in edges:
            if self.graph.has_edge(edge.source, edge.target):
                existing = self.graph[edge.source][edge.target]["weight"]
                self.graph[edge.source][edge.target]["weight"] = max(existing, edge.weight)
            else:
                self.graph.add_edge(edge.source, edge.target,
                                    weight=edge.weight, relation=edge.relation,
                                    evidence=edge.evidence)

    @property
    def num_nodes(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self.graph.number_of_edges()

    @property
    def connectivity(self) -> float:
        if self.num_nodes == 0:
            return 0.0
        largest_cc = max(nx.connected_components(self.graph), key=len)
        return len(largest_cc) / self.num_nodes

    def get_node(self, node_id: str) -> GraphNode | None:
        return self.nodes.get(node_id)

    def get_neighbors(self, node_id: str, max_hops: int = 2, max_results: int = 50) -> list[tuple[str, float]]:
        """BFS-style neighbor retrieval up to max_hops, returning (node_id, cumulative_weight).

        max_results caps the returned list to the top-N by cumulative weight, preventing
        combinatorial explosion on dense graphs with many cross-reference edges.
        """
        if node_id not in self.graph:
            return []

        visited: dict[str, float] = {node_id: 0.0}
        frontier = [(node_id, 0.0, 0)]

        while frontier:
            current, cum_weight, hops = frontier.pop(0)
            if hops >= max_hops:
                continue
            for neighbor in self.graph.neighbors(current):
                edge_weight = self.graph[current][neighbor].get("weight", 0.5)
                new_weight = cum_weight + edge_weight
                if neighbor not in visited or new_weight > visited[neighbor]:
                    visited[neighbor] = new_weight
                    frontier.append((neighbor, new_weight, hops + 1))

        results = [
            (nid, w) for nid, w in visited.items() if nid != node_id
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:max_results]

    def get_cross_document_neighbors(self, node_id: str, max_hops: int = 3) -> list[tuple[str, float]]:
        """Get neighbors that come from a different source document."""
        node = self.get_node(node_id)
        if not node:
            return []
        src_doc = node.chunk.source_document
        neighbors = self.get_neighbors(node_id, max_hops=max_hops)
        return [(nid, w) for nid, w in neighbors
                if self.nodes[nid].chunk.source_document != src_doc]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        nx.write_gpickle(self.graph, str(path / "graph.gpickle"))
        with open(path / "nodes.pkl", "wb") as f:
            pickle.dump(self.nodes, f)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        self.graph = nx.read_gpickle(str(path / "graph.gpickle"))
        with open(path / "nodes.pkl", "rb") as f:
            self.nodes = pickle.load(f)
