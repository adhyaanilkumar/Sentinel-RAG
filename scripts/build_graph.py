"""Build the knowledge graph from the FM corpus."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from core.config_loader import load_config
from core.document_processor import load_corpus
from graph.knowledge_graph import KnowledgeGraph


def main():
    config = load_config()
    corpus_dir = Path(__file__).resolve().parent.parent / "data" / "corpus"
    graph_dir = Path(__file__).resolve().parent.parent / "data" / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)

    print("Loading corpus...")
    chunks = load_corpus(
        corpus_dir,
        chunk_size=config.retrieval.chunk_size,
        chunk_overlap=config.retrieval.chunk_overlap,
    )

    print(f"\nBuilding knowledge graph from {len(chunks)} chunks...")
    kg = KnowledgeGraph()
    graph_cfg = {
        "adjacency_same_section_bonus": config.graph.adjacency_same_section_bonus,
        "adjacency_cross_section_factor": config.graph.adjacency_cross_section_factor,
    }
    kg.build(chunks, graph_cfg)

    print(f"Graph: {kg.num_nodes} nodes, {kg.num_edges} edges")
    print(f"Connectivity: {kg.connectivity:.2%} of nodes in largest component")

    entity_counts = {}
    for node in kg.nodes.values():
        for etype in node.entity_types.values():
            entity_counts[etype] = entity_counts.get(etype, 0) + 1

    print("\nEntity type distribution:")
    for etype, count in sorted(entity_counts.items(), key=lambda x: -x[1]):
        print(f"  {etype}: {count}")

    edge_types = {}
    for u, v, data in kg.graph.edges(data=True):
        rel = data.get("relation", "unknown").split(":")[0]
        edge_types[rel] = edge_types.get(rel, 0) + 1

    print("\nEdge type distribution:")
    for etype, count in sorted(edge_types.items(), key=lambda x: -x[1]):
        print(f"  {etype}: {count}")

    print(f"\nSaving graph to {graph_dir}")
    with open(graph_dir / "graph.gpickle", "wb") as f:
        pickle.dump(kg.graph, f)
    with open(graph_dir / "nodes.pkl", "wb") as f:
        pickle.dump(kg.nodes, f)
    print("Done.")


if __name__ == "__main__":
    main()
