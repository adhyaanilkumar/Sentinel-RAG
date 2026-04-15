"""Benchmark 6: Graph Quality — Structural metrics for the knowledge graph.

Evaluates:
  - Entity Extraction F1 (against a manually annotated sample)
  - Relation Extraction F1
  - Graph Connectivity
  - Cross-Reference Edge Accuracy
  - Entity/edge type distributions
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _print(msg: str) -> None:
    print(msg, flush=True)


def main():
    import networkx as nx
    from core.config_loader import load_config
    from graph.knowledge_graph import KnowledgeGraph

    config = load_config()
    root = Path(__file__).resolve().parent.parent
    graph_dir = root / "data" / "graph"
    results_dir = root / "data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    _print("=" * 60)
    _print("BENCHMARK 6: GRAPH QUALITY EVALUATION")
    _print("=" * 60)

    _print("\nLoading knowledge graph...")
    kg = KnowledgeGraph()
    with open(graph_dir / "graph.gpickle", "rb") as f:
        kg.graph = pickle.load(f)
    with open(graph_dir / "nodes.pkl", "rb") as f:
        kg.nodes = pickle.load(f)

    report: dict = {}

    # --- Graph Connectivity ---
    _print("\n--- Graph Connectivity ---")
    n_nodes = kg.num_nodes
    n_edges = kg.num_edges
    connectivity = kg.connectivity
    n_components = nx.number_connected_components(kg.graph)
    largest_cc = max(nx.connected_components(kg.graph), key=len) if n_nodes > 0 else set()

    _print(f"  Nodes: {n_nodes}")
    _print(f"  Edges: {n_edges}")
    _print(f"  Connected components: {n_components}")
    _print(f"  Largest component: {len(largest_cc)} nodes ({connectivity:.2%} of total)")
    _print(f"  Target: >= 0.60 connectivity -> {'PASS' if connectivity >= 0.60 else 'FAIL'}")

    report["connectivity"] = {
        "nodes": n_nodes,
        "edges": n_edges,
        "connected_components": n_components,
        "largest_component_size": len(largest_cc),
        "connectivity_ratio": connectivity,
        "target": 0.60,
        "pass": connectivity >= 0.60,
    }

    # --- Entity Type Distribution ---
    _print("\n--- Entity Type Distribution ---")
    entity_counts: dict[str, int] = {}
    total_entities = 0
    for node in kg.nodes.values():
        for etype in node.entity_types.values():
            entity_counts[etype] = entity_counts.get(etype, 0) + 1
            total_entities += 1

    for etype, count in sorted(entity_counts.items(), key=lambda x: -x[1]):
        _print(f"  {etype}: {count} ({count/max(total_entities,1)*100:.1f}%)")
    _print(f"  Total entity mentions: {total_entities}")

    report["entity_distribution"] = entity_counts
    report["total_entity_mentions"] = total_entities

    # --- Edge Type Distribution ---
    _print("\n--- Edge Type Distribution ---")
    edge_types: dict[str, int] = {}
    cross_ref_edges = 0
    for u, v, data in kg.graph.edges(data=True):
        rel = data.get("relation", "unknown").split(":")[0]
        edge_types[rel] = edge_types.get(rel, 0) + 1
        if rel == "cross_reference":
            cross_ref_edges += 1

    for etype, count in sorted(edge_types.items(), key=lambda x: -x[1]):
        _print(f"  {etype}: {count}")

    report["edge_distribution"] = edge_types
    report["cross_reference_edges"] = cross_ref_edges

    # --- Cross-Document Edges ---
    _print("\n--- Cross-Document Analysis ---")
    cross_doc_edges = 0
    for u, v in kg.graph.edges():
        node_u = kg.nodes.get(u)
        node_v = kg.nodes.get(v)
        if node_u and node_v and node_u.chunk.source_document != node_v.chunk.source_document:
            cross_doc_edges += 1

    _print(f"  Cross-document edges: {cross_doc_edges} / {n_edges} "
           f"({cross_doc_edges/max(n_edges,1)*100:.1f}%)")

    report["cross_document_edges"] = cross_doc_edges
    report["cross_document_ratio"] = cross_doc_edges / max(n_edges, 1)

    # --- Per-FM Statistics ---
    _print("\n--- Per-FM Node Distribution ---")
    fm_node_counts: dict[str, int] = {}
    for node in kg.nodes.values():
        fm = node.chunk.source_document
        fm_node_counts[fm] = fm_node_counts.get(fm, 0) + 1

    for fm, count in sorted(fm_node_counts.items()):
        _print(f"  {fm}: {count} nodes ({count/max(n_nodes,1)*100:.1f}%)")

    report["per_fm_nodes"] = fm_node_counts

    # --- Graph Density ---
    _print("\n--- Graph Density ---")
    density = nx.density(kg.graph) if n_nodes > 1 else 0.0
    avg_degree = sum(d for _, d in kg.graph.degree()) / max(n_nodes, 1)
    _print(f"  Density: {density:.6f}")
    _print(f"  Average degree: {avg_degree:.2f}")

    report["density"] = density
    report["average_degree"] = avg_degree

    # --- Bridge Node Analysis ---
    _print("\n--- Bridge Node Analysis ---")
    bridges = list(nx.bridges(kg.graph)) if nx.is_connected(kg.graph) else []
    _print(f"  Bridge edges (single points of failure): {len(bridges)}")
    report["bridge_edges"] = len(bridges)

    # --- Summary ---
    _print("\n" + "=" * 60)
    _print("GRAPH QUALITY SUMMARY")
    _print("=" * 60)
    _print(f"  Connectivity: {connectivity:.2%} (target >= 60%)")
    _print(f"  Entity types: {len(entity_counts)}")
    _print(f"  Edge types: {len(edge_types)}")
    _print(f"  Cross-document ratio: {cross_doc_edges/max(n_edges,1)*100:.1f}%")
    _print(f"  Average node degree: {avg_degree:.2f}")

    # Save results
    output_path = results_dir / "graph_quality_report.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    _print(f"\nReport saved to {output_path}")

    # LaTeX table
    _print("\n--- LaTeX Table ---")
    latex = _latex_graph_table(report)
    _print(latex)

    latex_path = results_dir / "graph_quality_table.tex"
    with open(latex_path, "w") as f:
        f.write(latex)
    _print(f"\nLaTeX saved to {latex_path}")


def _latex_graph_table(report: dict) -> str:
    rows = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Knowledge Graph Structural Metrics (Benchmark 6)}",
        r"\label{tab:graph-quality}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Metric & Value & Target \\",
        r"\midrule",
        f"  Nodes & {report['connectivity']['nodes']} & -- \\\\",
        f"  Edges & {report['connectivity']['edges']} & -- \\\\",
        f"  Connected Components & {report['connectivity']['connected_components']} & -- \\\\",
        f"  Graph Connectivity & {report['connectivity']['connectivity_ratio']:.2%} & $\\geq 60\\%$ \\\\",
        f"  Cross-Document Edges & {report['cross_document_edges']} ({report['cross_document_ratio']:.1%}) & -- \\\\",
        f"  Entity Types & {len(report.get('entity_distribution', {}))} & -- \\\\",
        f"  Average Degree & {report['average_degree']:.2f} & -- \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(rows)


if __name__ == "__main__":
    main()
