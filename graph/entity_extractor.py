"""Military entity and relation extraction for knowledge graph construction."""

from __future__ import annotations

import math
import re
from typing import Optional

from core.data_models import DocumentChunk, GraphEdge, GraphNode


MILITARY_ENTITY_PATTERNS: dict[str, list[str]] = {
    "unit": [
        r"\b(?:BCT|brigade combat team|division|corps|battalion|company|platoon|squad|"
        r"theater army|field army|ARFOR|CFLCC|JTF)\b",
    ],
    "equipment": [
        r"\b(?:UAS|unmanned (?:aircraft|aerial) system|MLRS|artillery|"
        r"air defense|AMD|IADS|integrated air defense|"
        r"M1|Bradley|Stryker|HIMARS|Patriot)\b",
    ],
    "operation": [
        r"\b(?:movement to contact|attack|exploitation|pursuit|"
        r"area defense|mobile defense|retrograde|delay|withdrawal|retirement|"
        r"reconnaissance|security operations|passage of lines|relief in place|"
        r"screen|guard|cover|linkup|troop movement|breach)\b",
    ],
    "doctrine_term": [
        r"\b(?:mission command|multidomain operations|convergence|"
        r"relative advantage|operational reach|endurance|"
        r"decisive point|defeat mechanism|combat power|"
        r"warfighting function|operations process|"
        r"area of operations|zone|sector|"
        r"main effort|supporting effort|reserve|"
        r"deep operations|close operations|rear operations|"
        r"MDMP|military decision.making process|"
        r"IPOE|intelligence preparation|"
        r"common operational picture|COP|"
        r"common intelligence picture)\b",
    ],
    "personnel_role": [
        r"\b(?:commander|G-[1-9]|S-[1-9]|chief of staff|"
        r"liaison officer|sergeant major|XO|second in command)\b",
    ],
    "location": [
        r"\b(?:support area|deep area|close area|rear area|"
        r"battle position|assembly area|forward line of (?:own )?troops|FLOT|FEBA|"
        r"main supply route|MSR|line of communication|LOC|"
        r"joint security area|JSA|strategic support area)\b",
    ],
    "threat": [
        r"\b(?:peer threat|adversary|enemy|A2/AD|antiaccess|area denial|"
        r"systems warfare|preclusion|isolation|sanctuary)\b",
    ],
}

FM_CROSS_REF_PATTERNS = [
    r"(?:see|refer to|per|pursuant to|in accordance with|as described in|as defined in|IAW)\s+"
    r"(?:FM|ATP|ADP|AR|TC|JP|ADRP)\s+[\d\-]+(?:[,\.]\s*(?:Chapter|Section|Paragraph|Para|Table|Figure|Appendix)\s+[\w\d\-\.]+)?",
    r"(?:see|refer to)\s+(?:Chapter|Section|Paragraph|Para|Annex|Appendix|Table|Figure)\s+[\w\d\-\.]+",
    r"(?:as (?:defined|described|discussed|outlined) in)\s+(?:Chapter|Section|Paragraph)\s+[\w\d\-\.]+",
    r"\((?:FM|ATP|ADP|AR|JP)\s+[\d\-]+\)",
]


def extract_entities(text: str) -> tuple[list[str], dict[str, str]]:
    """Extract military entities from text. Returns (entity_list, entity_type_map)."""
    entities = []
    entity_types = {}
    for etype, patterns in MILITARY_ENTITY_PATTERNS.items():
        for pattern in patterns:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                ent = m.group(0).strip().lower()
                if ent not in entity_types:
                    entities.append(ent)
                    entity_types[ent] = etype
    return entities, entity_types


def extract_cross_references(text: str) -> list[str]:
    """Extract cross-reference mentions (e.g., 'See FM 3-0', 'per ADP 6-0')."""
    refs = []
    for pattern in FM_CROSS_REF_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            refs.append(m.group(0).strip())
    return refs


def build_graph_nodes(chunks: list[DocumentChunk]) -> list[GraphNode]:
    """Convert document chunks into graph nodes with extracted entities."""
    nodes = []
    for chunk in chunks:
        entities, entity_types = extract_entities(chunk.text)
        node = GraphNode(
            id=chunk.id,
            chunk=chunk,
            entities=entities,
            entity_types=entity_types,
        )
        nodes.append(node)
    return nodes


def build_graph_edges(nodes: list[GraphNode], config: Optional[dict] = None) -> list[GraphEdge]:
    """Build edges between graph nodes based on shared entities, cross-references, and adjacency."""
    cfg = config or {}
    same_section_bonus = cfg.get("adjacency_same_section_bonus", 1.0)
    cross_section_factor = cfg.get("adjacency_cross_section_factor", 0.5)
    entity_max_freq = cfg.get("entity_max_chunk_frequency", 200)

    node_map = {n.id: n for n in nodes}
    entity_to_nodes: dict[str, list[str]] = {}
    for node in nodes:
        for ent in node.entities:
            entity_to_nodes.setdefault(ent, []).append(node.id)

    # IDF: entities appearing in fewer chunks are more specific/discriminative
    total_nodes = max(len(nodes), 1)
    entity_idf: dict[str, float] = {
        ent: math.log(total_nodes / len(nids))
        for ent, nids in entity_to_nodes.items()
    }
    max_idf = max(entity_idf.values()) if entity_idf else 1.0
    max_idf = max(max_idf, 1e-9)  # guard against all-zero IDF (all entities in every chunk)

    edges: list[GraphEdge] = []
    edge_set: set[tuple[str, str]] = set()

    for ent, nids in entity_to_nodes.items():
        if len(nids) > entity_max_freq:
            continue
        idf_weight = entity_idf[ent] / max_idf  # normalized to [0, 1]
        for i, src_id in enumerate(nids):
            for tgt_id in nids[i + 1:]:
                pair = (min(src_id, tgt_id), max(src_id, tgt_id))
                if pair in edge_set:
                    continue
                edge_set.add(pair)
                src, tgt = node_map[src_id], node_map[tgt_id]
                if src.chunk.source_document == tgt.chunk.source_document and src.chunk.section_id == tgt.chunk.section_id:
                    base_weight = same_section_bonus
                elif src.chunk.source_document == tgt.chunk.source_document:
                    base_weight = cross_section_factor
                else:
                    base_weight = cross_section_factor * 0.8
                weight = base_weight * (0.5 + 0.5 * idf_weight)  # scale by IDF, min 50% of base
                edges.append(GraphEdge(
                    source=src_id, target=tgt_id,
                    relation=f"shared_entity:{ent}",
                    weight=weight, evidence=ent,
                ))

    for node in nodes:
        cross_refs = extract_cross_references(node.chunk.text)
        for ref in cross_refs:
            fm_match = re.search(r"(FM|ATP|ADP|AR|JP)\s+([\d\-]+)", ref, re.IGNORECASE)
            if not fm_match:
                continue
            target_fm = f"{fm_match.group(1).upper()} {fm_match.group(2)}"
            for other in nodes:
                if other.id == node.id:
                    continue
                if other.chunk.source_document == target_fm:
                    pair = (min(node.id, other.id), max(node.id, other.id))
                    if pair not in edge_set:
                        edge_set.add(pair)
                        edges.append(GraphEdge(
                            source=node.id, target=other.id,
                            relation=f"cross_reference:{ref}",
                            weight=1.5,
                            evidence=ref,
                        ))

    for i in range(len(nodes) - 1):
        src, tgt = nodes[i], nodes[i + 1]
        if src.chunk.source_document == tgt.chunk.source_document:
            pair = (src.id, tgt.id)
            if pair not in edge_set:
                edge_set.add(pair)
                edges.append(GraphEdge(
                    source=src.id, target=tgt.id,
                    relation="adjacency",
                    weight=0.3 if src.chunk.section_id == tgt.chunk.section_id else 0.1,
                    evidence="sequential_chunks",
                ))

    return edges
