"""Pydantic data models for Sentinel-RAG."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class QueryCategory(str, Enum):
    TRAP_A = "trap_a_overriding_directive"
    TRAP_B = "trap_b_distant_definition"
    TRAP_C = "trap_c_scattered_components"
    CONTROL = "control_single_hop"
    TEMPORAL = "temporal"


class HopCount(int, Enum):
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4


class GoldAnnotation(BaseModel):
    """A single gold-standard query-answer pair for evaluation."""

    id: str
    query: str
    ground_truth_answer: str
    section_references: list[str] = Field(
        description="FM sections that contain the answer components"
    )
    information_units: list[str] = Field(
        description="Atomic facts needed for a complete answer (the checklist)"
    )
    source_documents: list[str] = Field(
        description="Which Field Manuals are involved"
    )
    category: QueryCategory
    hop_count: HopCount
    difficulty: str = "medium"


class DocumentChunk(BaseModel):
    """A chunk of text from a military document."""

    id: str
    text: str
    source_document: str
    section_id: str
    section_title: str = ""
    page_number: int = 0
    timestamp: Optional[datetime] = None
    metadata: dict = Field(default_factory=dict)


class GraphNode(BaseModel):
    """A node in the knowledge graph, representing a document chunk."""

    id: str
    chunk: DocumentChunk
    entities: list[str] = Field(default_factory=list)
    entity_types: dict[str, str] = Field(default_factory=dict)
    relevance_score: float = 0.0
    temporal_weight: float = 1.0


class GraphEdge(BaseModel):
    """An edge in the knowledge graph."""

    source: str
    target: str
    relation: str
    weight: float = 1.0
    evidence: str = ""


class RetrievalResult(BaseModel):
    """Result from a retrieval operation."""

    chunks: list[DocumentChunk]
    scores: list[float]
    nodes_used: list[str] = Field(default_factory=list)
    edges_traversed: list[str] = Field(default_factory=list)
    retrieval_method: str = "unknown"
    latency_seconds: float = 0.0
    token_count: int = 0


class GenerationResult(BaseModel):
    """Result from the generation pipeline."""

    answer: str
    retrieved_context: str
    retrieval_result: RetrievalResult
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_seconds: float = 0.0
    num_iterations: int = 1
    model_name: str = ""


class BenchmarkResult(BaseModel):
    """Result of a single benchmark evaluation."""

    query_id: str
    category: QueryCategory
    hop_count: int
    system_name: str
    generation_result: GenerationResult
    context_recall: float = 0.0
    context_precision: float = 0.0
    evidence_recall: float = 0.0
    mrr_at_10: float = 0.0
    faithfulness: float = 0.0
    answer_correctness: float = 0.0
    answer_relevancy: float = 0.0
    rouge_l: float = 0.0
    information_unit_coverage: float = 0.0
    fatal_error: bool = False
    definition_retrieved: bool = False
    component_recall: float = 0.0
    temporal_correctness: float = 0.0
    stale_info_flagged: bool = False
