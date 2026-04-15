"""Load configuration from config.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class EmbeddingConfig(BaseModel):
    model_name: str = "all-MiniLM-L6-v2"
    dimension: int = 384


class RetrievalConfig(BaseModel):
    top_k: int = 10
    similarity_threshold: float = 0.3
    chunk_size: int = 512
    chunk_overlap: int = 64


class GraphConfig(BaseModel):
    cross_ref_patterns: list[str] = Field(default_factory=list)
    entity_types: list[str] = Field(default_factory=list)
    adjacency_same_section_bonus: float = 1.0
    adjacency_cross_section_factor: float = 0.5
    bridge_node_protection_threshold: int = 1


class CSSWeights(BaseModel):
    relevance: float = 2.0
    context_cohesion: float = 0.4
    subquery_coverage: float = 1.5
    cross_ref_bonus: float = 1.2
    entity_overlap: float = 0.8
    temporal_recency: float = 1.0


class CSSConfig(BaseModel):
    weights: CSSWeights = Field(default_factory=CSSWeights)
    token_budget: int = 3000
    redundancy_threshold: float = 0.92


class TemporalConfig(BaseModel):
    decay_function: str = "exponential"
    half_life_hours: float = 72
    stale_threshold_hours: float = 168
    flag_stale: bool = True


class EvaluationConfig(BaseModel):
    significance_level: float = 0.05
    min_queries: int = 40
    ragas_metrics: list[str] = Field(default_factory=lambda: [
        "context_recall", "context_precision", "faithfulness",
        "answer_correctness", "answer_relevancy",
    ])


class AppConfig(BaseModel):
    llm_provider: str = "openai"
    openai_model: str = "gpt-4o-mini"
    gemini_model: str = "gemini-2.0-flash"
    domain: str = "military"
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    css: CSSConfig = Field(default_factory=CSSConfig)
    temporal: TemporalConfig = Field(default_factory=TemporalConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)


def load_config(config_path: str | Path | None = None) -> AppConfig:
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    config_path = Path(config_path)
    if not config_path.exists():
        return AppConfig()
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}
    return AppConfig(**raw)
