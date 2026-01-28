"""
Shared dependencies for API endpoints.

Provides dependency injection for services used across routes.
"""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from app.config import Settings, get_settings
from app.core.rag_engine import RAGEngine
from app.core.llm import LLMService
from app.core.vision import VisionProcessor
from app.core.cache import ResponseCache
from app.ingestion.embedder import EmbeddingService


# Settings dependency
SettingsDep = Annotated[Settings, Depends(get_settings)]


@lru_cache
def get_cache() -> ResponseCache:
    """Get cached ResponseCache instance."""
    settings = get_settings()
    if settings.CACHE_RESPONSES:
        return ResponseCache(db_path=settings.CACHE_DB_PATH)
    return None


@lru_cache
def get_embedding_service() -> EmbeddingService:
    """Get cached EmbeddingService instance."""
    settings = get_settings()
    return EmbeddingService(
        api_key=settings.OPENAI_API_KEY,
        model=settings.OPENAI_EMBEDDING_MODEL,
        mock_mode=settings.MOCK_MODE,
    )


@lru_cache
def get_rag_engine() -> RAGEngine:
    """Get cached RAGEngine instance."""
    settings = get_settings()
    embedding_service = get_embedding_service()
    return RAGEngine(
        persist_dir=settings.CHROMA_PERSIST_DIR,
        embedding_service=embedding_service,
        top_k=settings.RETRIEVAL_TOP_K,
        similarity_threshold=settings.SIMILARITY_THRESHOLD,
    )


@lru_cache
def get_llm_service() -> LLMService:
    """Get cached LLMService instance."""
    settings = get_settings()
    cache = get_cache()
    return LLMService(
        api_key=settings.OPENAI_API_KEY,
        model=settings.OPENAI_MODEL,
        temperature=settings.OPENAI_TEMPERATURE,
        max_tokens=settings.OPENAI_MAX_TOKENS,
        cache=cache,
        mock_mode=settings.MOCK_MODE,
    )


@lru_cache
def get_vision_processor() -> VisionProcessor:
    """Get cached VisionProcessor instance."""
    settings = get_settings()
    return VisionProcessor(
        api_key=settings.OPENAI_API_KEY,
        model=settings.OPENAI_MODEL,
        prompts_dir=settings.PROMPTS_DIR,
        mock_mode=settings.MOCK_MODE,
    )


# Typed dependencies for use in route functions
CacheDep = Annotated[ResponseCache | None, Depends(get_cache)]
EmbeddingDep = Annotated[EmbeddingService, Depends(get_embedding_service)]
RAGDep = Annotated[RAGEngine, Depends(get_rag_engine)]
LLMDep = Annotated[LLMService, Depends(get_llm_service)]
VisionDep = Annotated[VisionProcessor, Depends(get_vision_processor)]
