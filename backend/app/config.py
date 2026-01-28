"""
Configuration settings for Sentinel RAG.

IMPORTANT: API parameters are LOCKED for reproducibility.
Do not modify these values without updating the prompt changelog.
"""

import os
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )
    
    # Application
    APP_NAME: str = "Sentinel RAG"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False
    
    # API Configuration
    API_PREFIX: str = "/api"
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    # OpenAI Configuration - LOCKED FOR REPRODUCIBILITY
    OPENAI_API_KEY: str = ""  # Set via environment variable
    OPENAI_MODEL: str = "gpt-4-vision-preview"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_TEMPERATURE: float = 0.0  # Deterministic for reproducibility
    OPENAI_MAX_TOKENS: int = 1024
    OPENAI_TOP_P: float = 1.0
    
    # Retrieval Configuration - LOCKED FOR REPRODUCIBILITY
    RETRIEVAL_TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.7
    
    # Chunking Configuration - LOCKED FOR REPRODUCIBILITY
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    
    # Performance Configuration
    REQUEST_TIMEOUT_SECONDS: int = 30
    IMAGE_ANALYSIS_TIMEOUT_SECONDS: int = 10
    
    # Cache Configuration
    CACHE_RESPONSES: bool = True
    CACHE_DB_PATH: str = "data/response_cache.db"
    
    # Mock Mode (for demos and testing without API)
    MOCK_MODE: bool = False
    
    # File Upload Configuration
    MAX_UPLOAD_SIZE_MB: int = 10
    ALLOWED_IMAGE_TYPES: list[str] = ["image/png", "image/jpeg", "image/jpg"]
    UPLOAD_DIR: str = "data/uploads"
    
    # Knowledge Base Configuration
    KNOWLEDGE_BASE_PATH: str = "data/knowledge_base"
    CHROMA_PERSIST_DIR: str = "data/chroma"
    
    # Prompts Configuration
    PROMPTS_DIR: str = "prompts"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Convenience function to get settings
settings = get_settings()
