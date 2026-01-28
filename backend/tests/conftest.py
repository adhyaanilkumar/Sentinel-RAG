"""
Pytest fixtures for Sentinel RAG tests.

Provides shared test fixtures and configuration.
"""

import os
import tempfile
from typing import Generator

import pytest
from fastapi.testclient import TestClient

# Set test environment before importing app
os.environ["MOCK_MODE"] = "true"
os.environ["OPENAI_API_KEY"] = "test-key"
os.environ["DEBUG"] = "true"


@pytest.fixture(scope="session", autouse=True)
def clear_caches():
    """Clear LRU caches before tests to ensure fresh instances."""
    from app.config import get_settings
    from app.api.dependencies import (
        get_cache,
        get_embedding_service,
        get_llm_service,
        get_rag_engine,
        get_vision_processor,
    )
    
    # Clear all caches
    get_settings.cache_clear()
    get_cache.cache_clear()
    get_embedding_service.cache_clear()
    get_llm_service.cache_clear()
    get_rag_engine.cache_clear()
    get_vision_processor.cache_clear()
    
    yield
    
    # Clear again after tests
    get_settings.cache_clear()
    get_cache.cache_clear()
    get_embedding_service.cache_clear()
    get_llm_service.cache_clear()
    get_rag_engine.cache_clear()
    get_vision_processor.cache_clear()


@pytest.fixture(scope="session")
def test_data_dir() -> str:
    """Get path to test data directory."""
    return os.path.join(os.path.dirname(__file__), "..", "data", "test_cases")


@pytest.fixture(scope="function")
def temp_dir() -> Generator[str, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        yield tmpdir


@pytest.fixture(scope="session")
def client() -> Generator[TestClient, None, None]:
    """Create a test client for the FastAPI app."""
    from app.main import app
    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="function")
def temp_chroma_dir(temp_dir: str) -> str:
    """Create a temporary ChromaDB directory."""
    chroma_dir = os.path.join(temp_dir, "chroma")
    os.makedirs(chroma_dir)
    return chroma_dir


@pytest.fixture(scope="function")
def mock_embedding_service():
    """Create a mock embedding service for testing."""
    class MockEmbeddingService:
        def __init__(self):
            self.model = "test-model"
            self._embedding_dim = 1536
        
        @property
        def embedding_dimension(self):
            return self._embedding_dim
        
        async def embed_text(self, text: str) -> list[float]:
            # Return a deterministic embedding based on text hash
            import hashlib
            hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
            return [(hash_val >> i) % 100 / 100.0 for i in range(self._embedding_dim)]
        
        async def embed_batch(self, texts: list[str]) -> list[list[float]]:
            return [await self.embed_text(t) for t in texts]
    
    return MockEmbeddingService()


@pytest.fixture(scope="function")
def rag_engine(temp_chroma_dir: str, mock_embedding_service):
    """Create a RAG engine for testing."""
    from app.core.rag_engine import RAGEngine
    return RAGEngine(
        persist_dir=temp_chroma_dir,
        embedding_service=mock_embedding_service,
        top_k=5,
        similarity_threshold=0.5,  # Lower threshold for testing
    )


@pytest.fixture(scope="function")
def sample_documents():
    """Sample documents for testing."""
    from app.models.documents import DocumentChunk
    return [
        DocumentChunk(
            content="BATTLE REPORT: Operation Northern Shield. A naval engagement involving destroyer formations in the northern sector.",
            metadata={"title": "Operation Northern Shield", "category": "battle_report"}
        ),
        DocumentChunk(
            content="EQUIPMENT PROFILE: Valkyrie-class Destroyer. A modern naval destroyer with advanced radar systems.",
            metadata={"title": "Valkyrie-class Destroyer", "category": "equipment"}
        ),
        DocumentChunk(
            content="TACTICAL PATTERN: Pincer Formation. A flanking maneuver used by naval groups to encircle targets.",
            metadata={"title": "Pincer Formation", "category": "tactical_pattern"}
        ),
    ]


@pytest.fixture(scope="function")
def sample_image_bytes() -> bytes:
    """Generate sample image bytes for testing."""
    # Create a minimal valid PNG image (1x1 pixel, red)
    # PNG header + IHDR chunk + IDAT chunk + IEND chunk
    import struct
    import zlib
    
    def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        chunk = chunk_type + data
        crc = zlib.crc32(chunk) & 0xffffffff
        return struct.pack(">I", len(data)) + chunk + struct.pack(">I", crc)
    
    # PNG signature
    signature = b'\x89PNG\r\n\x1a\n'
    
    # IHDR chunk (1x1 image, 8-bit RGB)
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    ihdr = png_chunk(b'IHDR', ihdr_data)
    
    # IDAT chunk (compressed pixel data - red pixel)
    raw_data = b'\x00\xff\x00\x00'  # filter byte + RGB
    compressed = zlib.compress(raw_data)
    idat = png_chunk(b'IDAT', compressed)
    
    # IEND chunk
    iend = png_chunk(b'IEND', b'')
    
    return signature + ihdr + idat + iend
