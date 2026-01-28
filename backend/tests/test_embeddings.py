"""
Unit tests for Embedding Service.

Tests embedding generation and similarity calculations.
"""

import pytest
from app.ingestion.embedder import EmbeddingService


class TestMockEmbeddingService:
    """Tests using mock embedding service."""
    
    @pytest.mark.asyncio
    async def test_embed_text(self, mock_embedding_service):
        """Test single text embedding."""
        embedding = await mock_embedding_service.embed_text("test text")
        
        assert isinstance(embedding, list)
        assert len(embedding) == mock_embedding_service.embedding_dimension
        assert all(isinstance(v, float) for v in embedding)
    
    @pytest.mark.asyncio
    async def test_embed_batch(self, mock_embedding_service):
        """Test batch embedding."""
        texts = ["text one", "text two", "text three"]
        embeddings = await mock_embedding_service.embed_batch(texts)
        
        assert len(embeddings) == len(texts)
        for emb in embeddings:
            assert len(emb) == mock_embedding_service.embedding_dimension
    
    @pytest.mark.asyncio
    async def test_deterministic_embeddings(self, mock_embedding_service):
        """Test that same text produces same embedding."""
        text = "consistent text for testing"
        
        emb1 = await mock_embedding_service.embed_text(text)
        emb2 = await mock_embedding_service.embed_text(text)
        
        assert emb1 == emb2
    
    @pytest.mark.asyncio
    async def test_different_texts_different_embeddings(self, mock_embedding_service):
        """Test that different texts produce different embeddings."""
        emb1 = await mock_embedding_service.embed_text("text one")
        emb2 = await mock_embedding_service.embed_text("text two")
        
        assert emb1 != emb2


class TestEmbeddingServiceUnit:
    """Unit tests for EmbeddingService methods."""
    
    def test_embedding_dimension_known_models(self):
        """Test embedding dimension lookup for known models."""
        # Test without API key (uses dimension lookup)
        service = EmbeddingService(api_key="", model="text-embedding-3-small")
        assert service.embedding_dimension == 1536
        
        service = EmbeddingService(api_key="", model="text-embedding-3-large")
        assert service.embedding_dimension == 3072
    
    @pytest.mark.asyncio
    async def test_similarity_calculation(self):
        """Test cosine similarity calculation."""
        service = EmbeddingService(api_key="", model="test")
        
        # Identical vectors
        vec1 = [1.0, 0.0, 0.0]
        sim = await service.similarity(vec1, vec1)
        assert abs(sim - 1.0) < 0.001
        
        # Orthogonal vectors
        vec2 = [0.0, 1.0, 0.0]
        sim = await service.similarity(vec1, vec2)
        assert abs(sim) < 0.001
        
        # Opposite vectors
        vec3 = [-1.0, 0.0, 0.0]
        sim = await service.similarity(vec1, vec3)
        assert abs(sim - (-1.0)) < 0.001
    
    @pytest.mark.asyncio
    async def test_similarity_zero_vector(self):
        """Test similarity with zero vector."""
        service = EmbeddingService(api_key="", model="test")
        
        vec1 = [1.0, 2.0, 3.0]
        vec_zero = [0.0, 0.0, 0.0]
        
        sim = await service.similarity(vec1, vec_zero)
        assert sim == 0.0
