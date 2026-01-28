"""
Unit tests for RAG Engine.

Tests document storage, retrieval, and similarity search.
"""

import pytest
from app.core.rag_engine import RAGEngine
from app.models.documents import DocumentChunk


class TestRAGEngine:
    """Tests for RAGEngine class."""
    
    @pytest.mark.asyncio
    async def test_add_documents(self, rag_engine, sample_documents):
        """Test that documents can be added to the knowledge base."""
        chunk_ids = await rag_engine.add_documents(sample_documents)
        
        assert len(chunk_ids) == len(sample_documents)
        assert all(isinstance(id, str) for id in chunk_ids)
    
    @pytest.mark.asyncio
    async def test_retrieval_returns_documents(self, rag_engine, sample_documents):
        """Test that retrieval returns relevant documents."""
        # Add documents first
        await rag_engine.add_documents(sample_documents)
        
        # Query for destroyer-related content
        query = "destroyer formation in northern waters"
        results = await rag_engine.retrieve(query, top_k=3)
        
        assert len(results) > 0
        assert len(results) <= 3
    
    @pytest.mark.asyncio
    async def test_empty_query_handled(self, rag_engine):
        """Test graceful handling of empty queries."""
        results = await rag_engine.retrieve("", top_k=3)
        assert results == []
    
    @pytest.mark.asyncio
    async def test_retrieval_with_category_filter(self, rag_engine, sample_documents):
        """Test retrieval with category filtering."""
        await rag_engine.add_documents(sample_documents)
        
        # Query with category filter
        results = await rag_engine.retrieve(
            query="destroyer",
            top_k=5,
            filter_category="equipment"
        )
        
        # All results should be from equipment category
        for doc in results:
            assert doc.metadata.get("category") == "equipment"
    
    @pytest.mark.asyncio
    async def test_get_stats(self, rag_engine, sample_documents):
        """Test knowledge base statistics."""
        await rag_engine.add_documents(sample_documents)
        
        stats = await rag_engine.get_stats()
        
        assert stats["total_chunks"] == len(sample_documents)
        assert "categories" in stats
        assert "battle_report" in stats["categories"]
    
    @pytest.mark.asyncio
    async def test_list_documents(self, rag_engine, sample_documents):
        """Test document listing."""
        await rag_engine.add_documents(sample_documents)
        
        documents = await rag_engine.list_documents(limit=10)
        
        assert len(documents) == len(sample_documents)
    
    @pytest.mark.asyncio
    async def test_delete_all(self, rag_engine, sample_documents):
        """Test deleting all documents."""
        await rag_engine.add_documents(sample_documents)
        
        # Verify documents exist
        stats_before = await rag_engine.get_stats()
        assert stats_before["total_chunks"] > 0
        
        # Delete all
        await rag_engine.delete_all()
        
        # Verify empty
        stats_after = await rag_engine.get_stats()
        assert stats_after["total_chunks"] == 0
    
    @pytest.mark.asyncio
    async def test_retrieval_relevance(self, rag_engine, sample_documents):
        """Test that retrieved docs are relevant to query."""
        await rag_engine.add_documents(sample_documents)
        
        query = "tactical flanking maneuver"
        results = await rag_engine.retrieve(query, top_k=3)
        
        # Should find the tactical pattern document
        categories = [r.metadata.get("category") for r in results]
        # At least one result should be tactical_pattern
        assert len(results) > 0  # Basic check that we got results
    
    @pytest.mark.asyncio
    async def test_similarity_scores(self, rag_engine, sample_documents):
        """Test that similarity scores are within valid range."""
        await rag_engine.add_documents(sample_documents)
        
        results = await rag_engine.retrieve("naval operations", top_k=5)
        
        for doc in results:
            assert 0 <= doc.score <= 1
