"""
API integration tests.

Tests FastAPI endpoints end-to-end.
"""

import pytest
from fastapi.testclient import TestClient


class TestHealthEndpoints:
    """Tests for health check endpoints."""
    
    def test_health_check(self, client: TestClient):
        """Test health endpoint returns healthy status."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
    
    def test_root_endpoint(self, client: TestClient):
        """Test root endpoint returns API info."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "docs" in data


class TestKnowledgeEndpoints:
    """Tests for knowledge base endpoints."""
    
    def test_get_knowledge_stats(self, client: TestClient):
        """Test getting knowledge base statistics."""
        response = client.get("/api/knowledge")
        
        assert response.status_code == 200
        data = response.json()
        assert "total_documents" in data
        assert "total_chunks" in data
        assert "categories" in data
    
    def test_search_knowledge_base(self, client: TestClient):
        """Test searching the knowledge base."""
        response = client.get("/api/knowledge/search", params={
            "query": "destroyer formation",
            "limit": 5
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "query" in data
        assert "results" in data
        assert isinstance(data["results"], list)
    
    def test_list_documents(self, client: TestClient):
        """Test listing documents."""
        response = client.get("/api/knowledge/documents", params={
            "limit": 10,
            "offset": 0
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "documents" in data
        assert "total" in data
        assert "offset" in data
        assert "limit" in data


class TestAnalyzeEndpoint:
    """Tests for analysis endpoint."""
    
    def test_analyze_requires_file(self, client: TestClient):
        """Test that analyze endpoint requires a file."""
        response = client.post("/api/analyze")
        
        # Should return 422 (validation error) without file
        assert response.status_code == 422
    
    def test_analyze_with_invalid_file_type(self, client: TestClient):
        """Test rejection of invalid file types."""
        # Create a text file (not an image)
        response = client.post(
            "/api/analyze",
            files={"file": ("test.txt", b"not an image", "text/plain")}
        )
        
        assert response.status_code == 400
        assert "Invalid file type" in response.json()["detail"]
    
    def test_analyze_with_valid_image(self, client: TestClient, sample_image_bytes):
        """Test analysis with valid image."""
        response = client.post(
            "/api/analyze",
            files={"file": ("test.png", sample_image_bytes, "image/png")},
            data={"context": "Test context"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "image_analysis" in data
        assert "tactical_assessment" in data
        assert "timing" in data


class TestChatEndpoint:
    """Tests for chat endpoint."""
    
    def test_chat_requires_messages(self, client: TestClient):
        """Test that chat endpoint requires messages."""
        response = client.post("/api/chat", json={})
        
        assert response.status_code == 422
    
    def test_chat_with_message(self, client: TestClient):
        """Test chat with a valid message."""
        response = client.post("/api/chat", json={
            "messages": [
                {"role": "user", "content": "What do you see in the analysis?"}
            ]
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "message" in data
        assert data["message"]["role"] == "assistant"


class TestIngestEndpoint:
    """Tests for ingestion endpoint."""
    
    def test_ingest_requires_title_and_category(self, client: TestClient):
        """Test that ingest requires title and category."""
        response = client.post("/api/ingest", data={})
        
        assert response.status_code == 422
    
    def test_ingest_with_content(self, client: TestClient):
        """Test ingesting content directly."""
        response = client.post("/api/ingest", data={
            "title": "Test Document",
            "category": "battle_report",
            "content": "This is a test battle report about a naval engagement."
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert data["document_title"] == "Test Document"
        assert data["chunks_created"] > 0
