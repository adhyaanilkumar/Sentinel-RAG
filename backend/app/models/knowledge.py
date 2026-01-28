"""Knowledge base models."""

from typing import Any, Optional
from pydantic import BaseModel, Field


class CategoryStats(BaseModel):
    """Statistics for a document category."""
    
    name: str = Field(..., description="Category name")
    document_count: int = Field(..., description="Number of documents")
    chunk_count: int = Field(..., description="Number of chunks")


class KnowledgeBaseStats(BaseModel):
    """Statistics for the entire knowledge base."""
    
    total_documents: int = Field(..., description="Total number of documents")
    total_chunks: int = Field(..., description="Total number of chunks")
    categories: dict[str, int] = Field(
        ..., 
        description="Document count by category"
    )
    kb_version: str = Field(..., description="Knowledge base version hash")


class DocumentSummary(BaseModel):
    """Summary of a document in the knowledge base."""
    
    id: str = Field(..., description="Document identifier")
    title: str = Field(..., description="Document title")
    category: str = Field(..., description="Document category")
    chunk_count: int = Field(default=1, description="Number of chunks")


class DocumentList(BaseModel):
    """Paginated list of documents."""
    
    documents: list[dict[str, Any]] = Field(..., description="List of documents")
    total: int = Field(..., description="Total documents returned")
    offset: int = Field(..., description="Pagination offset")
    limit: int = Field(..., description="Pagination limit")
