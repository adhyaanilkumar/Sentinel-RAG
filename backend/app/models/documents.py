"""Document models for internal use."""

from typing import Any, Optional
from pydantic import BaseModel, Field


class Document(BaseModel):
    """A document chunk for storage and retrieval."""
    
    id: str = Field(..., description="Unique document identifier")
    content: str = Field(..., description="Document text content")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Document metadata (title, category, etc.)"
    )


class RetrievedDocument(BaseModel):
    """A document retrieved from the knowledge base with relevance score."""
    
    id: str = Field(..., description="Document identifier")
    content: str = Field(..., description="Document text content")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Document metadata"
    )
    score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Relevance score (0-1)"
    )


class DocumentChunk(BaseModel):
    """A chunk of a document for embedding."""
    
    content: str = Field(..., description="Chunk text content")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Chunk metadata including parent document info"
    )
