"""Ingestion request and response models."""

from typing import Optional
from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    """Request model for document ingestion."""
    
    title: str = Field(..., description="Document title")
    category: str = Field(
        ..., 
        description="Document category (battle_report, equipment, tactical_pattern, location, signature)"
    )
    content: Optional[str] = Field(None, description="Raw text content")
    metadata: Optional[dict] = Field(None, description="Additional metadata")


class IngestResponse(BaseModel):
    """Response model for document ingestion."""
    
    success: bool = Field(..., description="Whether ingestion completed successfully")
    document_title: str = Field(..., description="Title of ingested document")
    chunks_created: int = Field(..., description="Number of chunks created")
    chunk_ids: list[str] = Field(..., description="IDs of created chunks")
    error: Optional[str] = Field(None, description="Error message if success is False")
