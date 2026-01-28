"""Chat request and response models."""

from typing import Any, Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A single chat message."""
    
    role: str = Field(..., description="Message role (user/assistant/system)")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    
    messages: list[ChatMessage] = Field(
        ..., 
        min_length=1,
        description="Conversation history"
    )
    previous_analysis: Optional[str] = Field(
        None,
        description="Previous tactical assessment for context"
    )


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    
    success: bool = Field(..., description="Whether chat completed successfully")
    message: ChatMessage = Field(..., description="Assistant response message")
    retrieved_documents: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Documents retrieved for this response"
    )
    error: Optional[str] = Field(None, description="Error message if success is False")
