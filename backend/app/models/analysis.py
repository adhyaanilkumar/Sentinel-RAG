"""Analysis request and response models."""

from typing import Any, Optional
from pydantic import BaseModel, Field


class TimingInfo(BaseModel):
    """Timing information for analysis pipeline stages."""
    
    image_analysis_ms: int = Field(..., description="Image analysis time in milliseconds")
    retrieval_ms: int = Field(..., description="Document retrieval time in milliseconds")
    synthesis_ms: int = Field(..., description="Assessment synthesis time in milliseconds")
    total_ms: int = Field(..., description="Total pipeline time in milliseconds")


class AnalysisRequest(BaseModel):
    """Request model for analysis endpoint."""
    
    context: Optional[str] = Field(
        None,
        description="Additional context about the situation"
    )


class RetrievedDocumentSummary(BaseModel):
    """Summary of a retrieved document for response."""
    
    id: str = Field(..., description="Document identifier")
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Document content (may be truncated)")
    relevance_score: float = Field(..., description="Relevance score")
    category: str = Field(..., description="Document category")


class TacticalAssessment(BaseModel):
    """Structured tactical assessment output."""
    
    sensor_type: str = Field(..., description="Type of sensor data (radar/sonar/satellite)")
    observations: list[str] = Field(..., description="Key observations from the image")
    threat_level: str = Field(..., description="Assessed threat level (low/moderate/elevated/high/critical)")
    intent_analysis: str = Field(..., description="Analysis of likely enemy intent")
    confidence: str = Field(..., description="Assessment confidence (low/medium/high)")
    recommended_actions: list[str] = Field(..., description="Recommended response actions")
    caveats: list[str] = Field(default_factory=list, description="Important caveats or limitations")


class AnalysisResponse(BaseModel):
    """Response model for analysis endpoint."""
    
    success: bool = Field(..., description="Whether analysis completed successfully")
    image_analysis: str = Field(..., description="Raw image analysis from vision model")
    retrieved_documents: list[dict[str, Any]] = Field(
        ..., 
        description="Documents retrieved from knowledge base"
    )
    tactical_assessment: TacticalAssessment | str = Field(
        ..., 
        description="Structured tactical assessment"
    )
    timing: TimingInfo = Field(..., description="Pipeline timing information")
    error: Optional[str] = Field(None, description="Error message if success is False")


class AnalysisResultSimple(BaseModel):
    """Simplified analysis result for internal use."""
    
    success: bool
    analysis: Optional[str] = None
    confidence: str = "medium"
    note: Optional[str] = None
    error: Optional[str] = None
