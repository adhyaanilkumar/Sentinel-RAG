"""
Analysis endpoint - Main analysis pipeline for sensor data.

Handles image upload, vision analysis, RAG retrieval, and tactical assessment.
"""

import os
import tempfile
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.api.dependencies import (
    LLMDep,
    RAGDep,
    SettingsDep,
    VisionDep,
)
from app.models.analysis import AnalysisRequest, AnalysisResponse, TimingInfo
from app.core.timing import timed

router = APIRouter()


@asynccontextmanager
async def temporary_upload(file: UploadFile, settings):
    """
    Store upload temporarily and auto-delete after processing.
    
    Ensures uploaded files are cleaned up regardless of processing outcome.
    """
    temp_path = None
    try:
        # Validate file type
        if file.content_type not in settings.ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed types: {settings.ALLOWED_IMAGE_TYPES}"
            )
        
        # Validate file size
        content = await file.read()
        if len(content) > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {settings.MAX_UPLOAD_SIZE_MB}MB"
            )
        
        # Save to temp file
        suffix = os.path.splitext(file.filename)[1] if file.filename else ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            temp_path = tmp.name
        
        yield temp_path, content
        
    finally:
        # Always delete after processing
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_sensor_data(
    settings: SettingsDep,
    vision: VisionDep,
    rag: RAGDep,
    llm: LLMDep,
    file: UploadFile = File(..., description="Sensor image (radar, sonar, satellite)"),
    context: Optional[str] = Form(None, description="Additional context about the situation"),
):
    """
    Analyze sensor data with RAG-enhanced intelligence assessment.
    
    Pipeline:
    1. Analyze image with GPT-4V
    2. Generate retrieval query from image analysis
    3. Retrieve relevant historical documents
    4. Synthesize tactical assessment with context
    
    Returns structured intelligence report with timing information.
    """
    timing = {}
    start_total = time.perf_counter()
    
    try:
        async with temporary_upload(file, settings) as (temp_path, image_content):
            # Step 1: Vision analysis
            start = time.perf_counter()
            image_analysis = await vision.analyze_image(image_content)
            timing["image_analysis_ms"] = int((time.perf_counter() - start) * 1000)
            
            # Step 2: Generate retrieval query and retrieve documents
            start = time.perf_counter()
            query = f"{image_analysis}\n\nContext: {context}" if context else image_analysis
            retrieved_docs = await rag.retrieve(query)
            timing["retrieval_ms"] = int((time.perf_counter() - start) * 1000)
            
            # Step 3: Synthesize tactical assessment
            start = time.perf_counter()
            assessment = await llm.generate_assessment(
                image_analysis=image_analysis,
                retrieved_context=retrieved_docs,
                additional_context=context,
            )
            timing["synthesis_ms"] = int((time.perf_counter() - start) * 1000)
            
            timing["total_ms"] = int((time.perf_counter() - start_total) * 1000)
            
            return AnalysisResponse(
                success=True,
                image_analysis=image_analysis,
                retrieved_documents=[
                    {
                        "id": doc.id,
                        "title": doc.metadata.get("title", "Unknown"),
                        "content": doc.content[:500] + "..." if len(doc.content) > 500 else doc.content,
                        "relevance_score": doc.score,
                        "category": doc.metadata.get("category", "unknown"),
                    }
                    for doc in retrieved_docs
                ],
                tactical_assessment=assessment,
                timing=TimingInfo(**timing),
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )
