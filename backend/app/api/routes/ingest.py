"""
Ingestion endpoint - Add documents to the knowledge base.
"""

import os
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.api.dependencies import EmbeddingDep, RAGDep, SettingsDep
from app.models.ingestion import IngestRequest, IngestResponse
from app.ingestion.loader import DocumentLoader
from app.ingestion.chunker import SemanticChunker

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    settings: SettingsDep,
    rag: RAGDep,
    embedding: EmbeddingDep,
    file: Optional[UploadFile] = File(None, description="Document file (PDF, MD, TXT)"),
    content: Optional[str] = Form(None, description="Raw text content"),
    title: str = Form(..., description="Document title"),
    category: str = Form(..., description="Category (battle_report, equipment, tactical_pattern, location, signature)"),
    metadata: Optional[str] = Form(None, description="Additional metadata as JSON string"),
):
    """
    Ingest a document into the knowledge base.
    
    Documents are chunked semantically and embedded for retrieval.
    """
    try:
        loader = DocumentLoader()
        chunker = SemanticChunker(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        
        # Load content from file or direct input
        if file:
            content_bytes = await file.read()
            raw_content = loader.load_from_bytes(
                content_bytes, 
                filename=file.filename
            )
        elif content:
            raw_content = content
        else:
            raise HTTPException(
                status_code=400,
                detail="Either file or content must be provided"
            )
        
        # Parse metadata
        doc_metadata = {
            "title": title,
            "category": category,
        }
        if metadata:
            import json
            try:
                extra_metadata = json.loads(metadata)
                doc_metadata.update(extra_metadata)
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid metadata JSON"
                )
        
        # Chunk the document
        chunks = chunker.chunk(raw_content, doc_metadata)
        
        # Embed and store chunks
        chunk_ids = await rag.add_documents(chunks)
        
        return IngestResponse(
            success=True,
            document_title=title,
            chunks_created=len(chunk_ids),
            chunk_ids=chunk_ids,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {str(e)}"
        )


@router.post("/ingest/batch")
async def ingest_knowledge_base(
    settings: SettingsDep,
    rag: RAGDep,
):
    """
    Ingest all documents from the knowledge base directory.
    
    Reads all markdown files from the configured knowledge base path
    and ingests them into the vector store.
    """
    try:
        loader = DocumentLoader()
        chunker = SemanticChunker(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        
        kb_path = settings.KNOWLEDGE_BASE_PATH
        if not os.path.exists(kb_path):
            raise HTTPException(
                status_code=404,
                detail=f"Knowledge base path not found: {kb_path}"
            )
        
        total_docs = 0
        total_chunks = 0
        
        # Walk through knowledge base directory
        for root, dirs, files in os.walk(kb_path):
            for filename in files:
                if filename.endswith(('.md', '.txt')):
                    filepath = os.path.join(root, filename)
                    
                    # Determine category from directory
                    rel_path = os.path.relpath(root, kb_path)
                    category = rel_path.split(os.sep)[0] if rel_path != "." else "general"
                    
                    # Load and chunk document
                    raw_content, frontmatter = loader.load_file(filepath)
                    
                    metadata = {
                        "title": frontmatter.get("title", filename),
                        "category": frontmatter.get("category", category),
                        "source_file": filepath,
                        **frontmatter,
                    }
                    
                    chunks = chunker.chunk(raw_content, metadata)
                    await rag.add_documents(chunks)
                    
                    total_docs += 1
                    total_chunks += len(chunks)
        
        return {
            "success": True,
            "documents_processed": total_docs,
            "chunks_created": total_chunks,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch ingestion failed: {str(e)}"
        )
