"""
Knowledge base browsing endpoint.
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.api.dependencies import RAGDep, SettingsDep
from app.models.knowledge import KnowledgeBaseStats, DocumentList

router = APIRouter()


@router.get("/knowledge", response_model=KnowledgeBaseStats)
async def get_knowledge_base_stats(
    settings: SettingsDep,
    rag: RAGDep,
):
    """
    Get knowledge base statistics.
    
    Returns document counts by category and total embeddings.
    """
    try:
        stats = await rag.get_stats()
        return KnowledgeBaseStats(
            total_documents=stats["total_documents"],
            total_chunks=stats["total_chunks"],
            categories=stats["categories"],
            kb_version=stats.get("version", "unknown"),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get knowledge base stats: {str(e)}"
        )


@router.get("/knowledge/search")
async def search_knowledge_base(
    rag: RAGDep,
    query: str = Query(..., description="Search query"),
    category: Optional[str] = Query(None, description="Filter by category"),
    limit: int = Query(10, ge=1, le=50, description="Maximum results"),
):
    """
    Search the knowledge base.
    
    Returns documents matching the query, optionally filtered by category.
    """
    try:
        results = await rag.retrieve(
            query=query,
            top_k=limit,
            filter_category=category,
        )
        
        return {
            "query": query,
            "results": [
                {
                    "id": doc.id,
                    "title": doc.metadata.get("title", "Unknown"),
                    "category": doc.metadata.get("category", "unknown"),
                    "content": doc.content,
                    "relevance_score": doc.score,
                }
                for doc in results
            ],
            "total": len(results),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


@router.get("/knowledge/documents")
async def list_documents(
    rag: RAGDep,
    category: Optional[str] = Query(None, description="Filter by category"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results"),
):
    """
    List documents in the knowledge base.
    
    Returns paginated list of documents with metadata.
    """
    try:
        documents = await rag.list_documents(
            category=category,
            offset=offset,
            limit=limit,
        )
        
        return DocumentList(
            documents=[
                {
                    "id": doc.id,
                    "title": doc.metadata.get("title", "Unknown"),
                    "category": doc.metadata.get("category", "unknown"),
                    "chunk_count": doc.metadata.get("total_chunks", 1),
                }
                for doc in documents
            ],
            total=len(documents),
            offset=offset,
            limit=limit,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list documents: {str(e)}"
        )
