"""
RAG Engine - Retrieval-Augmented Generation core logic.

Handles document storage, embedding, and semantic retrieval.
"""

import os
import uuid
import logging
from typing import Optional
from collections import defaultdict

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.models.documents import Document, RetrievedDocument, DocumentChunk
from app.ingestion.embedder import EmbeddingService
from app.core.timing import timed

logger = logging.getLogger(__name__)


class RAGEngine:
    """
    RAG Engine for document retrieval and knowledge base management.
    
    Uses ChromaDB for vector storage and OpenAI embeddings for semantic search.
    """
    
    COLLECTION_NAME = "sentinel_knowledge_base"
    
    def __init__(
        self,
        persist_dir: str,
        embedding_service: EmbeddingService,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
    ):
        """
        Initialize the RAG engine.
        
        Args:
            persist_dir: Directory for ChromaDB persistence
            embedding_service: Service for generating embeddings
            top_k: Default number of documents to retrieve
            similarity_threshold: Minimum similarity score for retrieval
        """
        self.persist_dir = persist_dir
        self.embedding_service = embedding_service
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        
        # Ensure persist directory exists
        os.makedirs(persist_dir, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
            )
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"RAG Engine initialized with {self.collection.count()} documents")
    
    @timed("rag_retrieve")
    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_category: Optional[str] = None,
    ) -> list[RetrievedDocument]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve (overrides default)
            filter_category: Optional category filter
            
        Returns:
            List of retrieved documents with relevance scores
        """
        if not query.strip():
            return []
        
        k = top_k or self.top_k
        
        # Build filter if category specified
        where_filter = None
        if filter_category:
            where_filter = {"category": filter_category}
        
        # Generate query embedding
        query_embedding = await self.embedding_service.embed_text(query)
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        
        # Convert to RetrievedDocument objects
        documents = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                # ChromaDB returns distances, convert to similarity score
                # For cosine distance: similarity = 1 - distance
                distance = results["distances"][0][i] if results["distances"] else 0
                score = 1 - distance
                
                # Filter by similarity threshold
                if score < self.similarity_threshold:
                    continue
                
                documents.append(RetrievedDocument(
                    id=doc_id,
                    content=results["documents"][0][i],
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                    score=score,
                ))
        
        logger.info(f"Retrieved {len(documents)} documents for query: {query[:50]}...")
        return documents
    
    @timed("rag_add_documents")
    async def add_documents(
        self,
        chunks: list[DocumentChunk],
    ) -> list[str]:
        """
        Add document chunks to the knowledge base.
        
        Args:
            chunks: List of document chunks to add
            
        Returns:
            List of generated document IDs
        """
        if not chunks:
            return []
        
        # Generate IDs for chunks
        ids = [str(uuid.uuid4()) for _ in chunks]
        
        # Extract content and metadata
        contents = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        # Generate embeddings
        embeddings = await self.embedding_service.embed_batch(contents)
        
        # Add to collection
        self.collection.add(
            ids=ids,
            documents=contents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        
        logger.info(f"Added {len(chunks)} chunks to knowledge base")
        return ids
    
    async def list_documents(
        self,
        category: Optional[str] = None,
        offset: int = 0,
        limit: int = 20,
    ) -> list[Document]:
        """
        List documents in the knowledge base.
        
        Args:
            category: Optional category filter
            offset: Pagination offset
            limit: Maximum documents to return
            
        Returns:
            List of documents
        """
        where_filter = None
        if category:
            where_filter = {"category": category}
        
        # Get all documents (ChromaDB doesn't support true pagination)
        results = self.collection.get(
            where=where_filter,
            include=["documents", "metadatas"],
            limit=limit,
            offset=offset,
        )
        
        documents = []
        if results["ids"]:
            for i, doc_id in enumerate(results["ids"]):
                documents.append(Document(
                    id=doc_id,
                    content=results["documents"][i] if results["documents"] else "",
                    metadata=results["metadatas"][i] if results["metadatas"] else {},
                ))
        
        return documents
    
    async def get_stats(self) -> dict:
        """
        Get knowledge base statistics.
        
        Returns:
            Dictionary with statistics
        """
        total_chunks = self.collection.count()
        
        # Get all metadatas to count categories
        results = self.collection.get(include=["metadatas"])
        
        categories = defaultdict(int)
        document_titles = set()
        
        if results["metadatas"]:
            for metadata in results["metadatas"]:
                category = metadata.get("category", "unknown")
                categories[category] += 1
                
                title = metadata.get("title")
                if title:
                    document_titles.add(title)
        
        return {
            "total_documents": len(document_titles),
            "total_chunks": total_chunks,
            "categories": dict(categories),
        }
    
    async def delete_all(self):
        """Delete all documents from the knowledge base."""
        self.client.delete_collection(self.COLLECTION_NAME)
        self.collection = self.client.create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        logger.warning("All documents deleted from knowledge base")
    
    def reset(self):
        """Reset the entire database."""
        self.client.reset()
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        logger.warning("RAG Engine reset")
