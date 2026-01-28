"""
Embedding Service - Generate embeddings using OpenAI.

Handles text embedding for document vectorization.
"""

import hashlib
import logging
from typing import Optional

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.timing import timed

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Embedding service using OpenAI's text-embedding models.
    
    Generates vector embeddings for text chunks for semantic search.
    Supports mock mode for testing and demos without API calls.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        mock_mode: bool = False,
    ):
        """
        Initialize the embedding service.
        
        Args:
            api_key: OpenAI API key
            model: Embedding model to use
            mock_mode: If True, generate deterministic mock embeddings
        """
        self.mock_mode = mock_mode
        self.client = AsyncOpenAI(api_key=api_key) if api_key and not mock_mode else None
        self.model = model
        self._embedding_dim: Optional[int] = None
    
    @property
    def embedding_dimension(self) -> int:
        """Get the embedding dimension for the model."""
        if self._embedding_dim:
            return self._embedding_dim
        
        # Known dimensions for OpenAI models
        dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return dimensions.get(self.model, 1536)
    
    def _mock_embedding(self, text: str) -> list[float]:
        """Generate a deterministic mock embedding based on text hash."""
        if not text.strip():
            return [0.0] * self.embedding_dimension
        
        # Use hash to generate deterministic but varied embedding
        hash_val = int(hashlib.sha256(text.encode()).hexdigest(), 16)
        embedding = []
        for i in range(self.embedding_dimension):
            # Generate values between -1 and 1
            val = ((hash_val >> (i % 64)) % 1000) / 500.0 - 1.0
            embedding.append(val)
        
        # Normalize the vector
        import math
        norm = math.sqrt(sum(v * v for v in embedding))
        if norm > 0:
            embedding = [v / norm for v in embedding]
        
        return embedding
    
    @timed("embed_text")
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        if self.mock_mode:
            return self._mock_embedding(text)
        
        if not self.client:
            raise ValueError("OpenAI API key not configured")
        
        if not text.strip():
            # Return zero vector for empty text
            return [0.0] * self.embedding_dimension
        
        response = await self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        
        embedding = response.data[0].embedding
        self._embedding_dim = len(embedding)
        
        return embedding
    
    @timed("embed_batch")
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.
        
        More efficient than calling embed_text for each text.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        if self.mock_mode:
            return [self._mock_embedding(text) for text in texts]
        
        if not self.client:
            raise ValueError("OpenAI API key not configured")
        
        # Filter empty texts but track indices
        non_empty_indices = []
        non_empty_texts = []
        for i, text in enumerate(texts):
            if text.strip():
                non_empty_indices.append(i)
                non_empty_texts.append(text)
        
        # If all empty, return zero vectors
        if not non_empty_texts:
            return [[0.0] * self.embedding_dimension for _ in texts]
        
        # Batch API call
        response = await self.client.embeddings.create(
            model=self.model,
            input=non_empty_texts,
        )
        
        # Build result with zero vectors for empty texts
        embeddings = [[0.0] * self.embedding_dimension for _ in texts]
        for i, data in enumerate(response.data):
            original_idx = non_empty_indices[i]
            embeddings[original_idx] = data.embedding
        
        self._embedding_dim = len(response.data[0].embedding)
        
        logger.info(f"Generated {len(non_empty_texts)} embeddings")
        return embeddings
    
    async def similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First embedding vector
            vec2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        import math
        
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have same dimension")
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
