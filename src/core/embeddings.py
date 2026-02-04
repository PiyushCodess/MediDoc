"""
Embeddings module for generating and managing text embeddings.
"""
from typing import List, Optional
import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings

from ..utils.config import get_settings


class EmbeddingManager:
    """Manages text embeddings using OpenAI or HuggingFace models."""
    
    def __init__(self, use_openai: bool = True):
        """
        Initialize the embedding manager.
        
        Args:
            use_openai: If True, use OpenAI embeddings; otherwise use HuggingFace
        """
        self.settings = get_settings()
        self.use_openai = use_openai
        self.embeddings = self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize the appropriate embedding model."""
        if self.use_openai:
            return OpenAIEmbeddings(
                model=self.settings.embedding_model,
                openai_api_key=self.settings.openai_api_key
            )
        else:
            # Use HuggingFace embeddings as alternative
            model_name = self.settings.embedding_model
            if model_name.startswith("text-embedding"):
                # Default to a good HuggingFace model if OpenAI model name provided
                model_name = "sentence-transformers/all-MiniLM-L6-v2"
            
            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'},  # Change to 'cuda' for GPU
                encode_kwargs={'normalize_embeddings': True}
            )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        return self.embeddings.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        return self.embeddings.embed_query(text)
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        
        Returns:
            Embedding dimension
        """
        # Embed a test string to get dimension
        test_embedding = self.embed_query("test")
        return len(test_embedding)
    
    def batch_embed_documents(
        self,
        texts: List[str],
        batch_size: int = 100,
        show_progress: bool = False
    ) -> List[List[float]]:
        """
        Embed documents in batches for efficiency.
        
        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to embed per batch
            show_progress: Whether to show progress
            
        Returns:
            List of embedding vectors
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)
            
            if show_progress:
                progress = min(i + batch_size, len(texts))
                print(f"Embedded {progress}/{len(texts)} documents")
        
        return all_embeddings
    
    def compute_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Compute cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def find_most_similar(
        self,
        query_embedding: List[float],
        candidate_embeddings: List[List[float]],
        top_k: int = 5
    ) -> List[tuple[int, float]]:
        """
        Find the most similar embeddings to a query embedding.
        
        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: List of candidate embedding vectors
            top_k: Number of top results to return
            
        Returns:
            List of (index, similarity_score) tuples
        """
        similarities = []
        
        for i, candidate in enumerate(candidate_embeddings):
            similarity = self.compute_similarity(query_embedding, candidate)
            similarities.append((i, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]


class EmbeddingCache:
    """Simple cache for embeddings to avoid recomputation."""
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize the cache.
        
        Args:
            max_size: Maximum number of embeddings to cache
        """
        self.cache = {}
        self.max_size = max_size
        self.access_count = {}
    
    def get(self, text: str) -> Optional[List[float]]:
        """
        Get embedding from cache.
        
        Args:
            text: Text to look up
            
        Returns:
            Cached embedding or None if not found
        """
        if text in self.cache:
            self.access_count[text] = self.access_count.get(text, 0) + 1
            return self.cache[text]
        return None
    
    def set(self, text: str, embedding: List[float]):
        """
        Store embedding in cache.
        
        Args:
            text: Text key
            embedding: Embedding vector to cache
        """
        # Evict least accessed if cache is full
        if len(self.cache) >= self.max_size:
            least_accessed = min(self.access_count, key=self.access_count.get)
            del self.cache[least_accessed]
            del self.access_count[least_accessed]
        
        self.cache[text] = embedding
        self.access_count[text] = 0
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.access_count.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)


class CachedEmbeddingManager(EmbeddingManager):
    """Embedding manager with caching support."""
    
    def __init__(self, use_openai: bool = True, cache_size: int = 1000):
        """
        Initialize with cache.
        
        Args:
            use_openai: If True, use OpenAI embeddings
            cache_size: Maximum cache size
        """
        super().__init__(use_openai)
        self.cache = EmbeddingCache(max_size=cache_size)
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed query with caching.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        # Check cache first
        cached = self.cache.get(text)
        if cached is not None:
            return cached
        
        # Compute embedding
        embedding = super().embed_query(text)
        
        # Store in cache
        self.cache.set(text, embedding)
        
        return embedding
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self.cache.clear()


# Factory function for easy initialization
def create_embedding_manager(
    use_openai: bool = True,
    use_cache: bool = True,
    cache_size: int = 1000
) -> EmbeddingManager:
    """
    Factory function to create an embedding manager.
    
    Args:
        use_openai: If True, use OpenAI embeddings
        use_cache: If True, use caching
        cache_size: Cache size if caching is enabled
        
    Returns:
        EmbeddingManager instance
    """
    if use_cache:
        return CachedEmbeddingManager(
            use_openai=use_openai,
            cache_size=cache_size
        )
    else:
        return EmbeddingManager(use_openai=use_openai)