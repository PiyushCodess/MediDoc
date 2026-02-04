"""
Vector store module for managing FAISS indices and similarity search.
"""
import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

from ..utils.config import get_settings
from .embeddings import create_embedding_manager


class VectorStoreManager:
    """Manages FAISS vector store for document embeddings."""
    
    def __init__(self, use_openai_embeddings: bool = True):
        """
        Initialize the vector store manager.
        
        Args:
            use_openai_embeddings: Whether to use OpenAI embeddings
        """
        self.settings = get_settings()
        self.embedding_manager = create_embedding_manager(
            use_openai=use_openai_embeddings,
            use_cache=True
        )
        self.vector_store: Optional[FAISS] = None
        self.document_metadata: Dict[str, Any] = {}
        
        # Ensure vector store directory exists
        Path(self.settings.vector_store_path).mkdir(parents=True, exist_ok=True)
    
    def create_vector_store(self, documents: List[Document]) -> FAISS:
        """
        Create a new FAISS vector store from documents.
        
        Args:
            documents: List of Document objects to index
            
        Returns:
            FAISS vector store
        """
        if not documents:
            raise ValueError("Cannot create vector store with empty document list")
        
        # Create FAISS index
        self.vector_store = FAISS.from_documents(
            documents,
            self.embedding_manager.embeddings
        )
        
        # Store metadata
        for doc in documents:
            doc_id = doc.metadata.get('document_id')
            if doc_id and doc_id not in self.document_metadata:
                self.document_metadata[doc_id] = {
                    'filename': doc.metadata.get('source', 'Unknown'),
                    'num_chunks': 0
                }
            if doc_id:
                self.document_metadata[doc_id]['num_chunks'] += 1
        
        return self.vector_store
    
    def add_documents(self, documents: List[Document]):
        """
        Add documents to an existing vector store.
        
        Args:
            documents: List of Document objects to add
        """
        if not documents:
            return
        
        if self.vector_store is None:
            self.create_vector_store(documents)
        else:
            self.vector_store.add_documents(documents)
            
            # Update metadata
            for doc in documents:
                doc_id = doc.metadata.get('document_id')
                if doc_id and doc_id not in self.document_metadata:
                    self.document_metadata[doc_id] = {
                        'filename': doc.metadata.get('source', 'Unknown'),
                        'num_chunks': 0
                    }
                if doc_id:
                    self.document_metadata[doc_id]['num_chunks'] += 1
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter_dict: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> List[Document]:
        """
        Perform similarity search for relevant documents.
        
        Args:
            query: Query string
            k: Number of results to return
            filter_dict: Optional metadata filter
            score_threshold: Optional minimum similarity score
            
        Returns:
            List of relevant Document objects
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        if filter_dict:
            results = self.vector_store.similarity_search(
                query,
                k=k,
                filter=filter_dict
            )
        else:
            results = self.vector_store.similarity_search(query, k=k)
        
        # Apply score threshold if provided
        if score_threshold is not None:
            results_with_scores = self.vector_store.similarity_search_with_score(
                query, k=k
            )
            results = [
                doc for doc, score in results_with_scores
                if score >= score_threshold
            ]
        
        return results
    
    def similarity_search_with_scores(
        self,
        query: str,
        k: int = 4,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search and return scores.
        
        Args:
            query: Query string
            k: Number of results to return
            filter_dict: Optional metadata filter
            
        Returns:
            List of (Document, score) tuples
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        if filter_dict:
            # FAISS doesn't support filtering with scores directly
            # So we filter after retrieval
            results = self.vector_store.similarity_search_with_score(query, k=k*2)
            filtered = [
                (doc, score) for doc, score in results
                if all(doc.metadata.get(key) == value 
                      for key, value in filter_dict.items())
            ]
            return filtered[:k]
        else:
            return self.vector_store.similarity_search_with_score(query, k=k)
    
    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5
    ) -> List[Document]:
        """
        Perform maximum marginal relevance search for diverse results.
        
        Args:
            query: Query string
            k: Number of results to return
            fetch_k: Number of results to fetch before MMR
            lambda_mult: Diversity parameter (0=diverse, 1=relevant)
            
        Returns:
            List of diverse relevant Document objects
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        return self.vector_store.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult
        )
    
    def save(self, index_name: Optional[str] = None):
        """
        Save the vector store to disk.
        
        Args:
            index_name: Optional custom name for the index
        """
        if self.vector_store is None:
            raise ValueError("No vector store to save")
        
        if index_name is None:
            index_name = self.settings.faiss_index_name
        
        # Save FAISS index
        index_path = os.path.join(
            self.settings.vector_store_path,
            index_name
        )
        self.vector_store.save_local(index_path)
        
        # Save metadata separately
        metadata_path = os.path.join(
            self.settings.vector_store_path,
            f"{index_name}_metadata.pkl"
        )
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.document_metadata, f)
    
    def load(self, index_name: Optional[str] = None) -> bool:
        """
        Load a vector store from disk.
        
        Args:
            index_name: Optional custom name for the index
            
        Returns:
            True if loaded successfully, False otherwise
        """
        if index_name is None:
            index_name = self.settings.faiss_index_name
        
        index_path = os.path.join(
            self.settings.vector_store_path,
            index_name
        )
        
        # Check if index exists
        if not os.path.exists(index_path):
            return False
        
        try:
            # Load FAISS index
            self.vector_store = FAISS.load_local(
                index_path,
                self.embedding_manager.embeddings,
                allow_dangerous_deserialization=True
            )
            
            # Load metadata
            metadata_path = os.path.join(
                self.settings.vector_store_path,
                f"{index_name}_metadata.pkl"
            )
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    self.document_metadata = pickle.load(f)
            
            return True
        except Exception as e:
            print(f"Error loading vector store: {e}")
            return False
    
    def delete_document(self, document_id: str):
        """
        Delete all chunks of a document from the vector store.
        
        Note: FAISS doesn't support direct deletion, so this requires
        rebuilding the index without the specified document.
        
        Args:
            document_id: Document ID to delete
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        # Get all documents
        all_docs = self._get_all_documents()
        
        # Filter out documents with matching ID
        remaining_docs = [
            doc for doc in all_docs
            if doc.metadata.get('document_id') != document_id
        ]
        
        # Rebuild vector store
        if remaining_docs:
            self.vector_store = None
            self.document_metadata = {}
            self.create_vector_store(remaining_docs)
        else:
            # No documents left
            self.vector_store = None
            self.document_metadata = {}
    
    def _get_all_documents(self) -> List[Document]:
        """
        Get all documents from the vector store.
        
        Returns:
            List of all Document objects
        """
        if self.vector_store is None:
            return []
        
        # This is a workaround since FAISS doesn't expose all documents directly
        # We perform a dummy search with high k to get many documents
        try:
            docs = self.vector_store.similarity_search("", k=10000)
            return docs
        except:
            return []
    
    def get_document_count(self) -> int:
        """
        Get the total number of document chunks in the vector store.
        
        Returns:
            Number of chunks
        """
        if self.vector_store is None:
            return 0
        
        # FAISS index size
        return self.vector_store.index.ntotal
    
    def get_document_metadata(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific document.
        
        Args:
            document_id: Document ID
            
        Returns:
            Document metadata or None if not found
        """
        return self.document_metadata.get(document_id)
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all documents in the vector store.
        
        Returns:
            List of document metadata dictionaries
        """
        return [
            {
                'document_id': doc_id,
                **metadata
            }
            for doc_id, metadata in self.document_metadata.items()
        ]
    
    def clear(self):
        """Clear the vector store and metadata."""
        self.vector_store = None
        self.document_metadata = {}