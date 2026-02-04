"""
Pydantic models and schemas for the Clinical RAG System.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator


class DocumentMetadata(BaseModel):
    """Metadata for a clinical document."""
    
    document_id: str
    filename: str
    file_hash: str
    upload_date: datetime = Field(default_factory=datetime.now)
    file_size_bytes: int
    num_pages: Optional[int] = None
    num_chunks: int = 0
    
    # Clinical-specific metadata
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    publication_date: Optional[str] = None
    journal: Optional[str] = None
    study_type: Optional[str] = None
    trial_phase: Optional[str] = None
    drug_name: Optional[str] = None
    indication: Optional[str] = None
    nct_number: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "123e4567-e89b-12d3-a456-426614174000",
                "filename": "clinical_trial_phase3.pdf",
                "file_hash": "a1b2c3d4...",
                "file_size_bytes": 1024000,
                "num_pages": 45,
                "num_chunks": 120,
                "title": "Phase III Trial of Drug X",
                "trial_phase": "Phase III",
                "nct_number": "NCT12345678"
            }
        }


class QueryRequest(BaseModel):
    """Request model for querying documents."""
    
    question: str = Field(..., min_length=1, max_length=1000)
    session_id: Optional[str] = None
    document_ids: Optional[List[str]] = None
    top_k: Optional[int] = Field(default=4, ge=1, le=10)
    include_sources: bool = True
    
    @validator('question')
    def validate_question(cls, v):
        """Ensure question is not just whitespace."""
        if not v.strip():
            raise ValueError("Question cannot be empty or whitespace only")
        return v.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What were the primary endpoints of the study?",
                "session_id": "session_123",
                "top_k": 4,
                "include_sources": True
            }
        }


class QueryResponse(BaseModel):
    """Response model for query results."""
    
    answer: str
    sources: List[Dict[str, Any]] = []
    session_id: str
    processing_time_seconds: float
    tokens_used: Optional[int] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "The primary endpoints were...",
                "sources": [
                    {
                        "document_id": "doc_123",
                        "page": 12,
                        "content": "..."
                    }
                ],
                "session_id": "session_123",
                "processing_time_seconds": 2.5
            }
        }


class SummarizeRequest(BaseModel):
    """Request model for document summarization."""
    
    document_id: str
    summary_type: str = Field(
        default="comprehensive",
        pattern="^(brief|comprehensive|executive)$"
    )
    max_length: Optional[int] = Field(default=500, ge=100, le=2000)
    focus_areas: Optional[List[str]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "doc_123",
                "summary_type": "comprehensive",
                "max_length": 500,
                "focus_areas": ["efficacy", "safety", "endpoints"]
            }
        }


class SummarizeResponse(BaseModel):
    """Response model for summarization results."""
    
    summary: str
    document_id: str
    summary_type: str
    processing_time_seconds: float
    num_chunks_processed: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "summary": "This Phase III clinical trial...",
                "document_id": "doc_123",
                "summary_type": "comprehensive",
                "processing_time_seconds": 5.2,
                "num_chunks_processed": 45
            }
        }


class UploadResponse(BaseModel):
    """Response model for document upload."""
    
    document_id: str
    filename: str
    message: str
    num_chunks: int
    processing_time_seconds: float
    metadata: Optional[DocumentMetadata] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "doc_123",
                "filename": "trial_report.pdf",
                "message": "Document uploaded successfully",
                "num_chunks": 78,
                "processing_time_seconds": 12.3
            }
        }


class DocumentInfo(BaseModel):
    """Information about a document in the system."""
    
    document_id: str
    filename: str
    upload_date: datetime
    num_chunks: int
    file_size_mb: float
    metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "doc_123",
                "filename": "study_results.pdf",
                "upload_date": "2024-01-15T10:30:00",
                "num_chunks": 95,
                "file_size_mb": 2.5
            }
        }


class DocumentListResponse(BaseModel):
    """Response model for listing documents."""
    
    documents: List[DocumentInfo]
    total_count: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "documents": [],
                "total_count": 10
            }
        }


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""
    
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    sources: Optional[List[Dict[str, Any]]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "role": "user",
                "content": "What is the dosage regimen?",
                "timestamp": "2024-01-15T10:30:00"
            }
        }


class ChatSession(BaseModel):
    """A chat session with conversation history."""
    
    session_id: str
    messages: List[ChatMessage] = []
    created_at: datetime = Field(default_factory=datetime.now)
    last_activity: datetime = Field(default_factory=datetime.now)
    document_ids: List[str] = []
    
    def add_message(self, message: ChatMessage):
        """Add a message to the session."""
        self.messages.append(message)
        self.last_activity = datetime.now()
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "session_123",
                "messages": [],
                "created_at": "2024-01-15T10:00:00",
                "last_activity": "2024-01-15T10:30:00",
                "document_ids": ["doc_123", "doc_456"]
            }
        }


class ErrorResponse(BaseModel):
    """Standard error response."""
    
    error: str
    detail: Optional[str] = None
    error_code: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "Document not found",
                "detail": "No document with ID doc_123 exists",
                "error_code": "DOC_NOT_FOUND"
            }
        }


class HealthCheckResponse(BaseModel):
    """Health check response."""
    
    status: str
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = "1.0.0"
    vector_store_ready: bool
    llm_ready: bool
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-15T10:30:00",
                "version": "1.0.0",
                "vector_store_ready": True,
                "llm_ready": True
            }
        }