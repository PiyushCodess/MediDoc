"""
FastAPI backend for the Clinical RAG System.
"""
import time
import os
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..models.schemas import (
    QueryRequest,
    QueryResponse,
    SummarizeRequest,
    SummarizeResponse,
    UploadResponse,
    DocumentListResponse,
    DocumentInfo,
    ErrorResponse,
    HealthCheckResponse
)
from ..core.document_processor import DocumentProcessor
from ..core.vector_store import VectorStoreManager
from ..core.llm_chain import ClinicalRAGChain
from ..utils.config import get_settings, ensure_directories
from ..utils.helpers import generate_session_id, format_timestamp

# Initialize settings and directories
settings = get_settings()
ensure_directories()

# Initialize app
app = FastAPI(
    title="Clinical Document Q&A System",
    description="RAG-based system for querying clinical documents",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
document_processor = DocumentProcessor()
vector_store_manager = VectorStoreManager(use_openai_embeddings=True)
rag_chain = None  # Will be initialized after vector store is loaded

# Try to load existing vector store on startup
@app.on_event("startup")
async def startup_event():
    """Load existing vector store if available."""
    global rag_chain
    
    loaded = vector_store_manager.load()
    if loaded:
        print("✓ Loaded existing vector store")
        rag_chain = ClinicalRAGChain(vector_store_manager)
    else:
        print("⚠ No existing vector store found. Upload documents to get started.")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "Clinical Document Q&A System API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/api/v1/health", response_model=HealthCheckResponse, tags=["System"])
async def health_check():
    """Check system health status."""
    vector_store_ready = vector_store_manager.vector_store is not None
    llm_ready = rag_chain is not None
    
    return HealthCheckResponse(
        status="healthy" if (vector_store_ready and llm_ready) else "degraded",
        vector_store_ready=vector_store_ready,
        llm_ready=llm_ready
    )


@app.post("/api/v1/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Upload and process a PDF document.
    
    Args:
        file: PDF file to upload
        
    Returns:
        Upload response with document metadata
    """
    global rag_chain
    
    start_time = time.time()
    
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    
    # Check file size
    file_content = await file.read()
    file_size_mb = len(file_content) / (1024 * 1024)
    
    if file_size_mb > settings.max_file_size_mb:
        raise HTTPException(
            status_code=400,
            detail=f"File size exceeds maximum of {settings.max_file_size_mb}MB"
        )
    
    try:
        # Save uploaded file
        file_path = document_processor.save_uploaded_file(
            file_content,
            file.filename
        )
        
        # Process PDF
        result = document_processor.process_pdf(file_path, extract_metadata=True)
        
        # Add to vector store
        vector_store_manager.add_documents(result["chunks"])
        
        # Initialize RAG chain if not already done
        if rag_chain is None:
            rag_chain = ClinicalRAGChain(vector_store_manager)
        
        # Save vector store in background
        if background_tasks:
            background_tasks.add_task(vector_store_manager.save)
        else:
            vector_store_manager.save()
        
        processing_time = time.time() - start_time
        
        return UploadResponse(
            document_id=result["document_id"],
            filename=result["filename"],
            message="Document uploaded and processed successfully",
            num_chunks=result["num_chunks"],
            processing_time_seconds=round(processing_time, 2)
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )


@app.post("/api/v1/query", response_model=QueryResponse, tags=["Query"])
async def query_documents(request: QueryRequest):
    """
    Query documents with a question.
    
    Args:
        request: Query request with question and parameters
        
    Returns:
        Query response with answer and sources
    """
    if rag_chain is None:
        raise HTTPException(
            status_code=400,
            detail="No documents uploaded. Please upload documents first."
        )
    
    start_time = time.time()
    
    try:
        # Generate session ID if not provided
        session_id = request.session_id or generate_session_id()
        
        # Query the RAG system
        result = rag_chain.query(
            question=request.question,
            session_id=session_id if session_id else None,
            top_k=request.top_k or settings.top_k_results,
            return_source_documents=request.include_sources
        )
        
        processing_time = time.time() - start_time
        
        # Format sources
        sources = []
        if request.include_sources and "source_documents" in result:
            sources = result["source_documents"]
        
        return QueryResponse(
            answer=result["answer"],
            sources=sources,
            session_id=session_id,
            processing_time_seconds=round(processing_time, 2)
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@app.post("/api/v1/summarize", response_model=SummarizeResponse, tags=["Analysis"])
async def summarize_document(request: SummarizeRequest):
    """
    Summarize a document.
    
    Args:
        request: Summarization request
        
    Returns:
        Summary response
    """
    if rag_chain is None:
        raise HTTPException(
            status_code=400,
            detail="No documents available"
        )
    
    start_time = time.time()
    
    try:
        # Get document metadata
        metadata = vector_store_manager.get_document_metadata(request.document_id)
        if not metadata:
            raise HTTPException(
                status_code=404,
                detail=f"Document not found: {request.document_id}"
            )
        
        # Generate summary
        summary = rag_chain.summarize_document(
            document_id=request.document_id,
            summary_type=request.summary_type
        )
        
        processing_time = time.time() - start_time
        
        return SummarizeResponse(
            summary=summary,
            document_id=request.document_id,
            summary_type=request.summary_type,
            processing_time_seconds=round(processing_time, 2),
            num_chunks_processed=metadata.get("num_chunks", 0)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating summary: {str(e)}"
        )


@app.get("/api/v1/documents", response_model=DocumentListResponse, tags=["Documents"])
async def list_documents():
    """
    List all uploaded documents.
    
    Returns:
        List of document information
    """
    documents = vector_store_manager.list_documents()
    
    document_infos = []
    for doc in documents:
        doc_info = DocumentInfo(
            document_id=doc["document_id"],
            filename=doc["filename"],
            upload_date=format_timestamp(),  # Would need to store this
            num_chunks=doc["num_chunks"],
            file_size_mb=0.0  # Would need to store this
        )
        document_infos.append(doc_info)
    
    return DocumentListResponse(
        documents=document_infos,
        total_count=len(document_infos)
    )


@app.delete("/api/v1/documents/{document_id}", tags=["Documents"])
async def delete_document(document_id: str, background_tasks: BackgroundTasks):
    """
    Delete a document from the vector store.
    
    Args:
        document_id: Document ID to delete
        
    Returns:
        Success message
    """
    try:
        # Check if document exists
        metadata = vector_store_manager.get_document_metadata(document_id)
        if not metadata:
            raise HTTPException(
                status_code=404,
                detail=f"Document not found: {document_id}"
            )
        
        # Delete from vector store
        vector_store_manager.delete_document(document_id)
        
        # Save vector store in background
        background_tasks.add_task(vector_store_manager.save)
        
        return {"message": f"Document {document_id} deleted successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting document: {str(e)}"
        )


@app.get("/api/v1/stats", tags=["System"])
async def get_statistics():
    """Get system statistics."""
    return {
        "total_documents": len(vector_store_manager.list_documents()),
        "total_chunks": vector_store_manager.get_document_count(),
        "vector_store_path": settings.vector_store_path,
        "embedding_model": settings.embedding_model,
        "llm_model": settings.llm_model
    }


@app.post("/api/v1/sessions/{session_id}/clear", tags=["Sessions"])
async def clear_session(session_id: str):
    """
    Clear conversation history for a session.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Success message
    """
    if rag_chain is None:
        raise HTTPException(
            status_code=400,
            detail="RAG chain not initialized"
        )
    
    rag_chain.clear_session(session_id)
    return {"message": f"Session {session_id} cleared successfully"}


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            error_code=f"HTTP_{exc.status_code}"
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            error_code="INTERNAL_ERROR"
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload
    )