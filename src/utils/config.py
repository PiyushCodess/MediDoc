"""
Configuration management for the Clinical RAG System.
"""
import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # OpenAI Configuration
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    
    # LLM Configuration
    llm_model: str = Field(default="gpt-4", env="LLM_MODEL")
    llm_temperature: float = Field(default=0.1, env="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=2000, env="LLM_MAX_TOKENS")
    
    # Embedding Configuration
    embedding_model: str = Field(
        default="text-embedding-ada-002",
        env="EMBEDDING_MODEL"
    )
    
    # Document Processing
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")
    max_file_size_mb: int = Field(default=50, env="MAX_FILE_SIZE_MB")
    
    # Vector Store
    vector_store_path: str = Field(
        default="./data/vector_store",
        env="VECTOR_STORE_PATH"
    )
    faiss_index_name: str = Field(
        default="clinical_documents",
        env="FAISS_INDEX_NAME"
    )
    
    # RAG Configuration
    top_k_results: int = Field(default=4, env="TOP_K_RESULTS")
    similarity_threshold: float = Field(default=0.7, env="SIMILARITY_THRESHOLD")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_reload: bool = Field(default=True, env="API_RELOAD")
    
    # Streamlit Configuration
    streamlit_port: int = Field(default=8501, env="STREAMLIT_PORT")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="./logs/app.log", env="LOG_FILE")
    
    # Session Management
    session_timeout_minutes: int = Field(
        default=30,
        env="SESSION_TIMEOUT_MINUTES"
    )
    
    # Rate Limiting
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_period_seconds: int = Field(
        default=60,
        env="RATE_LIMIT_PERIOD_SECONDS"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


def get_settings() -> Settings:
    """Get application settings singleton."""
    return Settings()


def ensure_directories():
    """Ensure required directories exist."""
    settings = get_settings()
    
    directories = [
        "data/documents",
        "data/vector_store",
        "logs",
        settings.vector_store_path,
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


# Clinical-specific prompts and templates
CLINICAL_QA_TEMPLATE = """You are a knowledgeable medical AI assistant helping healthcare professionals and researchers analyze clinical documents.

Use the following pieces of context from clinical documents to answer the question. 
If you don't know the answer based on the provided context, say so clearly.
Always cite the source document for your answers.

Important guidelines:
- Be precise and accurate with medical terminology
- Cite specific sections or page numbers when possible
- If the information is about medications, dosages, or treatments, be especially careful
- Acknowledge uncertainty when appropriate
- Do not make up information not present in the context

Context:
{context}

Question: {question}

Answer (with source citations):"""


CLINICAL_SUMMARIZE_TEMPLATE = """You are a medical AI assistant specialized in summarizing clinical research documents.

Summarize the following clinical document excerpt, focusing on:
1. Study objectives and design
2. Patient population and eligibility criteria
3. Key findings and results
4. Safety profile and adverse events
5. Conclusions and clinical implications

Be concise but comprehensive. Use medical terminology appropriately.

Document excerpt:
{text}

Summary:"""


CLINICAL_CONDENSE_QUESTION_TEMPLATE = """Given the following conversation history and a follow-up question, 
rephrase the follow-up question to be a standalone question that captures all relevant context.
Preserve medical terminology and specificity.

Chat History:
{chat_history}

Follow-up Question: {question}

Standalone Question:"""


# Medical document metadata fields
CLINICAL_METADATA_FIELDS = [
    "title",
    "authors",
    "publication_date",
    "journal",
    "study_type",
    "trial_phase",
    "drug_name",
    "indication",
    "nct_number",  # ClinicalTrials.gov identifier
]