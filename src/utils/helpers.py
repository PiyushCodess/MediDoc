"""
Utility helper functions for the Clinical RAG System.
"""
import uuid
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime
import tiktoken


def generate_document_id() -> str:
    """Generate a unique document ID."""
    return str(uuid.uuid4())


def generate_session_id() -> str:
    """Generate a unique session ID."""
    return str(uuid.uuid4())


def hash_file(file_path: str) -> str:
    """Generate SHA-256 hash of a file for deduplication."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Count the number of tokens in a text string.
    
    Args:
        text: Text to count tokens for
        model: Model name to use for encoding
        
    Returns:
        Number of tokens
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    return len(encoding.encode(text))


def truncate_text(text: str, max_tokens: int, model: str = "gpt-4") -> str:
    """
    Truncate text to a maximum number of tokens.
    
    Args:
        text: Text to truncate
        max_tokens: Maximum number of tokens
        model: Model name to use for encoding
        
    Returns:
        Truncated text
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)


def format_sources(sources: List[Dict[str, Any]]) -> str:
    """
    Format source citations for display.
    
    Args:
        sources: List of source documents with metadata
        
    Returns:
        Formatted source string
    """
    if not sources:
        return "No sources available."
    
    formatted = "**Sources:**\n\n"
    for i, source in enumerate(sources, 1):
        metadata = source.get("metadata", {})
        page = metadata.get("page", "N/A")
        doc_name = metadata.get("source", "Unknown Document")
        
        formatted += f"{i}. {doc_name} (Page {page})\n"
        
        # Add excerpt if available
        content = source.get("page_content", "")
        if content:
            excerpt = content[:200] + "..." if len(content) > 200 else content
            formatted += f"   _{excerpt}_\n\n"
    
    return formatted


def extract_clinical_metadata(text: str) -> Dict[str, Optional[str]]:
    """
    Extract clinical metadata from document text using simple pattern matching.
    
    Args:
        text: Document text
        
    Returns:
        Dictionary of extracted metadata
    """
    import re
    
    metadata = {}
    
    # Try to extract NCT number (ClinicalTrials.gov)
    nct_match = re.search(r'NCT\d{8}', text, re.IGNORECASE)
    if nct_match:
        metadata['nct_number'] = nct_match.group(0)
    
    # Try to extract trial phase
    phase_match = re.search(
        r'Phase\s+(I{1,3}|IV|1|2|3|4)',
        text,
        re.IGNORECASE
    )
    if phase_match:
        metadata['trial_phase'] = phase_match.group(0)
    
    # Try to extract study type
    study_types = [
        'randomized controlled trial',
        'observational study',
        'case-control study',
        'cohort study',
        'meta-analysis',
        'systematic review'
    ]
    
    for study_type in study_types:
        if study_type in text.lower():
            metadata['study_type'] = study_type.title()
            break
    
    return metadata


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to prevent security issues.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    import re
    
    # Remove any path components
    filename = filename.split('/')[-1].split('\\')[-1]
    
    # Remove or replace unsafe characters
    filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
    
    # Limit length
    if len(filename) > 255:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        filename = name[:250] + ('.' + ext if ext else '')
    
    return filename


def validate_pdf_file(file_path: str) -> bool:
    """
    Validate that a file is a valid PDF.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if valid PDF, False otherwise
    """
    try:
        with open(file_path, 'rb') as f:
            header = f.read(5)
            return header == b'%PDF-'
    except Exception:
        return False


def format_timestamp(dt: Optional[datetime] = None) -> str:
    """
    Format a datetime as a human-readable string.
    
    Args:
        dt: Datetime to format (defaults to now)
        
    Returns:
        Formatted timestamp string
    """
    if dt is None:
        dt = datetime.now()
    
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def chunk_text_by_tokens(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
    model: str = "gpt-4"
) -> List[str]:
    """
    Chunk text by token count rather than character count.
    
    Args:
        text: Text to chunk
        chunk_size: Maximum tokens per chunk
        overlap: Number of overlapping tokens
        model: Model name for tokenization
        
    Returns:
        List of text chunks
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    tokens = encoding.encode(text)
    chunks = []
    
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunks.append(encoding.decode(chunk_tokens))
        start += chunk_size - overlap
    
    return chunks


class ProgressTracker:
    """Simple progress tracker for document processing."""
    
    def __init__(self, total: int):
        self.total = total
        self.current = 0
        self.start_time = datetime.now()
    
    def update(self, increment: int = 1):
        """Update progress."""
        self.current += increment
    
    def get_percentage(self) -> float:
        """Get current percentage complete."""
        if self.total == 0:
            return 100.0
        return (self.current / self.total) * 100
    
    def get_eta_seconds(self) -> Optional[float]:
        """Get estimated time remaining in seconds."""
        if self.current == 0:
            return None
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        rate = self.current / elapsed
        remaining = self.total - self.current
        
        return remaining / rate if rate > 0 else None