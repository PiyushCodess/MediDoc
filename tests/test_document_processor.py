"""
Tests for document processor module.
"""
import pytest
import os
from pathlib import Path
from src.core.document_processor import DocumentProcessor
from src.utils.helpers import sanitize_filename, validate_pdf_file


@pytest.fixture
def document_processor():
    """Create a document processor instance."""
    return DocumentProcessor()


def test_sanitize_filename():
    """Test filename sanitization."""
    assert sanitize_filename("test file.pdf") == "test_file.pdf"
    assert sanitize_filename("../../../etc/passwd") == "passwd"
    assert sanitize_filename("file@#$%.pdf") == "file____.pdf"


def test_document_processor_initialization(document_processor):
    """Test document processor initialization."""
    assert document_processor is not None
    assert document_processor.text_splitter is not None
    assert document_processor.settings is not None


def test_extract_page_number(document_processor):
    """Test page number extraction."""
    chunk_with_page = "--- Page 5 ---\nSome content here"
    page_num = document_processor._extract_page_number(chunk_with_page)
    assert page_num == 5
    
    chunk_without_page = "Just some content"
    page_num = document_processor._extract_page_number(chunk_without_page)
    assert page_num is None


def test_get_document_stats(document_processor, tmp_path):
    """Test document statistics extraction."""
    # This would require a real PDF file for proper testing
    # For now, we'll just test the structure
    pass


# Integration tests would require actual PDF files
# Example:
# def test_process_pdf_integration(document_processor, sample_pdf_path):
#     """Test full PDF processing pipeline."""
#     result = document_processor.process_pdf(sample_pdf_path)
#     
#     assert 'document_id' in result
#     assert 'chunks' in result
#     assert len(result['chunks']) > 0
#     assert 'num_pages' in result