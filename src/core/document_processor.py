"""
Document processing module for PDF ingestion and intelligent chunking.
"""
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import PyPDF2
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

from ..utils.config import get_settings
from ..utils.helpers import (
    generate_document_id,
    hash_file,
    sanitize_filename,
    validate_pdf_file,
    extract_clinical_metadata,
    ProgressTracker
)


class DocumentProcessor:
    """Handles PDF document ingestion and processing."""
    
    def __init__(self):
        """Initialize the document processor."""
        self.settings = get_settings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def process_pdf(
        self,
        file_path: str,
        extract_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Process a PDF file into chunks with metadata.
        
        Args:
            file_path: Path to the PDF file
            extract_metadata: Whether to extract clinical metadata
            
        Returns:
            Dictionary containing chunks and metadata
            
        Raises:
            ValueError: If file is not a valid PDF
            FileNotFoundError: If file doesn't exist
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not validate_pdf_file(file_path):
            raise ValueError(f"Invalid PDF file: {file_path}")
        
        # Generate document ID and hash
        document_id = generate_document_id()
        file_hash = hash_file(file_path)
        
        # Extract text from PDF
        text_content, num_pages = self._extract_text_from_pdf(file_path)
        
        # Extract metadata if requested
        clinical_metadata = {}
        if extract_metadata:
            clinical_metadata = extract_clinical_metadata(text_content)
        
        # Create chunks
        chunks = self._create_chunks(
            text_content,
            document_id,
            os.path.basename(file_path)
        )
        
        return {
            "document_id": document_id,
            "file_hash": file_hash,
            "filename": os.path.basename(file_path),
            "num_pages": num_pages,
            "num_chunks": len(chunks),
            "chunks": chunks,
            "clinical_metadata": clinical_metadata,
            "file_size_bytes": os.path.getsize(file_path)
        }
    
    def _extract_text_from_pdf(self, file_path: str) -> tuple[str, int]:
        """
        Extract text from a PDF file using multiple methods for best results.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Tuple of (extracted text, number of pages)
        """
        all_text = []
        num_pages = 0
        
        # Try pdfplumber first (better for tables and complex layouts)
        try:
            with pdfplumber.open(file_path) as pdf:
                num_pages = len(pdf.pages)
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text:
                        # Add page number marker
                        all_text.append(f"\n--- Page {page_num} ---\n")
                        all_text.append(text)
        except Exception as e:
            print(f"pdfplumber extraction failed: {e}, trying PyPDF2...")
            
            # Fallback to PyPDF2
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    num_pages = len(pdf_reader.pages)
                    
                    for page_num, page in enumerate(pdf_reader.pages, 1):
                        text = page.extract_text()
                        if text:
                            all_text.append(f"\n--- Page {page_num} ---\n")
                            all_text.append(text)
            except Exception as e2:
                raise ValueError(f"Failed to extract text from PDF: {e2}")
        
        combined_text = "\n".join(all_text)
        
        if not combined_text.strip():
            raise ValueError("No text could be extracted from the PDF")
        
        return combined_text, num_pages
    
    def _create_chunks(
        self,
        text: str,
        document_id: str,
        filename: str
    ) -> List[Document]:
        """
        Create text chunks with metadata.
        
        Args:
            text: Full document text
            document_id: Unique document identifier
            filename: Original filename
            
        Returns:
            List of Document objects with chunks and metadata
        """
        # Split text into chunks
        text_chunks = self.text_splitter.split_text(text)
        
        # Create Document objects with metadata
        documents = []
        for i, chunk in enumerate(text_chunks):
            # Extract page number from chunk if available
            page_num = self._extract_page_number(chunk)
            
            metadata = {
                "document_id": document_id,
                "source": filename,
                "chunk_id": i,
                "total_chunks": len(text_chunks),
                "page": page_num if page_num else "Unknown"
            }
            
            doc = Document(
                page_content=chunk,
                metadata=metadata
            )
            documents.append(doc)
        
        return documents
    
    def _extract_page_number(self, chunk: str) -> Optional[int]:
        """
        Extract page number from a text chunk if present.
        
        Args:
            chunk: Text chunk
            
        Returns:
            Page number if found, None otherwise
        """
        import re
        
        # Look for page markers like "--- Page 5 ---"
        match = re.search(r'---\s*Page\s+(\d+)\s*---', chunk)
        if match:
            return int(match.group(1))
        
        return None
    
    def extract_tables_from_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract tables from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of dictionaries containing table data and metadata
        """
        tables = []
        
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_tables = page.extract_tables()
                    
                    for table_num, table in enumerate(page_tables, 1):
                        tables.append({
                            "page": page_num,
                            "table_number": table_num,
                            "data": table,
                            "num_rows": len(table),
                            "num_cols": len(table[0]) if table else 0
                        })
        except Exception as e:
            print(f"Failed to extract tables: {e}")
        
        return tables
    
    def save_uploaded_file(self, file_content: bytes, filename: str) -> str:
        """
        Save an uploaded file to the documents directory.
        
        Args:
            file_content: File content as bytes
            filename: Original filename
            
        Returns:
            Path to saved file
        """
        # Sanitize filename
        safe_filename = sanitize_filename(filename)
        
        # Ensure documents directory exists
        docs_dir = Path("data/documents")
        docs_dir.mkdir(parents=True, exist_ok=True)
        
        # Create unique filename if necessary
        file_path = docs_dir / safe_filename
        counter = 1
        while file_path.exists():
            name, ext = safe_filename.rsplit('.', 1)
            file_path = docs_dir / f"{name}_{counter}.{ext}"
            counter += 1
        
        # Write file
        with open(file_path, 'wb') as f:
            f.write(file_content)
        
        return str(file_path)
    
    def batch_process_pdfs(
        self,
        file_paths: List[str],
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple PDF files in batch.
        
        Args:
            file_paths: List of file paths to process
            progress_callback: Optional callback function for progress updates
            
        Returns:
            List of processing results
        """
        results = []
        tracker = ProgressTracker(len(file_paths))
        
        for file_path in file_paths:
            try:
                result = self.process_pdf(file_path)
                results.append({
                    "success": True,
                    "file_path": file_path,
                    "result": result
                })
            except Exception as e:
                results.append({
                    "success": False,
                    "file_path": file_path,
                    "error": str(e)
                })
            
            tracker.update()
            
            if progress_callback:
                progress_callback(
                    current=tracker.current,
                    total=tracker.total,
                    percentage=tracker.get_percentage()
                )
        
        return results
    
    def get_document_stats(self, file_path: str) -> Dict[str, Any]:
        """
        Get statistics about a PDF document without full processing.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary of document statistics
        """
        stats = {
            "file_size_bytes": os.path.getsize(file_path),
            "file_size_mb": os.path.getsize(file_path) / (1024 * 1024),
        }
        
        try:
            with pdfplumber.open(file_path) as pdf:
                stats["num_pages"] = len(pdf.pages)
                
                # Sample first page for quick analysis
                first_page = pdf.pages[0]
                stats["has_images"] = len(first_page.images) > 0
                stats["has_tables"] = len(first_page.extract_tables()) > 0
                
        except Exception as e:
            stats["error"] = str(e)
        
        return stats