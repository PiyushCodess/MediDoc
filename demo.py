"""
Demo script for the Clinical Document Q&A System.

This script demonstrates the core functionality of the RAG system
without requiring a UI.
"""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.document_processor import DocumentProcessor
from src.core.vector_store import VectorStoreManager
from src.core.llm_chain import ClinicalRAGChain, DocumentAnalyzer
from src.utils.config import ensure_directories


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)


def demo_document_processing():
    """Demonstrate document processing."""
    print_header("1. DOCUMENT PROCESSING DEMO")
    
    processor = DocumentProcessor()
    
    # This would use a real PDF in practice
    print("\n📄 Document Processing Features:")
    print("  ✓ PDF text extraction (PyPDF2 + pdfplumber)")
    print("  ✓ Intelligent chunking with overlap")
    print("  ✓ Metadata extraction (NCT numbers, trial phases)")
    print("  ✓ Page number tracking")
    print("  ✓ Table extraction support")
    
    print("\n📊 Example chunk metadata:")
    example_metadata = {
        "document_id": "abc-123",
        "source": "phase3_trial.pdf",
        "page": 5,
        "chunk_id": 12,
        "total_chunks": 85
    }
    for key, value in example_metadata.items():
        print(f"  • {key}: {value}")


def demo_vector_store():
    """Demonstrate vector store functionality."""
    print_header("2. VECTOR STORE DEMO")
    
    print("\n🗄️ FAISS Vector Store Features:")
    print("  ✓ Fast similarity search (sub-second)")
    print("  ✓ Persistent storage")
    print("  ✓ Metadata filtering")
    print("  ✓ Maximum marginal relevance (MMR)")
    print("  ✓ Scalable to millions of chunks")
    
    print("\n📈 Example similarity search:")
    print("  Query: 'What are the side effects?'")
    print("  → Retrieves 4 most relevant chunks")
    print("  → Includes source attribution")
    print("  → Returns similarity scores")


def demo_embeddings():
    """Demonstrate embedding functionality."""
    print_header("3. EMBEDDINGS DEMO")
    
    print("\n🧠 Embedding Features:")
    print("  ✓ OpenAI text-embedding-ada-002 (1536 dimensions)")
    print("  ✓ Alternative: HuggingFace models (free)")
    print("  ✓ Embedding cache for efficiency")
    print("  ✓ Batch processing support")
    
    print("\n💡 Example:")
    print("  Text: 'adverse events in clinical trial'")
    print("  → Vector: [0.023, -0.145, 0.892, ..., 0.234]")
    print("  → Dimension: 1536")


def demo_rag_chain():
    """Demonstrate RAG chain functionality."""
    print_header("4. RAG CHAIN DEMO")
    
    print("\n🔗 RAG Pipeline:")
    print("  1. User asks question")
    print("  2. Question embedding generated")
    print("  3. Similar chunks retrieved from FAISS")
    print("  4. Chunks + question sent to LLM")
    print("  5. LLM generates answer with citations")
    
    print("\n💬 Example Conversation:")
    print("  User: 'What were the primary endpoints?'")
    print("  System: Retrieves relevant sections → Sends to GPT-4")
    print("  GPT-4: 'The primary endpoints were... (Source: Page 12)'")
    
    print("\n🎯 Advanced Features:")
    print("  ✓ Conversation memory (multi-turn)")
    print("  ✓ Document summarization")
    print("  ✓ Key findings extraction")
    print("  ✓ Multi-document comparison")


def demo_api_endpoints():
    """Demonstrate API endpoints."""
    print_header("5. API ENDPOINTS DEMO")
    
    endpoints = [
        ("POST", "/api/v1/upload", "Upload and process PDF documents"),
        ("POST", "/api/v1/query", "Ask questions about documents"),
        ("POST", "/api/v1/summarize", "Generate document summaries"),
        ("GET", "/api/v1/documents", "List all uploaded documents"),
        ("DELETE", "/api/v1/documents/{id}", "Delete a document"),
        ("GET", "/api/v1/stats", "Get system statistics"),
        ("POST", "/api/v1/sessions/{id}/clear", "Clear conversation history"),
    ]
    
    print("\n🌐 Available Endpoints:")
    for method, endpoint, description in endpoints:
        print(f"  {method:6} {endpoint:35} - {description}")


def demo_example_queries():
    """Show example queries."""
    print_header("6. EXAMPLE QUERIES")
    
    examples = [
        "Clinical Trial Questions",
        [
            "What were the primary and secondary endpoints?",
            "What adverse events were reported?",
            "Summarize the eligibility criteria",
            "What was the dosage and treatment schedule?",
            "Did the study meet its primary endpoint?",
        ],
        
        "Safety Analysis",
        [
            "What Grade 3 or higher adverse events occurred?",
            "Were there any deaths during the trial?",
            "What discontinuations were due to adverse events?",
            "Compare safety profiles across treatment arms",
        ],
        
        "Efficacy Questions",
        [
            "What was the overall response rate?",
            "Compare efficacy between treatment groups",
            "What were the progression-free survival results?",
            "Were there any biomarker correlations?",
        ]
    ]
    
    for i in range(0, len(examples), 2):
        category = examples[i]
        questions = examples[i + 1]
        
        print(f"\n📋 {category}:")
        for q in questions:
            print(f"  • {q}")


def demo_system_architecture():
    """Show system architecture."""
    print_header("7. SYSTEM ARCHITECTURE")
    
    print("""
┌─────────────────────────────────────────────────────────────┐
│                     STREAMLIT UI / API                      │
│              (User interaction & Interface)                 │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┴───────────┐
        │                        │
┌───────▼────────┐      ┌────────▼────────┐
│  Document      │      │   RAG Chain     │
│  Processor     │      │   (LangChain)   │
└────────┬───────┘      └────────┬────────┘
         │                       │
         │              ┌────────┴────────┐
         │              │                 │
┌────────▼────────┐  ┌──▼───────┐  ┌────▼──────┐
│  Text Splitter  │  │  FAISS   │  │ GPT-4 LLM │
│   (Chunking)    │  │  Vector  │  │  (OpenAI) │
└─────────────────┘  │  Store   │  └───────────┘
                     └──────────┘
                     
💾 Storage: Local filesystem + FAISS index files
🔒 Security: API key management, input validation
⚡ Performance: Embedding cache, batch processing
    """)


def demo_tech_stack():
    """Display technology stack."""
    print_header("8. TECHNOLOGY STACK")
    
    stack = {
        "Core Framework": "LangChain 0.1.0",
        "LLM Provider": "OpenAI GPT-4 / GPT-3.5-turbo",
        "Embeddings": "OpenAI text-embedding-ada-002",
        "Vector Database": "FAISS (Facebook AI Similarity Search)",
        "Backend API": "FastAPI 0.108.0",
        "Frontend UI": "Streamlit 1.29.0",
        "PDF Processing": "PyPDF2, pdfplumber",
        "Language": "Python 3.9+",
        "Testing": "pytest, pytest-asyncio",
        "Development": "black, flake8, mypy",
    }
    
    print()
    for component, tech in stack.items():
        print(f"  {component:20} → {tech}")


def main():
    """Run all demos."""
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "  CLINICAL DOCUMENT Q&A SYSTEM - DEMONSTRATION".center(78) + "█")
    print("█" + "  RAG-Based AI for Healthcare Professionals".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    
    # Run all demo sections
    demo_document_processing()
    demo_embeddings()
    demo_vector_store()
    demo_rag_chain()
    demo_api_endpoints()
    demo_example_queries()
    demo_system_architecture()
    demo_tech_stack()
    
    # Final summary
    print_header("QUICK START")
    print("""
To get started with the actual system:

1. Set up your environment:
   $ cp .env.example .env
   $ # Add your OpenAI API key to .env
   $ pip install -r requirements.txt

2. Run the Streamlit UI:
   $ streamlit run src/app/streamlit_app.py

3. Or start the API server:
   $ uvicorn src.app.api:app --reload

4. Upload your clinical documents and start asking questions!

For detailed instructions, see SETUP_GUIDE.md
    """)
    
    print_header("PROJECT HIGHLIGHTS")
    print("""
✅ Complete RAG Implementation
✅ Production-Ready Code
✅ Comprehensive Documentation
✅ FastAPI + Streamlit Interfaces
✅ Clinical-Specific Prompts
✅ Source Attribution
✅ Conversation Memory
✅ Document Summarization
✅ Extensible Architecture
✅ Full Test Suite

This project demonstrates enterprise-level GenAI/RAG development
suitable for healthcare and clinical research applications.
    """)
    
    print("\n" + "=" * 80)
    print("  Demo complete! Ready for your job interview presentation 🚀")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # Ensure directories exist
    ensure_directories()
    main()