"""
Streamlit frontend for the Clinical Document Q&A System.
"""
import streamlit as st
import sys
import os
from pathlib import Path
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.document_processor import DocumentProcessor
from src.core.vector_store import VectorStoreManager
from src.core.llm_chain import ClinicalRAGChain, DocumentAnalyzer
from src.utils.config import get_settings, ensure_directories
from src.utils.helpers import generate_session_id, format_sources


# Page configuration
st.set_page_config(
    page_title="Clinical Document Q&A",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        color: #1a1a1a;
    }
    .answer-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
        color: #1a1a1a;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
def init_session_state():
    """Initialize Streamlit session state variables."""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.session_id = generate_session_id()
        st.session_state.chat_history = []
        st.session_state.vector_store_ready = False
        st.session_state.uploaded_docs = []
        
        # Initialize components
        ensure_directories()
        st.session_state.settings = get_settings()
        st.session_state.document_processor = DocumentProcessor()
        st.session_state.vector_store_manager = VectorStoreManager(
            use_openai_embeddings=True
        )
        
        # Try to load existing vector store
        loaded = st.session_state.vector_store_manager.load()
        if loaded:
            st.session_state.vector_store_ready = True
            st.session_state.rag_chain = ClinicalRAGChain(
                st.session_state.vector_store_manager
            )
            st.session_state.document_analyzer = DocumentAnalyzer(
                st.session_state.rag_chain
            )
            
            # Load document list
            docs = st.session_state.vector_store_manager.list_documents()
            st.session_state.uploaded_docs = docs


def main():
    """Main application function."""
    init_session_state()
    
    # Header
    st.markdown('<div class="main-header">🏥 MediDoc Q&A </div>', 
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">RAG-powered AI for Clinical Research Documents</div>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Document Management")
        
        # Document upload
        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF Documents",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload clinical trial reports, research papers, or medical documents"
        )
        
        if uploaded_files:
            if st.button("Process Uploaded Documents", type="primary"):
                process_documents(uploaded_files)
        
        st.divider()
        
        # Document list
        st.subheader("Uploaded Documents")
        if st.session_state.uploaded_docs:
            for doc in st.session_state.uploaded_docs:
                with st.expander(f"📄 {doc['filename'][:30]}..."):
                    st.write(f"**Chunks:** {doc['num_chunks']}")
                    st.write(f"**ID:** {doc['document_id'][:8]}...")
        else:
            st.info("No documents uploaded yet")
        
        st.divider()
        
        # Settings
        st.subheader("⚙️ Settings")
        top_k = st.slider("Results per query", 1, 10, 4)
        include_sources = st.checkbox("Show sources", value=True)
        
        if st.button("Clear Conversation"):
            st.session_state.chat_history = []
            if 'rag_chain' in st.session_state:
                st.session_state.rag_chain.clear_session(st.session_state.session_id)
            st.success("Conversation cleared!")
            st.rerun()
    
    # Main content area - use radio buttons instead of tabs to avoid chat_input restriction
    view_mode = st.radio(
        "Select View:",
        ["💬 Ask Questions", "📊 Document Analysis", "📝 Summarization"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    st.divider()
    
    if view_mode == "💬 Ask Questions":
        show_qa_interface(top_k, include_sources)
    elif view_mode == "📊 Document Analysis":
        show_analysis_interface()
    else:
        show_summarization_interface()


def process_documents(uploaded_files):
    """Process uploaded PDF documents."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            status_text.text(f"Processing {uploaded_file.name}...")
            
            # Save file
            file_path = st.session_state.document_processor.save_uploaded_file(
                uploaded_file.read(),
                uploaded_file.name
            )
            
            # Process PDF
            result = st.session_state.document_processor.process_pdf(
                file_path,
                extract_metadata=True
            )
            
            # Add to vector store
            st.session_state.vector_store_manager.add_documents(result["chunks"])
            
            # Initialize RAG chain if needed
            if not st.session_state.vector_store_ready:
                st.session_state.rag_chain = ClinicalRAGChain(
                    st.session_state.vector_store_manager
                )
                st.session_state.document_analyzer = DocumentAnalyzer(
                    st.session_state.rag_chain
                )
                st.session_state.vector_store_ready = True
            
            # Update document list
            st.session_state.uploaded_docs.append({
                'document_id': result['document_id'],
                'filename': result['filename'],
                'num_chunks': result['num_chunks']
            })
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
    
    # Save vector store
    st.session_state.vector_store_manager.save()
    status_text.text("All documents processed successfully!")
    time.sleep(1)
    status_text.empty()
    progress_bar.empty()
    st.success(f"✅ Processed {len(uploaded_files)} document(s)")
    st.rerun()


def show_qa_interface(top_k, include_sources):
    """Show the Q&A interface."""
    st.header("Ask Questions About Your Documents")
    
    if not st.session_state.vector_store_ready:
        st.warning("⚠️ Please upload documents first to start asking questions.")
        
        # Example questions
        st.subheader("Example Questions You Can Ask:")
        st.markdown("""
        - What were the primary endpoints of the study?
        - What side effects were reported during the trial?
        - Summarize the eligibility criteria for participants
        - What was the dosage regimen and treatment schedule?
        - What were the main conclusions of the research?
        """)
        return
    
    # Chat interface
    st.subheader("Conversation")
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            with st.chat_message("user"):
                st.write(message['content'])
        else:
            with st.chat_message("assistant"):
                # Use Streamlit's native info box for better visibility
                st.info(message["content"])
                
                if 'sources' in message and message['sources']:
                    with st.expander("📚 View Sources"):
                        for i, source in enumerate(message['sources'], 1):
                            st.markdown(f"**Source {i}:** {source['source']} (Page {source['page']})")
                            st.caption(f"{source['content'][:200]}...")
                            st.divider()
    
    # Question input
    question = st.chat_input("Ask a question about your documents...")
    
    if question:
        # Add user message to chat
        st.session_state.chat_history.append({
            'role': 'user',
            'content': question
        })
        
        # Show user message
        with st.chat_message("user"):
            st.write(question)
        
        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.rag_chain.query(
                        question=question,
                        session_id=st.session_state.session_id,
                        top_k=top_k,
                        return_source_documents=include_sources
                    )
                    
                    answer = result['answer']
                    sources = result.get('source_documents', [])
                    
                    # Display answer with native Streamlit styling
                    st.info(answer)
                    
                    # Display sources
                    if include_sources and sources:
                        with st.expander("📚 View Sources"):
                            for i, source in enumerate(sources, 1):
                                st.markdown(f"**Source {i}:** {source['source']} (Page {source['page']})")
                                st.caption(f"{source['content'][:200]}...")
                                st.divider()
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': answer,
                        'sources': sources if include_sources else []
                    })
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")


def show_analysis_interface():
    """Show document analysis interface."""
    st.header("Document Analysis Tools")
    
    if not st.session_state.vector_store_ready:
        st.warning("⚠️ Please upload documents first.")
        return
    
    # Select document
    doc_options = {doc['filename']: doc['document_id'] 
                   for doc in st.session_state.uploaded_docs}
    
    if not doc_options:
        st.info("No documents available for analysis")
        return
    
    selected_doc_name = st.selectbox(
        "Select Document to Analyze",
        options=list(doc_options.keys())
    )
    
    selected_doc_id = doc_options[selected_doc_name]
    
    st.divider()
    
    # Analysis options
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 Extract Key Findings", type="primary"):
            with st.spinner("Analyzing document..."):
                try:
                    findings = st.session_state.document_analyzer.extract_key_findings(
                        selected_doc_id
                    )
                    
                    st.subheader("Key Findings")
                    
                    for key, value in findings.items():
                        with st.expander(f"📌 {key.title()}", expanded=True):
                            st.write(value)
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    with col2:
        st.info("More analysis tools coming soon!")


def show_summarization_interface():
    """Show document summarization interface."""
    st.header("Document Summarization")
    
    if not st.session_state.vector_store_ready:
        st.warning("⚠️ Please upload documents first.")
        return
    
    # Select document
    doc_options = {doc['filename']: doc['document_id'] 
                   for doc in st.session_state.uploaded_docs}
    
    if not doc_options:
        st.info("No documents available for summarization")
        return
    
    selected_doc_name = st.selectbox(
        "Select Document",
        options=list(doc_options.keys()),
        key="summary_doc_select"
    )
    
    selected_doc_id = doc_options[selected_doc_name]
    
    # Summary type
    summary_type = st.radio(
        "Summary Type",
        options=["Brief", "Comprehensive", "Executive"],
        horizontal=True
    )
    
    if st.button("Generate Summary", type="primary"):
        with st.spinner("Generating summary..."):
            try:
                summary = st.session_state.rag_chain.summarize_document(
                    document_id=selected_doc_id,
                    summary_type=summary_type.lower()
                )
                
                st.subheader(f"{summary_type} Summary")
                st.info(summary)
                
                # Download button
                st.download_button(
                    label="📥 Download Summary",
                    data=summary,
                    file_name=f"{selected_doc_name}_summary.txt",
                    mime="text/plain"
                )
            
            except Exception as e:
                st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()