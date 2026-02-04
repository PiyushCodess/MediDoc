"""
LLM Chain module for RAG-based question answering using LangChain.
"""
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import LLMChain

from ..utils.config import (
    get_settings,
    CLINICAL_QA_TEMPLATE,
    CLINICAL_SUMMARIZE_TEMPLATE,
    CLINICAL_CONDENSE_QUESTION_TEMPLATE
)
from .vector_store import VectorStoreManager


class ClinicalRAGChain:
    """RAG chain for clinical document question answering."""
    
    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        use_conversation_memory: bool = True
    ):
        """
        Initialize the RAG chain.
        
        Args:
            vector_store_manager: Vector store manager instance
            use_conversation_memory: Whether to maintain conversation history
        """
        self.settings = get_settings()
        self.vector_store_manager = vector_store_manager
        self.use_conversation_memory = use_conversation_memory
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=self.settings.llm_model,
            temperature=self.settings.llm_temperature,
            max_tokens=self.settings.llm_max_tokens,
            openai_api_key=self.settings.openai_api_key
        )
        
        # Create prompts
        self.qa_prompt = PromptTemplate(
            template=CLINICAL_QA_TEMPLATE,
            input_variables=["context", "question"]
        )
        
        self.condense_question_prompt = PromptTemplate(
            template=CLINICAL_CONDENSE_QUESTION_TEMPLATE,
            input_variables=["chat_history", "question"]
        )
        
        # Session management
        self.sessions: Dict[str, ConversationBufferMemory] = {}
    
    def _get_or_create_session(self, session_id: str) -> ConversationBufferMemory:
        """
        Get or create a conversation session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            ConversationBufferMemory for the session
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
        
        return self.sessions[session_id]
    
    def query(
        self,
        question: str,
        session_id: Optional[str] = None,
        top_k: int = 4,
        return_source_documents: bool = True
    ) -> Dict[str, Any]:
        """
        Query the RAG system with a question.
        
        Args:
            question: User question
            session_id: Optional session ID for conversation history
            top_k: Number of documents to retrieve
            return_source_documents: Whether to return source documents
            
        Returns:
            Dictionary with answer and optional source documents
        """
        if self.vector_store_manager.vector_store is None:
            raise ValueError("Vector store not initialized. Please upload documents first.")
        
        # Get retriever
        retriever = self.vector_store_manager.vector_store.as_retriever(
            search_kwargs={"k": top_k}
        )
        
        if session_id and self.use_conversation_memory:
            # Use conversational retrieval chain
            memory = self._get_or_create_session(session_id)
            
            chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,
                memory=memory,
                return_source_documents=return_source_documents,
                combine_docs_chain_kwargs={"prompt": self.qa_prompt},
                condense_question_prompt=self.condense_question_prompt
            )
            
            result = chain({"question": question})
        else:
            # Use simple retrieval QA chain
            chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=return_source_documents,
                chain_type_kwargs={"prompt": self.qa_prompt}
            )
            
            result = chain({"query": question})
        
        # Format response
        response = {
            "answer": result.get("answer") or result.get("result", ""),
            "question": question
        }
        
        if return_source_documents and "source_documents" in result:
            response["source_documents"] = self._format_source_documents(
                result["source_documents"]
            )
        
        return response
    
    def _format_source_documents(
        self,
        source_docs: List[Any]
    ) -> List[Dict[str, Any]]:
        """
        Format source documents for the response.
        
        Args:
            source_docs: List of source Document objects
            
        Returns:
            List of formatted source dictionaries
        """
        formatted_sources = []
        
        for doc in source_docs:
            formatted_sources.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "Unknown"),
                "document_id": doc.metadata.get("document_id", "Unknown")
            })
        
        return formatted_sources
    
    def summarize_document(
        self,
        document_id: str,
        summary_type: str = "comprehensive"
    ) -> str:
        """
        Summarize a document from the vector store.
        
        Args:
            document_id: Document ID to summarize
            summary_type: Type of summary (brief, comprehensive, executive)
            
        Returns:
            Summary text
        """
        if self.vector_store_manager.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        # Retrieve all chunks for the document
        all_docs = self.vector_store_manager.similarity_search(
            query="",  # Dummy query
            k=1000,
            filter_dict={"document_id": document_id}
        )
        
        if not all_docs:
            raise ValueError(f"No document found with ID: {document_id}")
        
        # Combine document chunks
        combined_text = "\n\n".join([doc.page_content for doc in all_docs])
        
        # Truncate if too long (to fit in context)
        max_chars = 8000  # Conservative limit
        if len(combined_text) > max_chars:
            combined_text = combined_text[:max_chars] + "\n\n[Document truncated...]"
        
        # Create summary prompt based on type
        if summary_type == "brief":
            instruction = "Provide a brief 2-3 sentence summary of the key findings."
        elif summary_type == "executive":
            instruction = "Provide an executive summary highlighting the main objectives, methods, results, and conclusions."
        else:  # comprehensive
            instruction = CLINICAL_SUMMARIZE_TEMPLATE
        
        prompt = PromptTemplate(
            template=f"{instruction}\n\nDocument:\n{{text}}\n\nSummary:",
            input_variables=["text"]
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        summary = chain.run(text=combined_text)
        
        return summary.strip()
    
    def get_conversation_history(
        self,
        session_id: str
    ) -> List[Dict[str, str]]:
        """
        Get conversation history for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of conversation messages
        """
        if session_id not in self.sessions:
            return []
        
        memory = self.sessions[session_id]
        messages = memory.load_memory_variables({}).get("chat_history", [])
        
        formatted_history = []
        for msg in messages:
            formatted_history.append({
                "role": "user" if msg.type == "human" else "assistant",
                "content": msg.content
            })
        
        return formatted_history
    
    def clear_session(self, session_id: str):
        """
        Clear conversation history for a session.
        
        Args:
            session_id: Session identifier
        """
        if session_id in self.sessions:
            self.sessions[session_id].clear()
    
    def clear_all_sessions(self):
        """Clear all conversation sessions."""
        self.sessions.clear()


class DocumentAnalyzer:
    """Specialized analyzer for clinical documents."""
    
    def __init__(self, rag_chain: ClinicalRAGChain):
        """
        Initialize the analyzer.
        
        Args:
            rag_chain: ClinicalRAGChain instance
        """
        self.rag_chain = rag_chain
    
    def extract_key_findings(self, document_id: str) -> Dict[str, Any]:
        """
        Extract key findings from a clinical document.
        
        Args:
            document_id: Document ID
            
        Returns:
            Dictionary of key findings
        """
        findings = {}
        
        # Define key questions
        questions = {
            "objectives": "What are the main study objectives?",
            "methods": "What methods and study design were used?",
            "results": "What are the primary results and findings?",
            "safety": "What safety concerns or adverse events were reported?",
            "conclusions": "What are the main conclusions?"
        }
        
        for key, question in questions.items():
            try:
                response = self.rag_chain.query(
                    question=question,
                    top_k=3,
                    return_source_documents=False
                )
                findings[key] = response["answer"]
            except Exception as e:
                findings[key] = f"Error extracting {key}: {str(e)}"
        
        return findings
    
    def compare_documents(
        self,
        question: str,
        document_ids: List[str]
    ) -> Dict[str, str]:
        """
        Compare multiple documents on a specific question.
        
        Args:
            question: Question to ask about each document
            document_ids: List of document IDs to compare
            
        Returns:
            Dictionary mapping document IDs to answers
        """
        comparisons = {}
        
        for doc_id in document_ids:
            try:
                # Query with document filter
                docs = self.rag_chain.vector_store_manager.similarity_search(
                    query=question,
                    k=3,
                    filter_dict={"document_id": doc_id}
                )
                
                if docs:
                    # Use the retrieved docs to answer
                    response = self.rag_chain.query(
                        question=question,
                        top_k=3,
                        return_source_documents=False
                    )
                    comparisons[doc_id] = response["answer"]
                else:
                    comparisons[doc_id] = "No relevant information found"
            except Exception as e:
                comparisons[doc_id] = f"Error: {str(e)}"
        
        return comparisons