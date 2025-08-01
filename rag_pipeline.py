"""
Production-Ready RAG Pipeline using Google Gemini API

This module contains all the core RAG (Retrieval-Augmented Generation) logic,
completely decoupled from the UI. It handles document processing, embedding creation,
vector storage, and conversational chain setup using LangChain and Google's Gemini models.
"""

import os
import tempfile
from typing import List, Optional, Tuple, Any
from pathlib import Path

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage


class RAGPipeline:
    """
    A production-ready RAG pipeline that processes documents, creates embeddings,
    and provides conversational question-answering capabilities using Google Gemini.
    """
    
    def __init__(self, google_api_key: str):
        """
        Initialize the RAG pipeline with Google API credentials.
        
        Args:
            google_api_key: The Google API key for accessing Gemini services
            
        Raises:
            ValueError: If the API key is empty or None
        """
        if not google_api_key or not google_api_key.strip():
            raise ValueError("Google API key cannot be empty")
            
        self.google_api_key = google_api_key
        self.embeddings = None
        self.vectorstore = None
        self.retrieval_chain = None
        self.chat_history = []
        
        # Initialize embeddings model
        self._initialize_embeddings()
    
    def _initialize_embeddings(self) -> None:
        """Initialize the Google Generative AI embeddings model."""
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=self.google_api_key
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize embeddings model: {str(e)}")
    
    def process_documents(self, uploaded_files: List[Any]) -> Tuple[bool, str]:
        """
        Process uploaded PDF files and create a vector store.
        
        Args:
            uploaded_files: List of uploaded file objects from Streamlit
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        if not uploaded_files:
            return False, "No files provided for processing"
        
        try:
            # Load and split documents
            documents = self._load_and_split_documents(uploaded_files)
            
            if not documents:
                return False, "No content could be extracted from the provided files"
            
            # Create vector store
            self._create_vector_store(documents)
            
            # Create retrieval chain
            self._create_retrieval_chain()
            
            # Reset chat history for new documents
            self.chat_history = []
            
            return True, f"Successfully processed {len(documents)} document chunks from {len(uploaded_files)} files"
            
        except Exception as e:
            return False, f"Error processing documents: {str(e)}"
    
    def _load_and_split_documents(self, uploaded_files: List[Any]) -> List[Document]:
        """
        Load PDF files and split them into chunks for processing.
        
        Args:
            uploaded_files: List of uploaded file objects
            
        Returns:
            List of Document objects containing the split text chunks
        """
        documents = []
        
        # Initialize text splitter with optimal parameters for RAG
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Reasonable chunk size for embedding models
            chunk_overlap=200,  # Overlap to maintain context between chunks
            length_function=len,
            separators=["\n\n", "\n", " ", ""]  # Hierarchical splitting
        )
        
        for uploaded_file in uploaded_files:
            try:
                # Create temporary file to work with PyPDFLoader
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                # Load PDF using LangChain's PyPDFLoader
                loader = PyPDFLoader(tmp_file_path)
                pdf_documents = loader.load()
                
                # Add source metadata to track which file each chunk came from
                for doc in pdf_documents:
                    doc.metadata['source_file'] = uploaded_file.name
                
                # Split documents into chunks
                split_docs = text_splitter.split_documents(pdf_documents)
                documents.extend(split_docs)
                
                # Clean up temporary file
                os.unlink(tmp_file_path)
                
            except Exception as e:
                # Log the error but continue with other files
                print(f"Warning: Failed to process {uploaded_file.name}: {str(e)}")
                continue
        
        return documents
    
    def _create_vector_store(self, documents: List[Document]) -> None:
        """
        Create a ChromaDB vector store from the processed documents.
        
        Args:
            documents: List of Document objects to store
        """
        try:
            # Create ChromaDB vector store with the documents
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name="rag_documents"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create vector store: {str(e)}")
    
    def _create_retrieval_chain(self) -> None:
        """Create the conversational retrieval chain using LangChain."""
        try:
            # Initialize the Gemini chat model with compatibility settings
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro-latest",
                google_api_key=self.google_api_key,
                temperature=0.1,  # Low temperature for more focused responses
                convert_system_message_to_human=True  # Required for Gemini compatibility
            )
            
            # Create retriever from vector store
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}  # Retrieve top 4 most similar chunks
            )
            
            # Contextualize question prompt - helps maintain conversation context
            contextualize_q_system_prompt = """
            Given a chat history and the latest user question which might reference context in the chat history,
            formulate a standalone question which can be understood without the chat history.
            Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
            """
            
            contextualize_q_prompt = ChatPromptTemplate.from_messages([
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            
            # Create history-aware retriever
            history_aware_retriever = create_history_aware_retriever(
                llm, retriever, contextualize_q_prompt
            )
            
            # QA system prompt - defines how the model should answer questions
            qa_system_prompt = """
            You are an assistant for question-answering tasks. Use the following pieces of retrieved context 
            to answer the question. If you don't know the answer, just say that you don't know. 
            Use three sentences maximum and keep the answer concise.
            
            Context: {context}
            """
            
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            
            # Create the question-answer chain
            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
            
            # Create the final retrieval chain
            self.retrieval_chain = create_retrieval_chain(
                history_aware_retriever, question_answer_chain
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to create retrieval chain: {str(e)}")
    
    def get_response(self, question: str) -> Tuple[bool, str, List[Document]]:
        """
        Get a response to a user question using the RAG pipeline.
        
        Args:
            question: The user's question
            
        Returns:
            Tuple of (success: bool, answer: str, source_documents: List[Document])
        """
        if not self.retrieval_chain:
            return False, "Please upload and process documents first.", []
        
        if not question or not question.strip():
            return False, "Please provide a valid question.", []
        
        try:
            # Invoke the retrieval chain with the question and chat history
            response = self.retrieval_chain.invoke({
                "input": question,
                "chat_history": self.chat_history
            })
            
            # Extract answer and source documents
            answer = response.get("answer", "I couldn't generate an answer.")
            source_docs = response.get("context", [])
            
            # Update chat history
            self.chat_history.extend([
                HumanMessage(content=question),
                AIMessage(content=answer)
            ])
            
            # Keep chat history manageable (last 10 exchanges)
            if len(self.chat_history) > 20:  # 10 human + 10 AI messages
                self.chat_history = self.chat_history[-20:]
            
            return True, answer, source_docs
            
        except Exception as e:
            return False, f"Error generating response: {str(e)}", []
    
    def clear_chat_history(self) -> None:
        """Clear the conversation history."""
        self.chat_history = []
    
    def get_chat_history(self) -> List[dict]:
        """
        Get the current chat history in a UI-friendly format.
        
        Returns:
            List of dictionaries with 'role' and 'content' keys
        """
        history = []
        for message in self.chat_history:
            if isinstance(message, HumanMessage):
                history.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                history.append({"role": "assistant", "content": message.content})
        return history
    
    def is_ready(self) -> bool:
        """Check if the pipeline is ready to answer questions."""
        return self.retrieval_chain is not None


def validate_google_api_key(api_key: str) -> Tuple[bool, str]:
    """
    Validate a Google API key by testing the embeddings model.
    
    Args:
        api_key: The API key to validate
        
    Returns:
        Tuple of (is_valid: bool, message: str)
    """
    if not api_key or not api_key.strip():
        return False, "API key cannot be empty"
    
    try:
        # Test the API key by initializing the embeddings model
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        
        # Test with a simple embedding request
        embeddings.embed_query("test")
        return True, "API key is valid"
        
    except Exception as e:
        return False, f"Invalid API key: {str(e)}"