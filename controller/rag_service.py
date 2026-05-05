"""
RAG Service for retrieving and augmenting generation with expense data.
Uses FAISS for vector storage and Hugging Face models for embeddings and LLM.
"""

import os
import json
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

# Try to import optional dependencies with fallbacks
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    print("Warning: FAISS not available. RAG functionality will be limited.")
    FAISS_AVAILABLE = False
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Warning: sentence-transformers not available. Using basic embeddings.")
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Warning: transformers not available. LLM generation disabled.")
    TRANSFORMERS_AVAILABLE = False
    pipeline = None

try:
    from langchain.schema import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    print("Warning: langchain not available. Using basic document handling.")
    LANGCHAIN_AVAILABLE = False
    
    # Fallback Document class
    class Document:
        def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
            self.page_content = page_content
            self.metadata = metadata or {}

from models.models import User, Expense

class RAGService:
    """
    Retrieval-Augmented Generation service for expense chatbot.
    Indexes user and expense data from database and uses it for context-aware responses.
    """
    
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 llm_model: str = "gpt2",
                 index_path: str = "local/faiss_index",
                 data_path: str = "local/documents.pkl"):
        """
        Initialize RAG service.
        
        Args:
            embedding_model: Hugging Face embedding model name
            llm_model: Hugging Face LLM model name for generation
            index_path: Path to store/load FAISS index
            data_path: Path to store/load document data
        """
        self.embedding_model_name = embedding_model
        self.llm_model_name = llm_model
        self.index_path = index_path
        self.data_path = data_path
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        
        # Initialize components based on availability
        self.embedder = None
        self.embedding_dim = 384  # Default dimension
        self.llm = None
        self.index = None
        self.documents = []
        self.metadata = []
        
        # Initialize embedding model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedder = SentenceTransformer(embedding_model)
                self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
            except Exception as e:
                print(f"Warning: Could not initialize embedding model: {e}")
        else:
            print("Warning: Using fallback embedding (random vectors)")
        
        # Initialize LLM
        if TRANSFORMERS_AVAILABLE:
            try:
                self.llm = pipeline(
                    "text-generation",
                    model=llm_model,
                    device=0 if self._has_gpu() else -1
                )
            except Exception as e:
                print(f"Warning: Could not initialize LLM: {e}")
        else:
            print("Warning: LLM generation disabled")
        
        # Try to load existing index
        self._load_index()
    
    def _get_embedding(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for texts, with fallback for missing dependencies."""
        if SENTENCE_TRANSFORMERS_AVAILABLE and self.embedder:
            return self.embedder.encode(texts)
        else:
            # Fallback: random embeddings
            np.random.seed(42)  # For reproducibility
            return np.random.randn(len(texts), self.embedding_dim).astype(np.float32)
    
    @staticmethod
    def _has_gpu() -> bool:
        """Check if GPU is available."""
        if not TRANSFORMERS_AVAILABLE:
            return False
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def index_data(self, db: Session) -> int:
        """
        Index all expense and user data from database.
        
        Args:
            db: SQLAlchemy database session
            
        Returns:
            Number of documents indexed
        """
        self.documents = []
        self.metadata = []
        
        # Index users
        users = db.query(User).all()
        for user in users:
            doc_text = self._format_user_doc(user)
            self.documents.append(Document(page_content=doc_text, metadata={"type": "user", "id": user.userid}))
            self.metadata.append({"type": "user", "id": user.userid, "user": user})
        
        # Index expenses
        expenses = db.query(Expense).all()
        for expense in expenses:
            doc_text = self._format_expense_doc(expense)
            self.documents.append(Document(page_content=doc_text, metadata={"type": "expense", "id": expense.expenseid}))
            self.metadata.append({"type": "expense", "id": expense.expenseid, "expense": expense})
        
        # Create embeddings and build FAISS index
        if self.documents:
            embeddings = self._get_embedding([doc.page_content for doc in self.documents])
            
            # Create FAISS index
            if FAISS_AVAILABLE:
                self.index = faiss.IndexFlatL2(self.embedding_dim)
                self.index.add(embeddings.astype(np.float32))
            else:
                print("Warning: FAISS not available, index creation skipped")
            
            # Save index
            self._save_index()
            
            return len(self.documents)
        
        return 0
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query string
            top_k: Number of top results to return
            
        Returns:
            List of (Document, similarity_score) tuples
        """
        if not FAISS_AVAILABLE or self.index is None or not self.documents:
            # Fallback: return random documents
            if self.documents:
                import random
                selected = random.sample(self.documents, min(top_k, len(self.documents)))
                return [(doc, 0.5) for doc in selected]  # Random similarity score
            return []
        
        # Embed query
        query_embedding = self._get_embedding([query])[0]
        
        # Search in FAISS
        distances, indices = self.index.search(
            np.array([query_embedding]).astype(np.float32),
            min(top_k, len(self.documents))
        )
        
        # Return documents with similarity scores
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if 0 <= idx < len(self.documents):
                # Convert L2 distance to similarity score
                similarity = 1 / (1 + distance)
                results.append((self.documents[idx], similarity))
        
        return results
    
    def generate_response(self, query: str, context: List[Document], max_length: int = 200) -> str:
        """
        Generate response augmented with retrieved context.
        
        Args:
            query: User query
            context: Retrieved context documents
            max_length: Maximum length of generated response
            
        Returns:
            Generated response
        """
        # Format context
        context_text = "\n".join([f"- {doc.page_content}" for doc in context])
        
        # Create prompt
        prompt = f"""Based on the following context about expenses and users, answer the query.

Context:
{context_text}

Query: {query}

Answer:"""
        
        # Generate response
        if TRANSFORMERS_AVAILABLE and self.llm:
            try:
                response = self.llm(prompt, max_length=max_length, num_return_sequences=1)
                return response[0]['generated_text'].split("Answer:")[-1].strip()
            except Exception as e:
                return f"Error generating response: {str(e)}"
        else:
            # Fallback: simple rule-based response
            return self._generate_fallback_response(query, context)
    
    def _generate_fallback_response(self, query: str, context: List[Document]) -> str:
        """Generate a simple fallback response when LLM is not available."""
        query_lower = query.lower()
        
        # Simple keyword-based responses
        if "total" in query_lower or "sum" in query_lower:
            return "Based on the expense data, I can help you calculate totals. Please check the regular chat endpoint for specific calculations."
        
        elif "user" in query_lower:
            if context:
                return f"I found information about users in the database. There are {len([c for c in context if 'User Information' in c.page_content])} user records available."
            return "I can help you with user information. Please check the regular chat endpoint for user details."
        
        elif "expense" in query_lower:
            if context:
                return f"I found {len([c for c in context if 'Expense Record' in c.page_content])} expense records. Use the regular chat endpoint for detailed expense queries."
            return "I can help you with expense information. Please check the regular chat endpoint for expense details."
        
        else:
            return "I'm a RAG system for expense data, but my AI generation capabilities are currently limited. Please use the regular chat endpoint for general questions, or check the database directly for expense data."
    
    def query(self, query: str, db: Session, top_k: int = 5, max_length: int = 200) -> Dict[str, Any]:
        """
        Full RAG pipeline: retrieve and generate response.
        
        Args:
            query: User query
            db: Database session
            top_k: Number of documents to retrieve
            max_length: Maximum length of generated response
            
        Returns:
            Dictionary with query, retrieved documents, and response
        """
        # Retrieve relevant documents
        retrieved_docs = self.retrieve(query, top_k=top_k)
        
        if not retrieved_docs:
            return {
                "query": query,
                "retrieved_documents": [],
                "response": "No relevant documents found in the knowledge base.",
                "sources": []
            }
        
        # Extract documents for context
        docs = [doc for doc, _ in retrieved_docs]
        scores = [score for _, score in retrieved_docs]
        
        # Generate response
        response = self.generate_response(query, docs, max_length=max_length)
        
        # Format sources
        sources = [
            {
                "type": metadata.get("type"),
                "id": metadata.get("id"),
                "score": float(score)
            }
            for metadata, score in zip(
                [doc.metadata for doc, _ in retrieved_docs],
                scores
            )
        ]
        
        return {
            "query": query,
            "retrieved_documents": [doc.page_content for doc in docs],
            "response": response,
            "sources": sources
        }
    
    def _format_user_doc(self, user: User) -> str:
        """Format user data as document string."""
        return f"""User Information:
Employee ID: {user.employeeid}
Name: {user.name}
Date of Birth: {user.dob}
Contact: {user.contact_number}
Email: {user.email}"""
    
    def _format_expense_doc(self, expense: Expense) -> str:
        """Format expense data as document string."""
        return f"""Expense Record:
Expense ID: {expense.expenseid}
User ID: {expense.userid}
Amount: {expense.amount}
Payment Mode: {expense.mode_of_payment}
Type: {expense.credit_or_debit}
Date: {expense.expense_date}
Description: {expense.description}"""
    
    def _save_index(self) -> None:
        """Save FAISS index and documents to disk."""
        try:
            # Save FAISS index
            if FAISS_AVAILABLE and self.index is not None:
                faiss.write_index(self.index, f"{self.index_path}.faiss")
            
            # Save documents and metadata
            with open(self.data_path, 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'metadata': self.metadata
                }, f)
        except Exception as e:
            print(f"Error saving index: {str(e)}")
    
    def _load_index(self) -> None:
        """Load FAISS index and documents from disk."""
        try:
            # Load FAISS index
            if FAISS_AVAILABLE:
                index_file = f"{self.index_path}.faiss"
                if os.path.exists(index_file):
                    self.index = faiss.read_index(index_file)
            
            # Load documents and metadata
            if os.path.exists(self.data_path):
                with open(self.data_path, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data.get('documents', [])
                    self.metadata = data.get('metadata', [])
        except Exception as e:
            print(f"Error loading index: {str(e)}")


# Global RAG service instance
_rag_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    """Get or create RAG service instance."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service


def initialize_rag(db: Session) -> int:
    """Initialize RAG service with database data."""
    rag = get_rag_service()
    return rag.index_data(db)
