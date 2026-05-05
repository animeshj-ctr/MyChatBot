"""
Integration tests for RAG system.
Run with: pytest tests/test_rag.py -v
"""

import pytest
import json
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import date, datetime

from main import app, get_db
from models.models import Base, User, Expense
from controller.rag_service import get_rag_service, RAGService
from controller.vector_db_indexer import get_indexer

# Test database setup
SQLALCHEMY_TEST_DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(
    SQLALCHEMY_TEST_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)


def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)


@pytest.fixture(scope="function")
def test_db():
    """Create test database and populate with sample data."""
    Base.metadata.create_all(bind=engine)
    
    db = TestingSessionLocal()
    
    # Add test users
    users = [
        User(
            employeeid=101,
            name="John Doe",
            dob=date(1990, 1, 15),
            contact_number="9876543210",
            email="john@example.com"
        ),
        User(
            employeeid=102,
            name="Jane Smith",
            dob=date(1992, 5, 20),
            contact_number="9876543211",
            email="jane@example.com"
        ),
    ]
    
    for user in users:
        db.add(user)
    db.commit()
    
    # Add test expenses
    expenses = [
        Expense(
            userid=1,
            mode_of_payment="credit_card",
            amount=5000.00,
            credit_or_debit="debit",
            expense_date=date(2024, 1, 10),
            description="Office supplies"
        ),
        Expense(
            userid=1,
            mode_of_payment="cash",
            amount=2000.00,
            credit_or_debit="debit",
            expense_date=date(2024, 1, 15),
            description="Travel"
        ),
        Expense(
            userid=2,
            mode_of_payment="cheque",
            amount=10000.00,
            credit_or_debit="credit",
            expense_date=date(2024, 1, 12),
            description="Project reimbursement"
        ),
    ]
    
    for expense in expenses:
        db.add(expense)
    db.commit()
    
    yield db
    
    db.close()
    Base.metadata.drop_all(bind=engine)


class TestRAGService:
    """Test cases for RAG service."""
    
    def test_rag_service_initialization(self):
        """Test RAG service initialization."""
        rag = RAGService()
        assert rag.embedding_dim == 384  # all-MiniLM-L6-v2
        assert rag.documents == []
        assert rag.index is None
    
    def test_document_formatting(self, test_db):
        """Test document formatting for users and expenses."""
        rag = RAGService()
        
        user = test_db.query(User).first()
        user_doc = rag._format_user_doc(user)
        
        assert "User Information:" in user_doc
        assert "John Doe" in user_doc
        assert "9876543210" in user_doc
        
        expense = test_db.query(Expense).first()
        expense_doc = rag._format_expense_doc(expense)
        
        assert "Expense Record:" in expense_doc
        assert "Office supplies" in expense_doc
        assert "5000" in expense_doc
    
    def test_indexing_data(self, test_db):
        """Test indexing data from database."""
        rag = RAGService(index_path="local/test_index", data_path="local/test_docs.pkl")
        
        num_docs = rag.index_data(test_db)
        
        assert num_docs == 5  # 2 users + 3 expenses
        assert len(rag.documents) == 5
        assert rag.index is not None
        assert rag.index.ntotal == 5
    
    def test_retrieval(self, test_db):
        """Test document retrieval."""
        rag = RAGService(index_path="local/test_index", data_path="local/test_docs.pkl")
        rag.index_data(test_db)
        
        # Query for expenses
        results = rag.retrieve("What are the expenses for user 1?", top_k=3)
        
        assert len(results) > 0
        assert len(results) <= 3
        
        # Check that results are tuples of (Document, score)
        for doc, score in results:
            assert hasattr(doc, 'page_content')
            assert isinstance(score, float)
            assert 0 <= score <= 1


class TestVectorDatabaseIndexer:
    """Test cases for vector database indexer."""
    
    def test_indexer_initialization(self):
        """Test indexer initialization."""
        indexer = get_indexer()
        assert indexer.index_stats["total_documents"] == 0
    
    def test_build_index(self, test_db):
        """Test building index."""
        indexer = get_indexer()
        result = indexer.build_index(test_db)
        
        assert result["status"] == "success"
        assert result["stats"]["users_indexed"] == 2
        assert result["stats"]["expenses_indexed"] == 3
        assert result["stats"]["total_documents"] == 5
    
    def test_validate_index(self, test_db):
        """Test index validation."""
        indexer = get_indexer()
        indexer.build_index(test_db)
        
        validation = indexer.validate_index(test_db)
        
        assert validation["is_valid"] is True
        assert validation["database_records"] == 5
        assert validation["indexed_documents"] == 5


class TestRAGEndpoints:
    """Test cases for RAG API endpoints."""
    
    def test_build_index_endpoint(self, test_db):
        """Test POST /rag/index endpoint."""
        response = client.post("/rag/index")
        
        # May fail due to auth, but should not be a server error
        assert response.status_code in [200, 401, 403, 422]
    
    def test_get_stats_endpoint(self, test_db):
        """Test GET /rag/stats endpoint."""
        response = client.get("/rag/stats")
        
        # May fail due to auth
        assert response.status_code in [200, 401, 403, 422]
    
    def test_validate_endpoint(self, test_db):
        """Test GET /rag/validate endpoint."""
        response = client.get("/rag/validate")
        
        # May fail due to auth
        assert response.status_code in [200, 401, 403, 422]
    
    def test_query_endpoint_no_index(self, test_db):
        """Test POST /rag/query without index."""
        payload = {
            "query": "What are the total expenses?",
            "top_k": 5,
            "max_length": 200
        }
        response = client.post("/rag/query", json=payload)
        
        # May fail due to auth or missing index
        assert response.status_code in [400, 401, 403, 422]


class TestRAGIntegration:
    """End-to-end RAG integration tests."""
    
    def test_full_rag_pipeline(self, test_db):
        """Test complete RAG pipeline: index -> retrieve -> generate."""
        rag = RAGService(index_path="local/test_index", data_path="local/test_docs.pkl")
        
        # Step 1: Index data
        num_docs = rag.index_data(test_db)
        assert num_docs == 5
        
        # Step 2: Query
        result = rag.query(
            "What expenses did user 1 have?",
            test_db,
            top_k=3,
            max_length=100
        )
        
        assert "query" in result
        assert "retrieved_documents" in result
        assert "response" in result
        assert "sources" in result
        
        assert len(result["retrieved_documents"]) > 0
        assert result["response"] != ""
    
    def test_multiple_queries(self, test_db):
        """Test multiple queries on same index."""
        rag = RAGService(index_path="local/test_index", data_path="local/test_docs.pkl")
        rag.index_data(test_db)
        
        queries = [
            "What are the expenses for user 1?",
            "Show me credit transactions",
            "List all payment modes"
        ]
        
        for query in queries:
            result = rag.query(query, test_db, top_k=3)
            assert len(result["retrieved_documents"]) > 0
            assert result["response"] != ""


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
