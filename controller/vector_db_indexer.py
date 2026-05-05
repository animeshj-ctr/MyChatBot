"""
Vector database indexer for RAG system.
Handles indexing and updating of expense and user data.
"""

import logging
from typing import Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session

from models.models import Expense, User
from .rag_service import get_rag_service

logger = logging.getLogger(__name__)


class VectorDatabaseIndexer:
    """Manages indexing of data for the RAG system."""
    
    def __init__(self):
        self.rag_service = get_rag_service()
        self.last_indexed_time = None
        self.index_stats = {
            "users_indexed": 0,
            "expenses_indexed": 0,
            "total_documents": 0,
            "last_index_time": None
        }
    
    def build_index(self, db: Session) -> Dict[str, Any]:
        """
        Build complete vector index from database.
        
        Args:
            db: Database session
            
        Returns:
            Dictionary with indexing statistics
        """
        try:
            logger.info("Starting vector database indexing...")
            
            # Get all users and expenses
            users = db.query(User).all()
            expenses = db.query(Expense).all()
            
            logger.info(f"Found {len(users)} users and {len(expenses)} expenses to index")
            
            # Index data
            num_docs = self.rag_service.index_data(db)
            
            self.index_stats = {
                "users_indexed": len(users),
                "expenses_indexed": len(expenses),
                "total_documents": num_docs,
                "last_index_time": datetime.now().isoformat()
            }
            
            self.last_indexed_time = datetime.now()
            logger.info(f"Indexing completed successfully. {num_docs} documents indexed.")
            
            return {
                "status": "success",
                "message": f"Successfully indexed {num_docs} documents",
                "stats": self.index_stats
            }
        
        except Exception as e:
            logger.error(f"Error during indexing: {str(e)}")
            return {
                "status": "error",
                "message": f"Error during indexing: {str(e)}",
                "stats": self.index_stats
            }
    
    def rebuild_index(self, db: Session) -> Dict[str, Any]:
        """
        Rebuild the entire vector index from scratch.
        
        Args:
            db: Database session
            
        Returns:
            Rebuilding result
        """
        logger.info("Rebuilding vector index...")
        return self.build_index(db)
    
    def update_index(self, db: Session, since: datetime = None) -> Dict[str, Any]:
        """
        Update index with new or modified documents.
        For now, we rebuild the entire index. In production, implement incremental updates.
        
        Args:
            db: Database session
            since: Only update documents modified since this time
            
        Returns:
            Update result
        """
        logger.info("Updating vector index...")
        # For now, rebuild the entire index
        # In production, implement incremental updates
        return self.build_index(db)
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get current index statistics."""
        return {
            "index_stats": self.index_stats,
            "last_indexed": self.last_indexed_time.isoformat() if self.last_indexed_time else None,
            "index_size": len(self.rag_service.documents),
            "embedding_dimension": self.rag_service.embedding_dim
        }
    
    def validate_index(self, db: Session) -> Dict[str, Any]:
        """
        Validate index consistency with database.
        
        Args:
            db: Database session
            
        Returns:
            Validation results
        """
        try:
            users_count = db.query(User).count()
            expenses_count = db.query(Expense).count()
            
            indexed_count = len(self.rag_service.documents)
            db_count = users_count + expenses_count
            
            is_valid = indexed_count == db_count
            
            return {
                "is_valid": is_valid,
                "database_records": db_count,
                "indexed_documents": indexed_count,
                "users_in_db": users_count,
                "expenses_in_db": expenses_count,
                "message": "Index is in sync with database" if is_valid else "Index is out of sync"
            }
        
        except Exception as e:
            return {
                "is_valid": False,
                "error": str(e),
                "message": f"Error validating index: {str(e)}"
            }
    
    def clear_index(self) -> Dict[str, Any]:
        """Clear the vector index."""
        self.rag_service.documents = []
        self.rag_service.metadata = []
        self.rag_service.index = None
        
        logger.info("Vector index cleared")
        
        return {
            "status": "success",
            "message": "Index cleared successfully"
        }


# Global indexer instance
_indexer: VectorDatabaseIndexer = None


def get_indexer() -> VectorDatabaseIndexer:
    """Get or create vector database indexer instance."""
    global _indexer
    if _indexer is None:
        _indexer = VectorDatabaseIndexer()
    return _indexer
