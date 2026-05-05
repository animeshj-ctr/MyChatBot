# Retrieval-Augmented Generation (RAG) Implementation Guide

## Overview

This document describes the RAG implementation for the MyChatBot expense tracking application. The RAG system retrieves relevant expense and user data from the database and uses Hugging Face language models to generate context-aware responses.

## Architecture

### Components

1. **RAG Service** (`controller/rag_service.py`)
   - Handles embedding generation using Sentence Transformers
   - Manages FAISS vector indexing
   - Retrieves relevant documents
   - Generates augmented responses using Hugging Face LLM

2. **Vector Database Indexer** (`controller/vector_db_indexer.py`)
   - Indexes all users and expenses from the database
   - Manages index lifecycle (build, update, validate, clear)
   - Tracks indexing statistics

3. **FastAPI Endpoints**
   - REST endpoints for RAG operations
   - WebSocket support for streaming responses

## Setup

### 1. Install Dependencies

All dependencies are in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Key packages:
- `langchain`: LLM orchestration
- `sentence-transformers`: Embedding generation (all-MiniLM-L6-v2)
- `faiss-cpu`: Vector similarity search
- `transformers`: Hugging Face model support

### 2. Environment Configuration

Ensure your `.env` file has:

```
DATABASE_URL=postgresql://username:password@localhost/expenses
HUGGINGFACEHUB_API_TOKEN=your_hf_token  # Optional for API-based models
```

### 3. Initialize the Index

Before querying, build the vector index:

```bash
curl -X POST http://localhost:8000/rag/index \
  -H "Authorization: Bearer YOUR_TOKEN"
```

## API Endpoints

### 1. Build/Rebuild Index

**POST** `/rag/index`

Builds the vector index from all users and expenses in the database.

**Response:**
```json
{
  "status": "success",
  "message": "Successfully indexed 150 documents",
  "stats": {
    "users_indexed": 50,
    "expenses_indexed": 100,
    "total_documents": 150,
    "last_index_time": "2024-01-15T10:30:00"
  }
}
```

### 2. Query RAG

**POST** `/rag/query`

Query the RAG system with a question.

**Request:**
```json
{
  "query": "What are the total expenses for user 5 this month?",
  "top_k": 5,
  "max_length": 200
}
```

**Response:**
```json
{
  "query": "What are the total expenses for user 5 this month?",
  "retrieved_documents": [
    "Expense Record: Expense ID: 42...",
    "User Information: Employee ID: 105..."
  ],
  "response": "Based on the retrieved data, user 5 has total expenses...",
  "sources": [
    {
      "type": "expense",
      "id": 42,
      "score": 0.95
    }
  ]
}
```

### 3. Get Index Statistics

**GET** `/rag/stats`

Get current index statistics and metadata.

**Response:**
```json
{
  "index_stats": {
    "users_indexed": 50,
    "expenses_indexed": 100,
    "total_documents": 150,
    "last_index_time": "2024-01-15T10:30:00"
  },
  "last_indexed": "2024-01-15T10:30:00",
  "index_size": 150,
  "embedding_dimension": 384
}
```

### 4. Validate Index

**GET** `/rag/validate`

Verify that the index is in sync with the database.

**Response:**
```json
{
  "is_valid": true,
  "database_records": 150,
  "indexed_documents": 150,
  "users_in_db": 50,
  "expenses_in_db": 100,
  "message": "Index is in sync with database"
}
```

### 5. Clear Index

**DELETE** `/rag/index`

Clear the vector index from memory and disk.

**Response:**
```json
{
  "status": "success",
  "message": "Index cleared successfully"
}
```

## WebSocket Streaming

### Endpoint

**WebSocket** `/rag/ws`

Connect to this endpoint for streaming RAG responses.

### Message Format

**Send (JSON):**
```json
{
  "query": "Show me high-value expenses",
  "top_k": 5,
  "max_length": 300
}
```

**Receive (JSON stream):**

1. Retrieved documents:
```json
{
  "type": "retrieved",
  "documents": ["Expense Record: ...", "User Information: ..."],
  "sources": [{"type": "expense", "id": 42, "score": 0.95}]
}
```

2. Streaming response:
```json
{
  "type": "response_partial",
  "text": "Based on the retrieved documents"
}
```

3. Complete response:
```json
{
  "type": "response_complete",
  "text": "Based on the retrieved documents, here are the high-value expenses..."
}
```

## Example Usage

### Python Client

```python
import requests
import json

BASE_URL = "http://localhost:8000"
TOKEN = "your_jwt_token"

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

# 1. Build index
response = requests.post(f"{BASE_URL}/rag/index", headers=headers)
print(response.json())

# 2. Query RAG
query_data = {
    "query": "What are the payment modes for user 3?",
    "top_k": 5,
    "max_length": 200
}
response = requests.post(f"{BASE_URL}/rag/query", json=query_data, headers=headers)
print(response.json())

# 3. Get stats
response = requests.get(f"{BASE_URL}/rag/stats", headers=headers)
print(response.json())
```

### JavaScript/WebSocket Client

```javascript
const token = "your_jwt_token";
const ws = new WebSocket(`ws://localhost:8000/rag/ws?token=${token}`);

ws.onopen = () => {
  ws.send(JSON.stringify({
    query: "Show me expenses from last month",
    top_k: 5,
    max_length: 300
  }));
};

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  
  if (message.type === "retrieved") {
    console.log("Retrieved documents:", message.documents);
  } else if (message.type === "response_partial") {
    console.log("Partial:", message.text);
  } else if (message.type === "response_complete") {
    console.log("Complete:", message.text);
  }
};
```

## Configuration

### Embedding Model

The default embedding model is `all-MiniLM-L6-v2` (384 dimensions). To change:

```python
# In rag_service.py
rag = RAGService(
    embedding_model="sentence-transformers/all-mpnet-base-v2",  # 768 dimensions
    llm_model="gpt2"
)
```

### LLM Model

Default is GPT-2. For better results, use:

```python
# Alternative models
"gpt2"                    # ~125M parameters
"facebook/opt-350m"       # ~350M parameters
"meta-llama/Llama-2-7b"   # 7B parameters (requires more VRAM)
```

## Performance Optimization

### Index Size

- Default embedding dimension: 384 (all-MiniLM-L6-v2)
- FAISS uses L2 distance for similarity
- For 10K+ documents, consider:
  - Using `IndexIVFFlat` for faster search
  - GPU acceleration with `faiss-gpu`

### Retrieval Speed

```python
# Retrieve top 3 most relevant documents
retrieved = rag.retrieve(query, top_k=3)  # Faster
```

### Memory Usage

- Vector index stored on disk at `local/faiss_index.faiss`
- Documents stored at `local/documents.pkl`
- Total size ≈ 150KB per 100 documents

## Troubleshooting

### Issue: "RAG index is empty"

**Solution:** Call `POST /rag/index` first to build the index.

### Issue: Slow responses

**Solution:**
- Reduce `top_k` in query
- Use smaller embedding model
- Enable GPU: `device=0` in RAGService

### Issue: GPU out of memory

**Solution:**
- Use `faiss-cpu` instead of `faiss-gpu`
- Switch to smaller model: `all-MiniLM-L6-v2` → `all-MiniLM-L6-v1`
- Reduce batch size in embedding generation

### Issue: Index out of sync

**Solution:**
```bash
curl -X DELETE http://localhost:8000/rag/index -H "Authorization: Bearer TOKEN"
curl -X POST http://localhost:8000/rag/index -H "Authorization: Bearer TOKEN"
```

## Best Practices

1. **Rebuild Index**: After bulk data imports, rebuild the index:
   ```python
   indexer.rebuild_index(db)
   ```

2. **Monitor Sync**: Regularly validate index synchronization:
   ```python
   validation = indexer.validate_index(db)
   if not validation["is_valid"]:
       indexer.rebuild_index(db)
   ```

3. **Query Design**: Be specific in queries for better retrieval:
   - ❌ "Show me expenses"
   - ✅ "Show me expenses for user 5 in January"

4. **Top-K Selection**: Adjust based on use case:
   - Quick answers: `top_k=3`
   - Comprehensive answers: `top_k=10`

## Future Enhancements

- [ ] Implement incremental index updates
- [ ] Add document re-ranking with cross-encoders
- [ ] Support for batch RAG queries
- [ ] Hybrid search (keyword + semantic)
- [ ] Custom chunking strategies for large documents
- [ ] Multi-language support
- [ ] GPU-accelerated retrieval with FAISS GPU

## Testing

Run integration tests:

```bash
# Start the server
python -m uvicorn main:app --reload

# In another terminal, run tests
python tests/test_rag.py
```

## References

- [LangChain Documentation](https://python.langchain.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Hugging Face Models](https://huggingface.co/models)
