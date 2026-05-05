# RAG Quick Start Guide

Get started with Retrieval-Augmented Generation in your MyChatBot project in 5 minutes.

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `langchain` - LLM orchestration
- `sentence-transformers` - Embedding models
- `faiss-cpu` - Vector similarity search
- `transformers` - Hugging Face models

## 2. Start the Application

```bash
uvicorn main:app --reload
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     RAG service initialized. Run POST /rag/index to build the vector database.
```

## 3. Build the Vector Index

Build the index from your expense data:

```bash
# Using curl
curl -X POST http://localhost:8000/rag/index \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"

# Expected response:
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

## 4. Query the RAG System

Ask a question about your expenses:

```bash
curl -X POST http://localhost:8000/rag/query \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the total expenses for user 5?",
    "top_k": 5,
    "max_length": 200
  }'

# Response includes:
# - retrieved_documents: Relevant expense/user records
# - response: AI-generated answer
# - sources: Which documents were used
```

## 5. Monitor Index Health

```bash
# Check statistics
curl http://localhost:8000/rag/stats \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"

# Validate index consistency
curl http://localhost:8000/rag/validate \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

## Usage Examples

### Python
```python
import requests

token = "your_jwt_token"
headers = {"Authorization": f"Bearer {token}"}

# Build index
requests.post("http://localhost:8000/rag/index", headers=headers)

# Query
response = requests.post(
    "http://localhost:8000/rag/query",
    headers=headers,
    json={
        "query": "Show me high-value expenses",
        "top_k": 5
    }
)
print(response.json()["response"])
```

### JavaScript/WebSocket
```javascript
const token = "your_jwt_token";
const ws = new WebSocket(`ws://localhost:8000/rag/ws?token=${token}`);

ws.onopen = () => {
  ws.send(JSON.stringify({
    query: "What payment modes are used most?",
    top_k: 5
  }));
};

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  console.log(message);
};
```

## API Endpoints Reference

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/rag/index` | POST | Build/rebuild vector index |
| `/rag/index` | DELETE | Clear the index |
| `/rag/query` | POST | Query with semantic search |
| `/rag/stats` | GET | Get index statistics |
| `/rag/validate` | GET | Validate index consistency |
| `/rag/ws` | WebSocket | Stream responses |

## Troubleshooting

### ❌ "RAG index is empty"
**Solution**: Call `POST /rag/index` first

### ❌ "401 Unauthorized"
**Solution**: Get a valid JWT token first:
```bash
curl -X POST http://localhost:8000/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=user1&password=password"
```

### ❌ Slow responses
**Solution**: 
- Reduce `top_k` in queries
- Use smaller embedding model
- Check index size with `/rag/stats`

### ❌ GPU out of memory
**Solution**: Use CPU-based FAISS:
```python
# Already using faiss-cpu (no GPU needed)
```

## Configuration

### Change Embedding Model
Edit `controller/rag_service.py`:
```python
RAGService(embedding_model="all-mpnet-base-v2")  # 768 dims, slower but better quality
```

### Change LLM Model
Edit `controller/rag_service.py`:
```python
RAGService(llm_model="facebook/opt-350m")  # Larger model, better quality
```

## Next Steps

1. **Integrate with UI**: Add RAG queries to your frontend
2. **Custom Prompts**: Modify response generation in `rag_service.py`
3. **Fine-tuning**: Train embedding model on your domain
4. **Monitoring**: Add logging and metrics to track RAG performance
5. **Production**: Deploy with GPU and distributed vector search

See [RAG_GUIDE.md](RAG_GUIDE.md) for detailed documentation.
