"""
RAG System Usage Examples

This file demonstrates practical usage of the RAG system in your MyChatBot application.
"""

# ============ EXAMPLE 1: Building the Index ============

import requests
from datetime import date

# Configuration
BASE_URL = "http://localhost:8000"
TOKEN = "your_jwt_token_here"

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

def build_rag_index():
    """Build the vector index from database."""
    response = requests.post(f"{BASE_URL}/rag/index", headers=headers)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Index built successfully!")
        print(f"   - Users indexed: {result['stats']['users_indexed']}")
        print(f"   - Expenses indexed: {result['stats']['expenses_indexed']}")
        print(f"   - Total documents: {result['stats']['total_documents']}")
    else:
        print(f"❌ Error building index: {response.status_code}")
        print(response.text)


# ============ EXAMPLE 2: Query the RAG System ============

def query_rag(question: str, top_k: int = 5):
    """Query the RAG system."""
    payload = {
        "query": question,
        "top_k": top_k,
        "max_length": 300
    }
    
    response = requests.post(f"{BASE_URL}/rag/query", json=payload, headers=headers)
    
    if response.status_code == 200:
        result = response.json()
        
        print(f"\n📝 Query: {result['query']}")
        print(f"\n📚 Retrieved Documents ({len(result['retrieved_documents'])} found):")
        for i, doc in enumerate(result['retrieved_documents'][:3], 1):
            print(f"   {i}. {doc[:100]}...")
        
        print(f"\n💡 Response:")
        print(f"   {result['response']}")
        
        print(f"\n🔗 Sources:")
        for source in result['sources']:
            print(f"   - {source['type']} (ID: {source['id']}, Score: {source['score']:.2f})")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)


# ============ EXAMPLE 3: Monitor Index Health ============

def check_index_stats():
    """Check index statistics."""
    response = requests.get(f"{BASE_URL}/rag/stats", headers=headers)
    
    if response.status_code == 200:
        stats = response.json()
        print("📊 Index Statistics:")
        print(f"   - Total documents: {stats['index_size']}")
        print(f"   - Embedding dimension: {stats['embedding_dimension']}")
        print(f"   - Last indexed: {stats['last_indexed']}")
    else:
        print(f"❌ Error: {response.status_code}")


def validate_index():
    """Validate index consistency."""
    response = requests.get(f"{BASE_URL}/rag/validate", headers=headers)
    
    if response.status_code == 200:
        validation = response.json()
        status_icon = "✅" if validation['is_valid'] else "⚠️"
        print(f"\n{status_icon} Index Validation:")
        print(f"   - Valid: {validation['is_valid']}")
        print(f"   - Database records: {validation['database_records']}")
        print(f"   - Indexed documents: {validation['indexed_documents']}")
        print(f"   - Message: {validation['message']}")
    else:
        print(f"❌ Error: {response.status_code}")


# ============ EXAMPLE 4: WebSocket Streaming ============

import asyncio
import websockets
import json

async def stream_rag_query(question: str):
    """Stream RAG response using WebSocket."""
    
    # Note: Replace with actual auth token handling
    uri = f"ws://localhost:8000/rag/ws"
    
    try:
        async with websockets.connect(uri) as websocket:
            # Send query
            await websocket.send(json.dumps({
                "query": question,
                "top_k": 5,
                "max_length": 300
            }))
            
            print(f"📤 Sent query: {question}\n")
            
            # Receive streaming response
            while True:
                try:
                    message = json.loads(await websocket.recv())
                    
                    if message.get("type") == "retrieved":
                        print("📚 Retrieved documents:")
                        for i, doc in enumerate(message['documents'][:2], 1):
                            print(f"   {i}. {doc[:80]}...")
                    
                    elif message.get("type") == "response_partial":
                        print(f"\r💡 {message['text']}", end="", flush=True)
                    
                    elif message.get("type") == "response_complete":
                        print(f"\n\n✅ Final response:\n{message['text']}")
                        break
                    
                    elif message.get("error"):
                        print(f"❌ Error: {message['error']}")
                        break
                
                except json.JSONDecodeError:
                    pass
    
    except Exception as e:
        print(f"❌ WebSocket error: {e}")


# ============ EXAMPLE 5: Use Cases ============

def main():
    """Run example use cases."""
    
    print("=" * 60)
    print("RAG System Examples")
    print("=" * 60)
    
    # Use Case 1: Build index
    print("\n1️⃣  Building RAG Index...")
    build_rag_index()
    
    # Use Case 2: Query examples
    print("\n\n2️⃣  Query Examples...")
    
    queries = [
        "What are the total expenses for user 5?",
        "Show me high-value expenses over 10000",
        "List payment modes used in January",
        "What is the email of employee 103?",
        "Show me credit transactions"
    ]
    
    for query in queries:
        query_rag(query, top_k=3)
        print("\n" + "─" * 60)
    
    # Use Case 3: Monitor health
    print("\n\n3️⃣  Index Health Check...")
    check_index_stats()
    print()
    validate_index()
    
    # Use Case 4: Stream example (async)
    print("\n\n4️⃣  WebSocket Streaming (if needed)...")
    print("   Run: asyncio.run(stream_rag_query('What are recent expenses?'))")


# ============ EXAMPLE 6: Integration with Chat ============

def augmented_chat(user_message: str, session_id: str = "default"):
    """
    Example of augmenting the chat endpoint with RAG context.
    This shows how to combine regular chat with RAG retrieval.
    """
    
    # First, try RAG retrieval
    print(f"🔍 Retrieving context for: {user_message}")
    
    payload = {
        "query": user_message,
        "top_k": 3,
        "max_length": 200
    }
    
    rag_response = requests.post(f"{BASE_URL}/rag/query", json=payload, headers=headers)
    
    if rag_response.status_code == 200:
        rag_result = rag_response.json()
        
        # Augment chat with RAG context
        chat_payload = {
            "session_id": session_id,
            "message": f"{user_message}\n\nContext: {' '.join(rag_result['retrieved_documents'][:2])}"
        }
        
        chat_response = requests.post(f"{BASE_URL}/chat", json=chat_payload, headers=headers)
        
        if chat_response.status_code == 200:
            result = chat_response.json()
            print(f"🤖 Bot: {result['bot']}")
            print(f"📚 Used sources: {len(rag_result['sources'])} documents")
        else:
            print(f"❌ Chat error: {chat_response.status_code}")
    else:
        print(f"❌ RAG error: {rag_response.status_code}")


# ============ EXAMPLE 7: Batch Queries ============

def batch_rag_queries(questions: list) -> list:
    """Process multiple queries efficiently."""
    
    results = []
    
    for i, question in enumerate(questions, 1):
        print(f"Processing {i}/{len(questions)}: {question[:50]}...")
        
        payload = {
            "query": question,
            "top_k": 3,
            "max_length": 150
        }
        
        response = requests.post(f"{BASE_URL}/rag/query", json=payload, headers=headers)
        
        if response.status_code == 200:
            results.append({
                "query": question,
                "response": response.json()['response'],
                "sources_count": len(response.json()['sources'])
            })
        else:
            print(f"  ⚠️  Query failed: {response.status_code}")
    
    return results


# ============ EXAMPLE 8: Error Handling ============

def safe_rag_query(question: str) -> dict:
    """Query RAG with error handling."""
    
    try:
        # Check if index exists
        stats_response = requests.get(f"{BASE_URL}/rag/stats", headers=headers)
        if stats_response.status_code != 200:
            return {"error": "Index not available", "status": "build_index_first"}
        
        stats = stats_response.json()
        if stats['index_size'] == 0:
            return {"error": "Index is empty", "status": "build_index_first"}
        
        # Query RAG
        payload = {"query": question, "top_k": 5, "max_length": 200}
        response = requests.post(f"{BASE_URL}/rag/query", json=payload, headers=headers)
        
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {"error": f"Query failed: {response.status_code}"}
    
    except Exception as e:
        return {"error": f"Exception: {str(e)}"}


if __name__ == "__main__":
    # Get JWT token first
    print("Note: Update TOKEN variable with your JWT token from /token endpoint")
    
    # Uncomment to run examples:
    # main()
    
    # Or run individual examples:
    # build_rag_index()
    # query_rag("What are total expenses?")
    # check_index_stats()
    # validate_index()
