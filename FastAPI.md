# Smart HR Policy RAG - FastAPI

Added a FastAPI wrapper around the existing LangGraph RAG workflow.

Endpoints:
- `POST /chat` : stream responses from the graph as SSE. Body: `{ "query": "..." }` or `{ "messages": ["..."] }`, optional `session_id`.
- `POST /ingest` : trigger ingestion (runs in background).
- `GET /graph_image` : returns the workflow PNG (mermaid renderer).

To Run locally:

```bash
pip install -r requirements.txt
uvicorn api:app --reload --port 8000
```

Quick curl examples:

```bash
# POST JSON body
curl -N -H "Content-Type: application/json" -X POST http://localhost:8000/chat -d '{"query":"Tell me about the leave policy"}'

# GET (quick test)
curl -N "http://localhost:8000/chat?query=Tell%20me%20about%20the%20leave%20policy"
```
