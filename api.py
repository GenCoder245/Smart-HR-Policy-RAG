from fastapi import FastAPI, BackgroundTasks, Request, HTTPException, Body, Query
from fastapi.responses import StreamingResponse, Response, JSONResponse
import json
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import structlog

from config import get_settings
from custom_logger import configure_logging
from src.retrieval.retriever import PolicyRetriever
from src.ingestion.ingest_docs import ingest_data_directory
from src.models.llm_models import get_language_model
from src.graph.graph_workflow import build_graph
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

app = FastAPI(title="Smart HR Policy RAG API")
logger = structlog.get_logger()

# Allow CORS for testing from local UIs (Need to adjust origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    global settings, policy_retriever, language_model, graph, sqlite_conn, checkpointer

    settings = get_settings()
    configure_logging(settings.log_level)

    # SQLite connection for checkpointer (shared, thread-safe in this usage)
    sqlite_conn = sqlite3.connect("checkpoints_fastapi.sqlite", check_same_thread=False)
    checkpointer = SqliteSaver(conn=sqlite_conn)

    # Initialize retriever and LLM
    policy_retriever = PolicyRetriever(settings=settings)
    policy_retriever.initialize()

    language_model = get_language_model(llm_settings=settings)

    # Build the langgraph graph
    graph = build_graph(language_model, policy_retriever, checkpointer)


@app.on_event("shutdown")
def shutdown_event():
    try:
        policy_retriever.close()
        logger.info("Workflow completed successfully. Exiting the program...")
    except Exception:
        pass


def _ensure_messages(payload: dict) -> list[HumanMessage]:
    # Accept either a single `query` string or a list `messages` of strings
    if "messages" in payload and isinstance(payload["messages"], list):
        return [HumanMessage(content=m) if isinstance(m, str) else HumanMessage(content=m.get("content", "")) for m in payload["messages"]]
    if "query" in payload and isinstance(payload["query"], str):
        return [HumanMessage(content=payload["query"])]
    raise HTTPException(status_code=400, detail="Provide `query` or `messages` in request body")


@app.post("/ingest")
def ingest_endpoint(background: BackgroundTasks):
    """Trigger data ingestion into the vectordb. Runs in background."""

    def _ingest():
        try:
            logger.info("Starting ingestion via API endpoint.")
            no_files, no_chunks, chunk_ids = ingest_data_directory(settings=settings, retriever=policy_retriever)
            logger.info(f"Inserted {no_chunks} chunks from {no_files} files via API ingestion.")
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")

    # Check how to notify user if background task failed
    background.add_task(_ingest)
    return JSONResponse({"status": "ingestion_started"})


def _make_event_generator(messages, session_id: str):
    def event_generator():
        memory_config = {"configurable": {"thread_id": session_id}}
        try:
            stream = graph.stream(input={"messages": messages, "retrieved_context": "", "next_node": ""}, config=memory_config, stream_mode="values")
            # The graph.invoke streaming object is iterable
            for chunk in stream:
                print(chunk)
                # build a compact JSON payload per event with a predictable schema
                payload = {"type": "chunk", "delta": None, "done": False, "meta": {}}
                try:
                    if isinstance(chunk, str):
                        print("if block")
                        payload["delta"] = chunk
                    elif isinstance(chunk, dict):
                        print("1st elif block")
                        # common fields in delta-like structures
                        if "text" in chunk:
                            payload["delta"] = chunk.get("text")
                        elif "content" in chunk:
                            payload["delta"] = chunk.get("content")
                        else:
                            # keep the whole dict under meta for client inspection
                            payload["meta"] = chunk
                            payload["delta"] = None
                    else:
                        print("else block")
                        payload["delta"] = str(chunk)
                except Exception:
                    payload["delta"] = str(chunk)

                yield f"data: {json.dumps(payload, default=str)}\n\n"

            # final done event so clients can know the stream finished
            yield f"data: {json.dumps({'type': 'done', 'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return event_generator



@app.post("/chat")
async def chat_stream(payload: dict = Body(default={}), session_id: str | None = Body(default=None)):
    """Stream responses from the LangGraph workflow as Server-Sent Events (SSE).

    Accepts JSON body: `{"query": "..."}` or `{"messages": ["..."]}` and optional `session_id`.
    """
    try:
        messages = _ensure_messages(payload)
    except HTTPException as e:
        raise e

    sid = session_id or payload.get("session_id") or payload.get("session") or "default"
    gen = _make_event_generator(messages, sid)
    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/chat")
def chat_get(query: str = Query(None), session_id: str | None = Query(None)):
    """Quick GET test: /chat?query=... (useful for quick checks)."""
    if not query:
        raise HTTPException(status_code=400, detail="Provide `query` parameter for GET /chat")
    messages = [HumanMessage(content=query)]
    sid = session_id or "default"
    gen = _make_event_generator(messages, sid)
    return StreamingResponse(gen(), media_type="text/event-stream")



@app.get("/graph_image")
def graph_image():
    """Return the graph workflow PNG (mermaid renderer) as image/png."""
    try:
        png_data = graph.get_graph().draw_mermaid_png()
        return Response(content=png_data, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
