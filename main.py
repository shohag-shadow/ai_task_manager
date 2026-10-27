import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv()

import db
import agent

app = FastAPI(title="AI Task Manager")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup() -> None:
    db.init_db()
    try:
        import vector_store
        for t in db.list_tasks():
            vector_store.upsert(t)
    except Exception as e:
        print(f"[startup] vector re-index warning: {e}")


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []


class ChatResponse(BaseModel):
    reply: str
    tool_log: list[dict] = []


@app.get("/api/health")
async def health() -> dict:
    return {"ok": True, "model": agent.MODEL}


@app.get("/api/tasks")
async def get_tasks() -> dict:
    return {"tasks": db.list_tasks()}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(500, "OPENAI_API_KEY is not configured.")
    if not req.message.strip():
        raise HTTPException(400, "Empty message.")
    try:
        result = agent.run_chat(
            message=req.message,
            history=[m.model_dump() for m in req.history],
        )
        return ChatResponse(**result)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"Agent error: {e}") from e


STATIC_DIR = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
