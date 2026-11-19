"""FastAPI Full-Stack Sunucusu — Bağımsız Otonom Yapay Zeka API'si."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.ai_engine import AIEngine
from app.memory_store import MemoryStore

app = FastAPI(title="Bağımsız Otonom Yapay Zeka API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

store = MemoryStore(os.getenv("DATABASE_PATH", "data/memory.db"))
engine = AIEngine()
AUTO_LEARN = os.getenv("AUTO_LEARN_ENABLED", "true").lower() == "true"


class KnowledgeCreate(BaseModel):
    title: str = Field(..., min_length=2)
    content: str = Field(..., min_length=2)
    category: str = "Genel"
    keywords: str = ""
    status: str = "approved"


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    system_prompt: str = "Sen kullanıcının kendi verileriyle büyüyen bağımsız yapay zeka asistanısın."
    auto_learn: bool = True


class SynthesizeRequest(BaseModel):
    topic: str = Field(..., min_length=2)
    count: int = Field(default=3, ge=1, le=5)
    auto_approve: bool = False


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": "0.2.0", "provider": engine.provider}


@app.get("/api/knowledge")
def list_knowledge(status: str | None = None) -> list[dict]:
    return store.list_knowledge(status=status)


@app.post("/api/knowledge")
def create_knowledge(payload: KnowledgeCreate) -> dict:
    return store.add_knowledge(
        title=payload.title,
        content=payload.content,
        category=payload.category,
        keywords=payload.keywords,
        status=payload.status,
    )


@app.post("/api/knowledge/{item_id}/approve")
def approve_knowledge(item_id: int) -> dict:
    updated = store.update_status(item_id, "approved")
    if not updated:
        raise HTTPException(status_code=404, detail="Kayıt bulunamadı")
    return updated


@app.delete("/api/knowledge/{item_id}")
def delete_knowledge(item_id: int) -> dict[str, bool]:
    deleted = store.delete_knowledge(item_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Kayıt bulunamadı")
    return {"deleted": True}


@app.post("/api/chat")
async def chat(payload: ChatRequest) -> dict:
    approved_items = store.list_knowledge(status="approved")
    result = await engine.generate_reply(payload.message, payload.system_prompt, approved_items)
    store.log_conversation(payload.message, result["reply"], result["provider"])

    captured_draft = None
    if AUTO_LEARN and payload.auto_learn:
        candidate = engine.extract_candidate_fact(payload.message)
        if candidate:
            captured_draft = store.add_knowledge(
                title=candidate["title"],
                content=candidate["content"],
                category=candidate["category"],
                keywords=candidate["keywords"],
                source=candidate["source"],
                status="draft",
                confidence=candidate["confidence"],
            )

    return {
        "reply": result["reply"],
        "provider": result["provider"],
        "sources": result["sources"],
        "captured_draft": captured_draft,
    }


@app.post("/api/autonomous/synthesize")
def synthesize(payload: SynthesizeRequest) -> dict:
    records = engine.synthesize_records(payload.topic, payload.count)
    created = [
        store.add_knowledge(
            title=r["title"],
            content=r["content"],
            category=r["category"],
            keywords=r["keywords"],
            source=r["source"],
            status="approved" if payload.auto_approve else "draft",
            confidence=r["confidence"],
        )
        for r in records
    ]
    return {"created_count": len(created), "items": created}


@app.get("/api/export/jsonl", response_class=PlainTextResponse)
def export_jsonl() -> str:
    return store.export_jsonl()


frontend_dir = Path(__file__).resolve().parents[2] / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
