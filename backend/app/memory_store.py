"""SQLite tabanlı bilgi tabanı, aday veri havuzu ve JSONL eğitim verisi deposu."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class MemoryStore:
    """Bilgi tabanı kayıtlarını, otonom üretilen taslakları ve sürüm geçmişini yönetir."""

    def __init__(self, db_path: str = "data/memory.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    category TEXT DEFAULT 'Genel',
                    keywords TEXT DEFAULT '',
                    content TEXT NOT NULL,
                    source TEXT DEFAULT 'manual',
                    status TEXT DEFAULT 'approved',
                    confidence REAL DEFAULT 1.0,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_message TEXT NOT NULL,
                    ai_reply TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def add_knowledge(
        self,
        title: str,
        content: str,
        category: str = "Genel",
        keywords: str = "",
        source: str = "manual",
        status: str = "approved",
        confidence: float = 1.0,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO knowledge (title, category, keywords, content, source, status, confidence, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (title.strip(), category.strip(), keywords.strip(), content.strip(), source, status, confidence, now),
            )
            row_id = cur.lastrowid
        return self.get_knowledge_by_id(row_id)  # type: ignore[return-value]

    def get_knowledge_by_id(self, item_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM knowledge WHERE id = ?", (item_id,)).fetchone()
        return dict(row) if row else None

    def list_knowledge(self, status: str | None = None) -> list[dict[str, Any]]:
        with self._connect() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM knowledge WHERE status = ? ORDER BY id DESC", (status,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM knowledge ORDER BY id DESC").fetchall()
        return [dict(r) for r in rows]

    def update_status(self, item_id: int, status: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            conn.execute("UPDATE knowledge SET status = ? WHERE id = ?", (status, item_id))
        return self.get_knowledge_by_id(item_id)

    def delete_knowledge(self, item_id: int) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM knowledge WHERE id = ?", (item_id,))
        return cur.rowcount > 0

    def log_conversation(self, user_message: str, ai_reply: str, provider: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO conversations (user_message, ai_reply, provider, created_at) VALUES (?, ?, ?, ?)",
                (user_message, ai_reply, provider, now),
            )

    def export_jsonl(self) -> str:
        """Onaylanmış kayıtları ince ayar (fine-tuning) ve RAG uyumlu JSONL formatında dışa aktarır."""
        items = self.list_knowledge(status="approved")
        lines = []
        for item in items:
            record = {
                "messages": [
                    {"role": "system", "content": f"Kategori: {item['category']}"},
                    {"role": "user", "content": f"{item['title']} hakkında bilgi ver."},
                    {"role": "assistant", "content": item["content"]},
                ],
                "metadata": {
                    "id": item["id"],
                    "source": item["source"],
                    "keywords": item["keywords"],
                    "confidence": item["confidence"],
                },
            }
            lines.append(json.dumps(record, ensure_ascii=False))
        return "\n".join(lines)
