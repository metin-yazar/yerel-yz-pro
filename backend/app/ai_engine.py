"""Sağlayıcıdan bağımsız (Ollama / Gemini / Yerel RAG) yapay zeka ve otonom veri üretim motoru."""

from __future__ import annotations

import os
import re
from typing import Any

import httpx


class AIEngine:
    """Bağımsız yerel model (Ollama) veya Gemini API ile çalışan modüler motor."""

    def __init__(self) -> None:
        self.provider = os.getenv("AI_PROVIDER", "ollama").lower()
        self.ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"[^a-zçğıöşü0-9\s]", " ", text.lower()).strip()

    def retrieve_context(self, query: str, knowledge_items: list[dict[str, Any]], top_k: int = 3) -> list[dict[str, Any]]:
        tokens = [t for t in self._normalize(query).split() if len(t) > 2]
        if not tokens:
            return []
        scored: list[tuple[float, dict[str, Any]]] = []
        for item in knowledge_items:
            hay_title = self._normalize(item.get("title", ""))
            hay_keys = self._normalize(item.get("keywords", ""))
            hay_body = self._normalize(item.get("content", ""))
            score = 0.0
            for t in tokens:
                if t in hay_keys:
                    score += 4.0
                if t in hay_title:
                    score += 3.0
                if t in hay_body:
                    score += 1.5
            if score > 0:
                scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    async def generate_reply(
        self,
        user_message: str,
        system_prompt: str,
        knowledge_items: list[dict[str, Any]],
    ) -> dict[str, Any]:
        context_items = self.retrieve_context(user_message, knowledge_items)
        context_block = "\n".join(
            f"- [{i['category']}] {i['title']}: {i['content']}" for i in context_items
        ) or "Eşleşen kayıt bulunamadı."

        full_prompt = (
            f"{system_prompt}\n\n"
            f"İlgili Bilgi Tabanı Kayıtları:\n{context_block}\n\n"
            f"Kullanıcı: {user_message}"
        )

        if self.provider == "gemini" and self.gemini_api_key:
            reply = await self._call_gemini(full_prompt)
            if reply:
                return {"reply": reply, "provider": f"gemini:{self.gemini_model}", "sources": context_items}

        if self.provider == "ollama":
            reply = await self._call_ollama(full_prompt)
            if reply:
                return {"reply": reply, "provider": f"ollama:{self.ollama_model}", "sources": context_items}

        # Hiçbir dış model ayakta değilse yerel RAG yanıtı döndür
        if context_items:
            return {
                "reply": context_items[0]["content"],
                "provider": "local-rag",
                "sources": context_items,
            }
        return {
            "reply": "Bu konuda bilgi tabanımda henüz onaylı kayıt yok. Otonom veri üreticiyle bu başlıkta yeni kayıt oluşturabilirsin.",
            "provider": "local-rag",
            "sources": [],
        }

    def extract_candidate_fact(self, user_message: str) -> dict[str, Any] | None:
        """Kullanıcının öğrettiği veya belirttiği yeni bilgiyi otonom olarak aday kayda dönüştürür."""
        clean = user_message.strip()
        if len(clean) < 18 or clean.endswith("?"):
            return None
        words = [w for w in self._normalize(clean).split() if len(w) > 3]
        keywords = ", ".join(dict.fromkeys(words[:6]))
        title = clean[:48].rstrip(" .,;:") + ("..." if len(clean) > 48 else "")
        return {
            "title": f"Öğrenilen Bilgi: {title}",
            "category": "Otonom Öğrenme",
            "keywords": keywords,
            "content": clean,
            "source": "autonomous-chat",
            "confidence": 0.85,
        }

    def synthesize_records(self, topic: str, count: int = 3) -> list[dict[str, Any]]:
        """Belirtilen konu için yapılandırılmış başlangıç veri taslakları üretir."""
        clean_topic = topic.strip()
        templates = [
            ("Temel Tanım", f"{clean_topic} hakkında temel çerçeve ve amaç tanımı. Bu taslağı kendi kurallarına göre genişletebilirsin."),
            ("Sık Sorulan Soru", f"{clean_topic} sürecinde en sık sorulan soru ve standart çözüm adımı."),
            ("Kalite & Kural", f"{clean_topic} konusunda yapay zekanın uyması gereken temel doğruluk ve üslup kuralı."),
            ("Örnek Senaryo", f"{clean_topic} ile ilgili gerçek bir kullanıcı talebine verilecek örnek yanıt akışı."),
            ("Sınır ve İstisna", f"{clean_topic} kapsamında yapay zekanın kesinlikle yapmaması veya kullanıcıya sorması gereken durumlar."),
        ]
        results = []
        for idx in range(min(max(count, 1), len(templates))):
            label, body = templates[idx]
            results.append(
                {
                    "title": f"{clean_topic} — {label}",
                    "category": "Sentetik Veri",
                    "keywords": f"{clean_topic.lower()}, {label.lower()}",
                    "content": body,
                    "source": "autonomous-synthesizer",
                    "confidence": 0.78,
                }
            )
        return results

    async def _call_ollama(self, prompt: str) -> str | None:
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={"model": self.ollama_model, "prompt": prompt, "stream": False},
                )
                if resp.status_code == 200:
                    return resp.json().get("response", "").strip() or None
        except httpx.HTTPError:
            return None
        return None

    async def _call_gemini(self, prompt: str) -> str | None:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.gemini_model}:generateContent?key={self.gemini_api_key}"
        )
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.post(
                    url,
                    json={"contents": [{"parts": [{"text": prompt}]}]},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (httpx.HTTPError, KeyError, IndexError):
            return None
        return None
