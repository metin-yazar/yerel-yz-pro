# Bağımsız Otonom Yapay Zeka Uygulaması (Full-Stack v0.2)

Kendi verisini üretebilen, sohbetten otomatik bilgi çıkarabilen, sağlayıcıdan bağımsız (yerel Ollama LLM veya Gemini API) ve GitHub CI/CD uyumlu full-stack yapay zeka uygulama mimarisi.

## Mimari Yapı
- `frontend/index.html`: Mobil uyumlu kontrol merkezi (Canlı Test, Otonom Veri Üretici, Bilgi Tabanı, Talimat, GitHub/JSONL Dışa Aktarma)
- `backend/app/main.py`: FastAPI REST API (`/api/chat`, `/api/knowledge`, `/api/autonomous/synthesize`, `/api/export/jsonl`)
- `backend/app/ai_engine.py`: Sağlayıcıdan bağımsız LLM motoru (Ollama yerel model / Gemini API / Yerel RAG) ve otonom veri çıkarıcı
- `backend/app/memory_store.py`: SQLite kalıcı hafıza ve ince ayar (fine-tuning) için JSONL dışa aktarma katmanı
- `docker-compose.yml`: Tek komutla full-stack + yerel LLM (Ollama) ayağa kaldırma
- `.github/workflows/ci.yml`: Her push işleminde otomatik çalışan GitHub Actions test hattı

## Kurulum ve Çalıştırma
```bash
cp .env.example .env
docker compose up --build
```
Tam bağımsız yerel model (Ollama) ile çalıştırmak için:
```bash
docker compose --profile local-llm up --build
```
Uygulama `http://localhost:8000` adresinde açılır.
