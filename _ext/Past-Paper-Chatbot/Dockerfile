# Past-Paper Chatbot — Ask-AI RAG service.
# Serves POST /chat (Ask/Find) and GET /page-image, which OA-backend proxies
# to via CHATBOT_SERVICE_URL. Listens on $PORT (Railway/Render inject it),
# falling back to 8002 for local `docker run`.

FROM python:3.11-slim

# PyMuPDF and chromadb ship manylinux wheels, so no apt build chain is needed.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Bake the embedding model into the image so the first request doesn't pay a
# cold download (and the container needs no outbound HF access at runtime).
# Must match ASK_AI_EMBED_MODEL / retrieve.py (BAAI/bge-base-en-v1.5, 768-dim).
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-base-en-v1.5')"

# App code. The search index lives in your self-hosted Supabase Postgres
# (public.ask_ai_chunks / pgvector), NOT in the image — the service reaches it
# via ASK_AI_DATABASE_URL at runtime. See DEPLOY.md.
COPY . .

EXPOSE 8002

# Shell form so ${PORT} is expanded by the platform at start.
CMD ["sh", "-c", "uvicorn scripts.api_server:app --host 0.0.0.0 --port ${PORT:-8002}"]
