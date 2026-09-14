# Deploying the Past-Paper Chatbot (Ask-AI RAG service)

This service is the "brain" behind **Ask AI**. `OA-backend` does not do retrieval
or generation itself — its `/rag/query` endpoint is a thin proxy that forwards to
this service's `POST /chat` (see `OA-backend/src/rag.routes.ts`). Until this
service is deployed and reachable, Ask/Find text answers fail with a 502.

```
Student → OA-frontend → OA-backend  /rag/query ──proxy──▶ THIS service /chat
                                     /rag/ask-image  (image path — Grok, not this service)
```

## What it needs to run

| Requirement | Notes |
|---|---|
| The container (this repo + `requirements.txt` + `Dockerfile`) | provided here |
| **A built Chroma vector store** at `data/vector_store/` | **NOT in git** — must be built or mounted (see below). This is the one hard prerequisite. |
| At least one LLM key | `XAI_API_KEY` (primary) or `SAMBANOVA_API_KEY` / `OPENROUTER_API_KEY`. Needed for **Ask** mode; Find mode is pure retrieval. |
| Google Drive OAuth (optional) | `GOOGLE_CLIENT_ID/SECRET/REFRESH_TOKEN` — only for the `/page-image` citation thumbnails. |

## Step 1 — Build the vector store (one-time, offline)

The index is gitignored because it is large and reproducible. Build it once from
the source past-paper PDFs:

```bash
python -m venv venv && source venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt

# 1. Put the source PDFs under data/raw/  (mirrors the Drive library structure;
#    the paths must match the keys in data/drive_file_map.json).
# 2. Extract text, chunk into questions, then embed into Chroma:
python scripts/extract_text.py        # data/raw/       → data/processed/
python scripts/chunk_questions.py     # data/processed/ → data/chunks/ (+ chunks.jsonl)
python scripts/build_vector_store.py  # data/chunks.jsonl → data/vector_store/  (Chroma)
```

Verify it loaded: `python scripts/retrieve.py "which years was photosynthesis asked in biology"`.

> **Version lock:** the `chromadb` version used to BUILD must match the one used to
> READ. After a successful build run `pip freeze > requirements.lock.txt` and
> deploy from that so build and runtime never drift.

## Step 2 — Get the index into the deployment

Pick one:

- **A. Bake it into the image (simplest).** From a machine that has `data/vector_store/`
  populated, deploy with the Railway CLI so the local files are in the build context:
  ```bash
  railway up          # uploads the local dir (incl. data/vector_store/) and builds the Dockerfile
  ```
  (`.dockerignore` deliberately keeps `data/vector_store/` so it ships; only the
  intermediate `raw/processed/chunks` are excluded.)

- **B. Mount a volume (keeps the image small, index updatable without a rebuild).**
  Create a Railway/Render volume, upload the built store into it, and set
  `VECTOR_STORE_DIR=/data/vector_store` to point the service at the mount.
  Building from GitHub push works with this option because the index is no longer
  expected inside the image.

If the index is missing, the container still boots and `/health` stays green, but
`/chat` returns **503** with a message telling you to build it — so a
misconfigured deploy is obvious instead of crash-looping.

## Step 3 — Deploy

`railway.json` + `Dockerfile` are set up for Railway (matches how `OA-backend`
deploys). The service listens on `$PORT` (injected by the platform). Locally:

```bash
docker build -t ppchatbot . && docker run -p 8002:8002 --env-file .env ppchatbot
```

Set the env vars from `.env.example`. **`CHATBOT_SERVICE_PUBLIC_URL` must be the
service's real public URL** (not localhost) or citation thumbnails 404.

## Step 4 — Point OA-backend at it

In the **OA-backend** environment set:

```
CHATBOT_SERVICE_URL=https://<this-service-public-url>
```

That's the only wiring needed — `rag.routes.ts` reads it (default
`http://127.0.0.1:8002` for local dev). Confirm end-to-end with an Ask query in
the app, or `curl https://<url>/health`.

## Known follow-up (not blocking)

`generate()` constructs a fresh `Retriever()` per request, which reloads the
MiniLM model on every call. For higher traffic, cache a single retriever instance
across requests (guarded for thread-safety) to cut per-request latency.
