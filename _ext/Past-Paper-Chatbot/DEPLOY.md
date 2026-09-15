# Deploying Ask-AI (Past-Paper Chatbot service)

This service is the "brain" behind **Ask AI**. `OA-backend`'s `/rag/query` is a
thin proxy that forwards to this service's `POST /chat`.

```
Student → OA-frontend → OA-backend  /rag/query ──proxy──▶ THIS service /chat
                                     /rag/ask-image  (image path — Grok, not this service)
```

**Retrieval runs on pgvector in your Supabase, reached over the Supabase REST API
(PostgREST)** — the same channel the app uses to read practice questions. There
is no separate vector DB and **no direct Postgres connection**, so it works
against the firewalled Oracle self-hosted Supabase from anywhere with HTTPS
(including Railway). The index is one new table, `public.ask_ai_chunks`, built
from the question bank that already lives in that database.

## What it needs

| Requirement | Notes |
|---|---|
| The container (`Dockerfile` + `requirements.txt`) | provided here |
| **`ask_ai_chunks` table + `match_ask_ai_chunks` function** | migration `024` (one-time DDL) |
| **The index, built** | the build job below (read-only on `questions`/`question_parts`) |
| `ASK_AI_SUPABASE_URL` + `ASK_AI_SUPABASE_SERVICE_KEY` | your Supabase REST URL + service_role key |
| At least one LLM key | `XAI_API_KEY` (primary) or `SAMBANOVA_API_KEY` / `OPENROUTER_API_KEY`. Ask only; Find is pure retrieval. |

## Step 1 — Apply migration 024 (one-time DDL)

`OA-backend/migrations/024_ask_ai_embeddings.sql` creates the `ask_ai_chunks`
table + the `match_ask_ai_chunks` RPC and enables `vector`/`pg_trgm`. It's
additive — it never touches `questions`/`question_parts`.

This is the **only** step that needs direct DB access (PostgREST can't run DDL).
Run it once the way you run your other migrations against the Oracle DB — e.g.
paste it into the self-hosted **Supabase Studio → SQL editor**, or run your
migration runner from the Oracle box (localhost Postgres). If HNSW isn't
supported by your pgvector version, see the note in the file.

## Step 2 — Build the index (over REST, runs anywhere)

Reads your questions **read-only** over REST, embeds each with
`bge-base-en-v1.5` (free, local), and upserts `ask_ai_chunks` for **both O and A
level**. Idempotent — re-run any time the bank changes.

```bash
pip install -r requirements.txt
ASK_AI_SUPABASE_URL="https://<your-supabase-host>" \
ASK_AI_SUPABASE_SERVICE_KEY="<service_role key>" \
  python scripts/build_embeddings_from_db.py
# smoke test first:  ... build_embeddings_from_db.py --subject "Biology" --limit 500
```

Embedding the full O+A bank on CPU takes roughly 30–60 min. Verify with:
`... python scripts/retrieve.py "which years was electrolysis asked" --level olevel`.

## Step 3 — Deploy the service

`railway.json` + `Dockerfile` target Railway. Because it only needs **HTTPS to
Supabase** (no DB port), it can run anywhere. Set from `.env.example`:

- `ASK_AI_SUPABASE_URL` + `ASK_AI_SUPABASE_SERVICE_KEY` — same Supabase as the build
- `XAI_API_KEY` (or a free fallback) — writes the answers
- `CHATBOT_SERVICE_PUBLIC_URL` — this service's public URL

`docker build -t ppchatbot . && docker run -p 8002:8002 --env-file .env ppchatbot`,
then `curl localhost:8002/health`. If the index isn't built yet, `/chat` returns
a clear **503** (container still boots). Budget ~2 GB RAM (loads the bge model).

## Step 4 — Point OA-backend at it

Set on **OA-backend**: `CHATBOT_SERVICE_URL=https://<this-service-public-url>`.
`rag.routes.ts` reads it (default `http://127.0.0.1:8002` for local dev).

## How O and A level stay separate

Every chunk is tagged with its `level` (`olevel`/`alevel`). The frontend sends the
student's `active_level` with each question, and `match_ask_ai_chunks` **always
filters on it** — O-level students never see A-level results and vice-versa. Each
chunk is also tagged with `subject`, `paper`, `variant`, `exam_year`, `session`,
`question_number`, and `topic`, for topic-wise and paper-wise scoping/grouping.

## Legacy

The old PDF → Chroma pipeline (`extract_text.py`, `chunk_questions.py`,
`build_vector_store.py`, `drive_file_map.json`, `/page-image`) is superseded and
off the live path — left in the repo for reference only.
