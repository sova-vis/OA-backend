"""
Build the Ask-AI semantic index from the question bank, entirely over the
Supabase REST API (PostgREST) — the same way the app reads practice questions.
No direct Postgres connection, so it works against the firewalled Oracle
self-hosted Supabase from anywhere with HTTPS.

READ-ONLY on the source: SELECTs public.questions / public.question_parts via
REST. Writes only to public.ask_ai_chunks (created by migration 024). One
embedded row per question, tagged level/subject/paper/variant/year/topic so
retrieval scopes by any of them and O- and A-level never mix.

Resumable: skips question_ids already in ask_ai_chunks, so a re-run continues
where a previous run stopped instead of re-embedding everything.

Model: BAAI/bge-base-en-v1.5 (768-dim); the same model embeds queries at serve
time (retrieve.py).

Run (after migration 024 has been applied):
    ASK_AI_SUPABASE_URL=https://<host> ASK_AI_SUPABASE_SERVICE_KEY=<service_role> \
      python scripts/build_embeddings_from_db.py

Options: --level olevel|alevel  --subject "Biology"  --limit N  --batch-size N  --no-resume
"""

import argparse
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from supabase import create_client

PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")

MODEL_NAME = os.environ.get("ASK_AI_EMBED_MODEL", "BAAI/bge-base-en-v1.5")
EMBED_DIM = 768
ZERO_UUID = "00000000-0000-0000-0000-000000000000"

Q_COLS = ("id,question_id,subject,level,type,exam_year,session,paper,variant,"
          "question_number,topic,question_text,options")


def client():
    url = (os.environ.get("ASK_AI_SUPABASE_URL") or os.environ.get("SUPABASE_URL") or "").strip()
    key = (os.environ.get("ASK_AI_SUPABASE_SERVICE_KEY")
           or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
           or os.environ.get("SUPABASE_KEY") or "").strip()
    if not url or not key:
        sys.exit("Set ASK_AI_SUPABASE_URL and ASK_AI_SUPABASE_SERVICE_KEY (service_role).")
    return create_client(url, key)


def render_options(options) -> str:
    """MCQ options -> 'A. ...\\nB. ...'. Tolerates {label,text}/{option,text}
    dicts, plain-string lists, or an object (PostgREST returns parsed JSON)."""
    if not options:
        return ""
    entries = []
    if isinstance(options, list):
        for i, opt in enumerate(options):
            if isinstance(opt, dict):
                label = str(opt.get("label") or opt.get("option") or chr(65 + i)).strip()
                text = str(opt.get("text") or opt.get("value") or "").strip()
            else:
                label, text = chr(65 + i), str(opt).strip()
            if text:
                entries.append(f"{label}. {text}")
    elif isinstance(options, dict):
        for label, text in options.items():
            if text:
                entries.append(f"{str(label).strip()}. {str(text).strip()}")
    return "\n".join(entries)


def compose_content(row: dict, parts: list) -> str:
    """The exact text we embed: the question, plus MCQ options or structured
    sub-parts. Answers are excluded — we index what a student searches for."""
    blocks = [str(row.get("question_text") or "").strip()]
    if row.get("type") == "mcq":
        opts = render_options(row.get("options"))
        if opts:
            blocks.append(opts)
    else:
        blocks.extend(str(p or "").strip() for p in parts)
    return "\n\n".join(b for b in blocks if b).strip()


def fetch_parts(sb, ids: list) -> dict:
    """Part bodies grouped by parent question uuid, in order."""
    if not ids:
        return {}
    rows = sb.table("question_parts").select("question_uid,order_index,body") \
        .in_("question_uid", ids).order("question_uid").order("order_index").execute().data
    grouped: dict = {}
    for r in rows:
        grouped.setdefault(r["question_uid"], []).append(r.get("body") or "")
    return grouped


def load_done_ids(sb) -> set:
    """question_ids already in ask_ai_chunks, so a resumed run skips them."""
    done = set()
    offset = 0
    while True:
        rows = sb.table("ask_ai_chunks").select("question_id").order("question_id").range(offset, offset + 999).execute().data
        if not rows:
            break
        done.update(r["question_id"] for r in rows)
        if len(rows) < 1000:
            break
        offset += 1000
    return done


def upsert_with_retry(sb, payload, attempts: int = 4) -> None:
    """A single transient REST hiccup shouldn't kill an hours-long build."""
    for i in range(attempts):
        try:
            sb.table("ask_ai_chunks").upsert(payload, on_conflict="question_id").execute()
            return
        except Exception:
            if i == attempts - 1:
                raise
            time.sleep(2 * (i + 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=["olevel", "alevel"], default=None)
    ap.add_argument("--subject", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    # Use all CPU cores for embedding — the main throughput lever on CPU.
    try:
        import torch
        torch.set_num_threads(max(1, os.cpu_count() or 4))
    except Exception:
        pass

    print(f"Loading embedding model {MODEL_NAME} ...", flush=True)
    model = SentenceTransformer(MODEL_NAME)
    dim = model.get_sentence_embedding_dimension()
    if dim != EMBED_DIM:
        sys.exit(f"Model outputs {dim}-dim but ask_ai_chunks.embedding is vector({EMBED_DIM}).")

    sb = client()
    done = set() if args.no_resume else load_done_ids(sb)
    last_id = ZERO_UUID
    if done:
        print(f"Resuming — {len(done)} questions already indexed; skipping them.", flush=True)
        # Jump near the resume boundary instead of re-scanning the whole done
        # prefix every restart (done rows are a contiguous id-ordered prefix).
        if not args.level and not args.subject:
            try:
                skip_to = max(0, len(done) - 300)
                r = sb.table("questions").select("id").order("id").range(skip_to, skip_to).execute().data
                if r:
                    last_id = r[0]["id"]
                    print(f"Resume: jumping to ~offset {skip_to} to skip re-scanning done rows.", flush=True)
            except Exception:
                pass  # fall back to a full scan-and-skip

    total = 0        # questions scanned
    embedded = 0     # newly embedded this run
    print("Building Ask-AI embeddings (reading questions over REST, read-only) ...", flush=True)

    while True:
        q = sb.table("questions").select(Q_COLS).gt("id", last_id).order("id").limit(args.batch_size)
        if args.level:
            q = q.eq("level", args.level)
        if args.subject:
            q = q.eq("subject", args.subject)
        rows = q.execute().data
        if not rows:
            break
        last_id = rows[-1]["id"]
        total += len(rows)

        todo = [r for r in rows if r["question_id"] not in done]
        if todo:
            parts_by_uid = fetch_parts(sb, [r["id"] for r in todo])
            prepared = []
            for r in todo:
                content = compose_content(r, parts_by_uid.get(r["id"], []))
                if content:
                    prepared.append((r, content))
            if prepared:
                vectors = model.encode([c for _, c in prepared],
                                       normalize_embeddings=True, batch_size=128)
                payload = [
                    {
                        "question_id": r["question_id"], "level": r.get("level"),
                        "subject": r.get("subject"), "type": r.get("type"),
                        "exam_year": r.get("exam_year"), "session": r.get("session"),
                        "paper": r.get("paper"), "variant": r.get("variant"),
                        "question_number": r.get("question_number"), "topic": r.get("topic"),
                        "content": content,
                        "embedding": "[" + ",".join(f"{float(x):.6f}" for x in vec) + "]",
                    }
                    for (r, content), vec in zip(prepared, vectors)
                ]
                upsert_with_retry(sb, payload)
                embedded += len(payload)

        print(f"  ... scanned {total}, {embedded} new embedded", flush=True)
        if args.limit and total >= args.limit:
            break

    print(f"Done. Scanned {total}; {embedded} newly embedded into public.ask_ai_chunks.", flush=True)


if __name__ == "__main__":
    main()
