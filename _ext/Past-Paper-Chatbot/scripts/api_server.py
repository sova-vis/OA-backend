"""
Step 7: HTTP API for the "Past-Paper app" (frontend-main + OA-backend-main)
to call into this project's RAG pipeline, replacing that app's own
"Ask AI" chatbot implementation.

Exposes:
  POST /chat         -> matches the shape OA-backend-main's /rag/query
                         already returns, so the Node backend can proxy to
                         this with minimal translation.
  GET  /page-image    -> renders one exact PDF page (downloaded from Drive)
                         as a PNG. Answer links point here instead of a
                         Drive view link, because neither Drive's own viewer
                         nor Google Docs Viewer honors a "#page=N" jump on a
                         direct link (confirmed by testing - both stay on
                         page 1 regardless of the fragment). An image link
                         can't have that problem: there's no "jump", the
                         requested page is just what gets rendered.

Run:
    venv\\Scripts\\python.exe -m uvicorn scripts.api_server:app --host 0.0.0.0 --port 8002
"""

import os
import sys
from pathlib import Path
from urllib.parse import quote, unquote

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_answer import DRIVE_FILE_MAP, _normalize_path, generate, question_preview  # noqa: E402
from drive_pdf import download_pdf_bytes, render_page_image  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")

PUBLIC_BASE_URL = os.environ.get("CHATBOT_SERVICE_PUBLIC_URL", "http://localhost:8002").rstrip("/")

app = FastAPI(title="Past-Paper Chatbot API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_pdf_bytes_cache: dict[str, bytes] = {}
_page_image_cache: dict[tuple[str, int], bytes] = {}


def _cached_pdf_bytes(drive_id: str) -> bytes:
    if drive_id not in _pdf_bytes_cache:
        _pdf_bytes_cache[drive_id] = download_pdf_bytes(drive_id)
    return _pdf_bytes_cache[drive_id]


def _cached_page_image(drive_id: str, page: int) -> bytes:
    key = (drive_id, page)
    if key not in _page_image_cache:
        _page_image_cache[key] = render_page_image(_cached_pdf_bytes(drive_id), page)
    return _page_image_cache[key]


def image_link_builder(source_file: str, page: int) -> str:
    encoded = quote(source_file.replace("\\", "/"), safe="")
    return f"{PUBLIC_BASE_URL}/page-image?source={encoded}&page={page}"


class HistoryMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    limit: int | None = None
    history: list[HistoryMessage] | None = None
    subject: str | None = None
    mode: str | None = None  # "ask" | "find" | None (auto-detect)


def build_citation(occurrence: dict) -> dict:
    """pageImageUrl and preview let the frontend render each occurrence as
    its own clickable card (its existing citation-card UI) instead of
    needing to parse a markdown table out of the answer text - the
    frontend has no markdown renderer at all, so a table embedded in
    `answer` would only ever show up as raw text."""
    return {
        "subject": occurrence.get("subject"),
        "year": occurrence.get("year"),
        "session": occurrence.get("session"),
        "paper": occurrence.get("paper"),
        "variant": occurrence.get("variant"),
        "questionNumber": occurrence.get("question_number"),
        "topicGeneral": None,
        "topicSyllabus": None,
        "preview": question_preview(occurrence.get("question_text", "")),
        "pageImageUrl": image_link_builder(occurrence.get("source_file", ""), occurrence.get("page", 1)),
    }


@app.post("/chat")
def chat(req: ChatRequest):
    answer, result = generate(
        req.question, subject=req.subject, mode=req.mode,
        include_table=False, link_builder=image_link_builder,
    )

    occurrences = result.get("occurrences") or []
    is_find = result.get("intent") == "paper_lookup"
    # Both modes show every match - Ask mode's explanation sits above the
    # same full table Find mode shows, rather than a single truncated
    # reference (which looked like a bug once both modes render as a table).
    citations = [build_citation(o) for o in occurrences]

    return {
        "type": "exam_question",
        "mode": "find" if is_find else "ask",
        "answer": answer,
        "citations": citations,
        "source_type": "past_paper" if occurrences else "none",
    }


@app.get("/page-image")
def page_image(source: str = Query(...), page: int = Query(1, ge=1)):
    source_file = unquote(source)
    drive_id = DRIVE_FILE_MAP.get(_normalize_path(source_file))
    if not drive_id:
        raise HTTPException(status_code=404, detail="Paper not found in Drive map")

    try:
        png_bytes = _cached_page_image(drive_id, page)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not render page: {e}")

    return Response(content=png_bytes, media_type="image/png")


@app.get("/health")
def health():
    return {"status": "ok"}
