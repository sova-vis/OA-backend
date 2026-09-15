"""
Retrieval for Ask-AI over the Supabase REST API (PostgREST) — the same channel
the app uses for everything else, so it works against the firewalled Oracle
self-hosted Supabase from anywhere with HTTPS. No direct Postgres connection.

Semantic search calls the public.match_ask_ai_chunks RPC (migration 024); the
exact-term boost is a REST `ilike` select. Two modes (unchanged):
  paper_lookup — "which years / how often was X asked"
  general_qa   — "explain X" (grounding chunks for the LLM)

Every search is scoped to ONE level (olevel|alevel) so O- and A-level never mix,
and optionally to a subject.

CLI test (needs ASK_AI_SUPABASE_URL + ASK_AI_SUPABASE_SERVICE_KEY):
    python scripts/retrieve.py "which years was electrolysis asked" --level olevel
"""

import datetime
import os
import re
import sys
import threading
from pathlib import Path

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from supabase import create_client

PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")

MODEL_NAME = os.environ.get("ASK_AI_EMBED_MODEL", "BAAI/bge-base-en-v1.5")
# bge retrieval convention: QUERIES get this instruction; the passages in the
# index do not (the build job embeds them raw). Keep in sync with the build.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class VectorStoreUnavailable(RuntimeError):
    """The REST index is unreachable or not built (migration 024 not applied) —
    the HTTP layer turns this into a clear 503, and the container still boots."""


PAPER_LOOKUP_PATTERNS = [
    r"\bwhich years?\b", r"\bwhat years?\b", r"\bhow many times\b", r"\bhow often\b",
    r"\bhas .* (been )?asked\b", r"\bwas .* asked\b", r"\bpast papers?\b",
    r"\bprevious (years?|papers?|exams?)\b", r"\bshow me questions?\b",
    r"\bfind questions?\b", r"\blist questions?\b",
]
PAPER_LOOKUP_RE = re.compile("|".join(PAPER_LOOKUP_PATTERNS), re.IGNORECASE)

YEAR_LIMIT_RE = re.compile(r"\b(?:last|past|previous|recent)\s+(\d{1,2})\s+years?\b", re.IGNORECASE)
DEFAULT_YEAR_LIMIT = 5


def requested_year_limit(query: str) -> int:
    m = YEAR_LIMIT_RE.search(query)
    return int(m.group(1)) if m else DEFAULT_YEAR_LIMIT


# Union of O- and A-level subject names, for auto-detecting a subject from free
# text when the UI didn't pin one. Longest first so "Business Studies" /
# "Computer Science" match before a shorter substring would.
SUBJECTS = [
    "Accounting", "Additional Maths", "Art and Design", "Biology", "Business Studies",
    "Business", "Chemistry", "Commerce", "Computer Science", "Economics",
    "English General Paper", "English Language", "English", "Environmental Management",
    "Further Mathematics", "Geography", "Global Perspectives", "History",
    "Information Technology", "Islamiyat", "Law", "Literature in English",
    "Mathematics", "Pakistan Studies", "Physics", "Psychology", "Religious Studies",
    "Sociology", "Statistics",
]
_SUBJECT_RES = sorted(
    ((s, re.compile(r"\b" + re.escape(s) + r"\b", re.IGNORECASE)) for s in SUBJECTS),
    key=lambda pair: -len(pair[0]),
)

ACRONYM_RE = re.compile(r"\b[A-Z]{2,6}\b")

GENERIC_QUERY_WORDS = {
    "in", "the", "a", "an", "was", "were", "is", "are", "asked", "for", "of",
    "on", "about", "come", "up", "to", "this", "that", "topic", "questions",
    "question", "how", "many", "times", "did", "has", "have", "had", "been",
    "being", "which", "what", "years", "year", "show", "find", "list", "past",
    "papers", "paper", "and", "related", "does", "do",
}


def detect_subject(query: str):
    for name, pattern in _SUBJECT_RES:
        if pattern.search(query):
            return name
    return None


def extract_core_topic(query: str, subject=None) -> str:
    text = YEAR_LIMIT_RE.sub(" ", query)
    if subject:
        text = re.sub(re.escape(subject), " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[?.!,]", " ", text)
    words = [w for w in text.split() if w.lower() not in GENERIC_QUERY_WORDS]
    return " ".join(words).strip()


def normalize_level(level):
    """Canonicalize to the value stored in questions.level ('olevel'|'alevel').
    Accepts 'O'/'A', 'O Level', 'a_level', etc. None -> no level filter."""
    if not level:
        return None
    s = re.sub(r"[^a-z]", "", str(level).lower())
    if s.startswith("o"):
        return "olevel"
    if s.startswith("a"):
        return "alevel"
    return None


# --- shared, lazily-initialised singletons (model + REST client) --------------
_model = None
_model_lock = threading.Lock()
_sb = None
_sb_lock = threading.Lock()


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = SentenceTransformer(MODEL_NAME)
    return _model


def get_sb():
    global _sb
    if _sb is None:
        url = (os.environ.get("ASK_AI_SUPABASE_URL") or os.environ.get("SUPABASE_URL") or "").strip()
        key = (os.environ.get("ASK_AI_SUPABASE_SERVICE_KEY")
               or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
               or os.environ.get("SUPABASE_KEY") or "").strip()
        if not url or not key:
            raise VectorStoreUnavailable(
                "Supabase REST not configured (ASK_AI_SUPABASE_URL / ASK_AI_SUPABASE_SERVICE_KEY)."
            )
        with _sb_lock:
            if _sb is None:
                _sb = create_client(url, key)
    return _sb


_SELECT_COLS = ("question_id,level,subject,type,exam_year,session,paper,variant,"
                "question_number,topic,content")


def _to_hit(r: dict, verified: bool) -> dict:
    """Map an ask_ai_chunks row to the hit shape generate_answer.py expects
    (exam_year -> year). No source PDF page from the DB, so source_file/page are
    None — citations then show the paper reference without a page image."""
    meta = {
        "subject": r.get("subject"),
        "year": r.get("exam_year"),
        "session": r.get("session"),
        "paper": r.get("paper"),
        "variant": r.get("variant"),
        "question_number": r.get("question_number"),
        "topic": r.get("topic"),
        "level": r.get("level"),
        "type": r.get("type"),
        "source_file": None,
        "page": None,
    }
    return {
        "text": r.get("content") or "",
        "metadata": meta,
        "distance": float(r.get("distance") or 0.0),
        "verified": verified,
    }


class Retriever:
    def __init__(self):
        self.model = get_model()

    def classify_intent(self, query: str) -> str:
        return "paper_lookup" if PAPER_LOOKUP_RE.search(query) else "general_qa"

    def search(self, query: str, subject=None, level=None, top_k: int = 10, core_topic=None):
        vec = self.model.encode([QUERY_INSTRUCTION + query], normalize_embeddings=True)[0]
        # pgvector text form — PostgREST casts this string to vector for the RPC.
        qvec = "[" + ",".join(f"{float(x):.6f}" for x in vec) + "]"
        level = normalize_level(level)
        sb = get_sb()
        try:
            # Semantic search via the pgvector RPC (always scoped by level).
            semantic = sb.rpc("match_ask_ai_chunks", {
                "query_embedding": qvec,
                "match_level": level,
                "match_subject": subject,
                "match_count": top_k,
            }).execute().data or []

            # Exact-term boost: acronyms + core topic, so a genuine match just
            # outside top-k (or an acronym embeddings rank poorly) isn't dropped.
            terms = set(ACRONYM_RE.findall(query))
            if core_topic and len(core_topic) >= 4:
                terms.add(core_topic)
            verified_ids = set()
            exact = []
            for term in terms:
                q = sb.table("ask_ai_chunks").select(_SELECT_COLS).ilike("content", f"*{term}*").limit(200)
                if level:
                    q = q.eq("level", level)
                if subject:
                    q = q.eq("subject", subject)
                for r in (q.execute().data or []):
                    verified_ids.add(r["question_id"])
                    exact.append(r)
        except VectorStoreUnavailable:
            raise
        except Exception as e:
            raise VectorStoreUnavailable(
                f"Ask-AI index query failed (is migration 024 applied and the index built?): {e}"
            ) from e

        by_id: dict = {}
        for r in semantic:
            by_id[r["question_id"]] = _to_hit(r, verified=False)
        for r in exact:
            if r["question_id"] not in by_id:
                by_id[r["question_id"]] = _to_hit(r, verified=True)
        for qid in verified_ids:
            if qid in by_id:
                by_id[qid]["verified"] = True

        hits = sorted(by_id.values(), key=lambda h: h["distance"])
        verified_hits = [h for h in hits if h["verified"]]
        other_hits = [h for h in hits if not h["verified"]]
        return verified_hits + other_hits[: max(0, top_k - len(verified_hits))]

    def paper_lookup_summary(self, hits, year_limit=None):
        verified_hits = [h for h in hits if h.get("verified")]
        candidates = verified_hits if verified_hits else hits
        seen = set()
        occurrences = []
        for h in candidates:
            m = h["metadata"]
            if m.get("question_number") is None:
                continue
            key = (m.get("subject"), m.get("year"), m.get("session"), m.get("paper"),
                   m.get("variant"), m.get("question_number"))
            if key in seen:
                continue
            seen.add(key)
            occurrences.append({**m, "question_text": h["text"]})
        if year_limit is not None and occurrences:
            this_year = datetime.date.today().year
            cutoff_year = this_year - year_limit + 1
            occurrences = [m for m in occurrences if (m.get("year") or 0) >= cutoff_year]
        occurrences.sort(key=lambda m: (m.get("year") or 0, m.get("session") or ""))
        return occurrences

    def route(self, query: str, subject=None, level=None, top_k: int = 10):
        intent = self.classify_intent(query)
        if subject is None:
            subject = detect_subject(query)
        search_query = YEAR_LIMIT_RE.sub("", query).strip()
        core_topic = extract_core_topic(query, subject)
        hits = self.search(search_query, subject=subject, level=level, top_k=top_k, core_topic=core_topic)
        result = {"intent": intent, "hits": hits}
        if intent == "paper_lookup":
            year_limit = requested_year_limit(query)
            result["year_limit"] = year_limit
            result["occurrences"] = self.paper_lookup_summary(hits, year_limit=year_limit)
        return result


def main():
    args = sys.argv[1:]
    level = None
    if "--level" in args:
        i = args.index("--level")
        level = args[i + 1] if i + 1 < len(args) else None
        del args[i:i + 2]
    query = args[0] if args else "which years was depreciation asked in accounting"

    r = Retriever()
    result = r.route(query, level=level)
    print(f"Query: {query!r}  (level={level})")
    print(f"Detected intent: {result['intent']}\n")
    if result["intent"] == "paper_lookup":
        print("Occurrences found:")
        for m in result["occurrences"]:
            print(f"  - {m.get('subject')} {m.get('year')} {m.get('session')} "
                  f"Paper {m.get('paper')} Variant {m.get('variant')} Q{m.get('question_number')}")
        print()
    print("Top retrieved chunks:")
    for h in result["hits"][:5]:
        m = h["metadata"]
        print(f"--- distance={h['distance']:.4f} | {m.get('subject')} {m.get('year')} "
              f"{m.get('session')} Paper {m.get('paper')} Q{m.get('question_number')} ---")
        print(h["text"][:200].replace("\n", " "))
        print()


if __name__ == "__main__":
    main()
