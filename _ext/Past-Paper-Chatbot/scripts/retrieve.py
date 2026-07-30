"""
Step 4: Retrieval logic / query routing.

Classifies an incoming user question into one of two modes:

  paper_lookup  - "which years was X asked", "how many times has Y come up",
                   "show me questions about Z" -> pure semantic search over
                   the question index; the answer is really just "here are
                   the matching questions and when they appeared."

  general_qa    - "explain X", "what is Y", "how do I calculate Z" -> needs
                   retrieved chunks as grounding *plus* general knowledge
                   (and, in Step 5, a web search) to produce an explanation.

This module only does retrieval + classification; turning results into a
final natural-language answer is Step 5 (LLM generation).

Run as a CLI for testing:
    venv\\Scripts\\python.exe scripts\\retrieve.py "which years was depreciation asked in accounting"
"""

import datetime
import re
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

PROJECT_DIR = Path(__file__).resolve().parent.parent
VECTOR_STORE_DIR = PROJECT_DIR / "data" / "vector_store"
COLLECTION_NAME = "past_papers"
MODEL_NAME = "all-MiniLM-L6-v2"

# Phrases that signal the user wants to know *where in the papers* a topic
# shows up (years/sessions/frequency), not an explanation of the topic.
PAPER_LOOKUP_PATTERNS = [
    r"\bwhich years?\b",
    r"\bwhat years?\b",
    r"\bhow many times\b",
    r"\bhow often\b",
    r"\bhas .* (been )?asked\b",
    r"\bwas .* asked\b",
    r"\bpast papers?\b",
    r"\bprevious (years?|papers?|exams?)\b",
    r"\bshow me questions?\b",
    r"\bfind questions?\b",
    r"\blist questions?\b",
]
PAPER_LOOKUP_RE = re.compile("|".join(PAPER_LOOKUP_PATTERNS), re.IGNORECASE)

# "last 10 years" / "past 3 years" / "previous 8 years" -> use that many most
# recent years instead of the default window.
YEAR_LIMIT_RE = re.compile(r"\b(?:last|past|previous|recent)\s+(\d{1,2})\s+years?\b", re.IGNORECASE)
DEFAULT_YEAR_LIMIT = 5


def requested_year_limit(query: str) -> int:
    m = YEAR_LIMIT_RE.search(query)
    return int(m.group(1)) if m else DEFAULT_YEAR_LIMIT


SUBJECTS = [
    "Accounting", "Additional Maths", "Art and Design", "Biology", "Business Studies",
    "Chemistry", "Commerce", "Computer Science", "Economics", "English",
    "Environmental Management", "Geography", "History", "Islamiyat", "Mathematics",
    "Pakistan Studies", "Physics", "Religious Studies", "Sociology", "Statistics",
]
# Longest names first so "Computer Science" matches before a shorter substring would.
_SUBJECT_RES = sorted(
    ((s, re.compile(r"\b" + re.escape(s) + r"\b", re.IGNORECASE)) for s in SUBJECTS),
    key=lambda pair: -len(pair[0]),
)

# Short (2-6 letter) all-caps tokens in the query, e.g. "SSL", "GDP", "CPU" -
# dense embeddings are unreliable for acronyms, so these get an exact-text
# match pass in addition to semantic search.
ACRONYM_RE = re.compile(r"\b[A-Z]{2,6}\b")

GENERIC_QUERY_WORDS = {
    "in", "the", "a", "an", "was", "were", "is", "are", "asked", "for", "of",
    "on", "about", "come", "up", "to", "this", "that", "topic", "questions",
    "question", "how", "many", "times", "did", "has", "have", "had", "been",
    "being", "which", "what", "years", "year", "show", "find", "list", "past",
    "papers", "paper", "and", "related", "does", "do",
}


def detect_subject(query: str) -> str | None:
    for name, pattern in _SUBJECT_RES:
        if pattern.search(query):
            return name
    return None


def extract_core_topic(query: str, subject: str | None = None) -> str:
    """Strip boilerplate phrasing ('which years was ... asked', 'in past N
    years', the subject name, generic filler words) to get at the actual
    topic being asked about, e.g. 'photosynthesis' out of 'Which years was
    photosynthesis asked in Biology in past 8 years?'. Used for an exact
    substring match pass so results aren't limited to whatever a top-k=10
    semantic search happens to rank highly.

    Note: deliberately NOT reusing PAPER_LOOKUP_RE here - its `\\bwas .*
    asked\\b` pattern is greedy and, applied as a substitution, would eat
    everything between "was" and "asked" - including the topic word itself
    (e.g. "was photosynthesis asked" -> topic swallowed). Word-level
    filtering against GENERIC_QUERY_WORDS below handles "was"/"asked" fine
    without that risk."""
    text = YEAR_LIMIT_RE.sub(" ", query)
    if subject:
        text = re.sub(re.escape(subject), " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[?.!,]", " ", text)
    words = [w for w in text.split() if w.lower() not in GENERIC_QUERY_WORDS]
    return " ".join(words).strip()


class Retriever:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
        self.collection = client.get_collection(COLLECTION_NAME)

    def classify_intent(self, query: str) -> str:
        return "paper_lookup" if PAPER_LOOKUP_RE.search(query) else "general_qa"

    def search(self, query: str, subject: str | None = None, top_k: int = 10,
               core_topic: str | None = None):
        where = {"subject": subject} if subject else None

        embedding = self.model.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=embedding, n_results=top_k, where=where
        )
        hits = []
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            hits.append({"text": doc, "metadata": meta, "distance": dist})

        # Exact-match boost: semantic search alone caps out at top_k, so a
        # genuine match that just misses the top-k cut (or an acronym like
        # "SSL"/"GDP" that embeddings handle poorly) can be silently dropped.
        # This applies both to acronyms AND the query's core topic phrase
        # (e.g. "photosynthesis") - for a "how many times was X asked"
        # question we need every real occurrence, not just whatever ranked
        # highest by semantic similarity. Uncapped by design.
        # Note: a broad/common term (e.g. "GDP" or "photosynthesis") will
        # legitimately show a high count since it's referenced across many
        # genuinely different questions - that's real completeness, not noise,
        # even though it may include some passing mentions alongside the core
        # matches. A semantic-relevance gate on top of the exact match (reject
        # a hit if its embedding is too far from the query) was tried and
        # reverted: no single threshold behaved consistently across queries -
        # a margin loose enough to keep legitimate SSL sub-part questions
        # (where the term appears in part (b)/(c), not the opening line) was
        # also loose enough to let through nearly all the noise it was meant
        # to filter for other topics, and a stricter margin wrongly excluded
        # clearly-relevant matches. Pure exact-match completeness is more
        # predictable and honest than a fragile approximation of relevance.
        exact_match_terms = set(ACRONYM_RE.findall(query))
        if core_topic and len(core_topic) >= 4:
            exact_match_terms.add(core_topic)

        verified_texts = set()
        for term in exact_match_terms:
            variants = {term}
            if term[:1].islower():
                variants.add(term[0].upper() + term[1:])  # catch sentence-start capitalization
            for variant in variants:
                exact = self.collection.get(
                    where=where,
                    where_document={"$contains": variant},
                    include=["documents", "metadatas"],
                )
                for doc, meta in zip(exact["documents"], exact["metadatas"]):
                    # Only actual exam questions count as "occurrences"; whole-document
                    # chunks (syllabus, mark schemes, examiner reports) have no question_number.
                    if meta.get("question_number") is None:
                        continue
                    verified_texts.add(doc)
                    if not any(h["text"] == doc for h in hits):
                        hits.append({"text": doc, "metadata": meta, "distance": 0.0})

        # Tag each hit so callers can tell a confirmed exact-text match apart
        # from a purely-semantic neighbor pulled in to fill out top_k. This
        # matters a lot for rare topics: if only 1 chunk genuinely contains
        # "franchising", the remaining top_k slots get filled with whatever
        # is semantically closest - which is NOT the same as containing the
        # word, and must never be reported as a genuine occurrence.
        for h in hits:
            h["verified"] = h["text"] in verified_texts

        # Verified exact matches are NEVER truncated, regardless of whether
        # they came from semantic search (possibly with a non-zero distance)
        # or the exact-match pass - a hit that overlapped with the semantic
        # top_k must not silently reduce how many verified matches survive the
        # cutoff. Only the purely-semantic (unverified) hits are capped.
        hits.sort(key=lambda h: h["distance"])
        verified_hits = [h for h in hits if h["verified"]]
        other_hits = [h for h in hits if not h["verified"]]
        return verified_hits + other_hits[: max(0, top_k - len(verified_hits))]

    def paper_lookup_summary(self, hits, year_limit: int | None = None):
        """Group hits by (subject, year, session, paper, variant, question) so
        the caller can answer 'which years was this asked' directly.

        year_limit restricts results to a literal recent calendar window
        anchored on TODAY's real date (e.g. 5 -> only years within
        [this_year - 4, this_year]) - NOT relative to whichever year this
        particular topic last happened to appear in. A topic last asked in
        2019 is genuinely outside "the last 5 years" if today is 2026, even
        though 2019 is that topic's own most recent occurrence; anchoring on
        the topic's own max year would wrongly call 2019 "recent" just
        because nothing more recent existed for that specific topic.

        Only uses exact-text-verified hits when any exist: for a rare topic
        with few genuine matches, search() pads the remaining top_k slots
        with the closest semantic neighbors to keep context useful for
        general_qa answers - but those are NOT confirmed occurrences of the
        term and must not be reported as "this was asked in year X" (this
        was a real bug: a semantically-similar-but-unrelated question was
        being listed as a match for "franchising" purely because it filled a
        leftover top_k slot). Falls back to all hits only if verification
        couldn't identify anything at all, so a query that couldn't extract
        a usable exact-match term doesn't silently return zero results."""
        verified_hits = [h for h in hits if h.get("verified")]
        candidates = verified_hits if verified_hits else hits

        seen = set()
        occurrences = []
        for h in candidates:
            m = h["metadata"]
            if m.get("question_number") is None:
                continue  # not an actual exam question (e.g. an examiner report)
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

    def route(self, query: str, subject: str | None = None, top_k: int = 10):
        intent = self.classify_intent(query)
        # If the caller didn't pin a subject (e.g. via the UI filter), infer
        # one from the query text itself when it names a subject explicitly.
        if subject is None:
            subject = detect_subject(query)
        # Strip any "last/past N years" phrase before it's used for semantic
        # search: leaving it in shifts the embedding away from the actual
        # topic (e.g. "photosynthesis ... in past 8 years" embeds noticeably
        # differently than "photosynthesis"), silently changing which chunks
        # get retrieved based purely on how the year window was phrased -
        # completely unrelated to the topic itself.
        search_query = YEAR_LIMIT_RE.sub("", query).strip()
        core_topic = extract_core_topic(query, subject)
        hits = self.search(search_query, subject=subject, top_k=top_k, core_topic=core_topic)
        result = {"intent": intent, "hits": hits}
        if intent == "paper_lookup":
            year_limit = requested_year_limit(query)
            result["year_limit"] = year_limit
            result["occurrences"] = self.paper_lookup_summary(hits, year_limit=year_limit)
        return result


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "which years was depreciation asked in accounting"
    r = Retriever()
    result = r.route(query)

    print(f"Query: {query!r}")
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
