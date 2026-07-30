"""
Step 5: LLM generation.

Takes a user question, runs it through Step 4's retriever to get intent +
grounding chunks, optionally pulls in live web search results for general_qa
questions, and asks a free LLM to produce a real answer with citations back
to the source papers.

Run:
    venv\\Scripts\\python.exe scripts\\generate_answer.py "which years was depreciation asked in accounting"
"""

import datetime
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote

import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieve import Retriever, requested_year_limit  # noqa: E402
from pdf_server import PDF_SERVER_PORT  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")


def _normalize_path(p: str) -> str:
    """Collapses whitespace differences between folder names on disk vs on
    Drive (e.g. "Physics" locally vs "Physics " on Drive) so the two can be
    matched segment-by-segment regardless of stray leading/trailing spaces."""
    return "/".join(seg.strip() for seg in p.replace("\\", "/").split("/"))


def _load_drive_map() -> dict:
    map_path = PROJECT_DIR / "data" / "drive_file_map.json"
    if not map_path.exists():
        return {}
    raw = json.loads(map_path.read_text(encoding="utf-8"))
    return {_normalize_path(k): v for k, v in raw.items()}


DRIVE_FILE_MAP = _load_drive_map()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
SAMBANOVA_API_KEY = os.environ.get("SAMBANOVA_API_KEY")
XAI_API_KEY = os.environ.get("XAI_API_KEY")

# Tried in order across providers; free tiers are prone to rate limits and
# congestion, so if one is unavailable we fall back to the next. xAI goes
# first: it's the paid provider, so no rate-limit/congestion concerns once
# the account has credits (currently doesn't - calls will 403 with
# "permission-denied" until credits are purchased at console.x.ai - falls
# through to the free providers below in the meantime, same as any other
# provider failure). SambaNova's Llama 3.3 70B goes next: confirmed
# genuinely free (unlike Cerebras's gpt-oss-120b, which returned 402
# Payment Required and was dropped), and a larger model than anything free
# on OpenRouter.
PROVIDERS = [
    {
        "name": "xai",
        "url": "https://api.x.ai/v1/chat/completions",
        "api_key": XAI_API_KEY,
        "model": "grok-4.5",
    },
    {
        "name": "sambanova",
        "url": "https://api.sambanova.ai/v1/chat/completions",
        "api_key": SAMBANOVA_API_KEY,
        "model": "Meta-Llama-3.3-70B-Instruct",
    },
    {
        "name": "openrouter",
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "api_key": OPENROUTER_API_KEY,
        "model": "openai/gpt-oss-20b:free",
    },
    {
        "name": "openrouter",
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "api_key": OPENROUTER_API_KEY,
        "model": "google/gemma-4-31b-it:free",
    },
    {
        "name": "openrouter",
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "api_key": OPENROUTER_API_KEY,
        "model": "nvidia/nemotron-3-super-120b-a12b:free",
    },
]


def web_search(query: str, max_results: int = 3):
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            return [
                {"title": r.get("title", ""), "body": r.get("body", ""), "href": r.get("href", "")}
                for r in ddgs.text(query, max_results=max_results)
            ]
    except Exception as e:
        print(f"[web search unavailable: {e}]", file=sys.stderr)
        return []


def build_context(result: dict, web_results: list):
    lines = []
    if result["intent"] == "paper_lookup":
        lines.append("Matching past-paper questions found (full detail for each occurrence):")
        for m in result["occurrences"]:
            lines.append(
                f"- Subject: {m.get('subject')} | Year: {m.get('year')} | "
                f"Session (month): {m.get('session')} | Paper: {m.get('paper')} | "
                f"Variant: {m.get('variant')} | Question number: {m.get('question_number')}\n"
                f"  Full question text: {m.get('question_text', '').strip()}"
            )
        lines.append("")
    else:
        lines.append("Retrieved past-paper excerpts:")
        for h in result["hits"][:6]:
            m = h["metadata"]
            lines.append(
                f"[{m.get('subject')} {m.get('year')} {m.get('session')} "
                f"Paper {m.get('paper')} Q{m.get('question_number')}]\n{h['text'][:500]}"
            )

    if web_results:
        lines.append("\nWeb search results:")
        for r in web_results:
            lines.append(f"[{r['title']}]({r['href']})\n{r['body']}")

    return "\n\n".join(lines)


# Matches the "letter alone on its own line, then its option text on the
# next line" layout multiple-choice questions get extracted into, e.g.
# "A \nextra carbon dioxide \nB \nextra dissolved nitrates \nC \n...".
# There's no explicit MCQ flag in the chunk metadata, so this text pattern
# is the only signal available - subjective/structured questions never
# produce four single-letter lines like this in sequence.
MCQ_OPTIONS_RE = re.compile(r"\n\s*A\s*\n.+\n\s*B\s*\n.+\n\s*C\s*\n.+\n\s*D\s*\n", re.DOTALL)


def is_mcq(text: str) -> bool:
    return bool(MCQ_OPTIONS_RE.search(text))


def select_worked_examples(hits: list, year_limit: int, limit: int = 3) -> list:
    """Up to `limit` real exam questions to walk through step-by-step in an
    Ask-mode answer: real questions only (never invented), subjective/
    structured questions only (no MCQs - a worked example is meant to model
    a full written answer, which an MCQ doesn't have), most relevant first
    (hits are already sorted verified-first then by embedding distance),
    restricted to the same recent-years window Find mode uses so "3 most
    relevant... from the last 5 years" means the same thing in both modes."""
    this_year = datetime.date.today().year
    cutoff_year = this_year - year_limit + 1
    seen = set()
    examples = []
    for h in hits:
        m = h["metadata"]
        if m.get("question_number") is None:
            continue  # not an actual exam question (e.g. syllabus/examiner report chunk)
        if (m.get("year") or 0) < cutoff_year:
            continue
        if is_mcq(h["text"]):
            continue
        key = (m.get("subject"), m.get("year"), m.get("session"), m.get("paper"),
               m.get("variant"), m.get("question_number"))
        if key in seen:
            continue
        seen.add(key)
        examples.append(h)
        if len(examples) >= limit:
            break
    return examples


def format_worked_examples_block(examples: list) -> str:
    lines = []
    for i, h in enumerate(examples, 1):
        m = h["metadata"]
        session = (m.get("session") or "").replace("_", "/")
        variant = f" Variant {m['variant']}" if m.get("variant") is not None else ""
        ref = f"{m.get('subject')} {m.get('year')} {session} Paper {m.get('paper')}{variant} Q{m.get('question_number')}"
        # Strip the bare leading question-number line the PDF extraction
        # leaves in place (e.g. a standalone "6" before the question stem) -
        # it's redundant since the ref above already gives the question
        # number, and left in, the model tends to reproduce it verbatim as
        # visual clutter.
        text = LEADING_QNUM_RE.sub("", h["text"].strip(), count=1)
        lines.append(f"Question {i} [{ref}]:\n{text}")
    return "\n\n".join(lines)


LEADING_QNUM_RE = re.compile(r"^\s*\d{1,2}\s*")

# Safety net for the "no leading OR on a bullet" prompt rule: LLM formatting
# compliance has proven inconsistent in testing (sometimes drops it, sometimes
# doesn't), so this strips it programmatically regardless of what the model
# actually output - matches "- OR " / "- Or " etc. at the start of a bullet
# line, case-insensitive, with or without a following colon.
BULLET_OR_PREFIX_RE = re.compile(r"^(\s*-\s+)or\b:?\s*", re.IGNORECASE | re.MULTILINE)


def strip_redundant_or_prefix(text: str) -> str:
    return BULLET_OR_PREFIX_RE.sub(r"\1", text)


def question_preview(text: str, num_words: int = 7) -> str:
    """A short 5-8 word snippet of the question, used as the link text in
    the results table (the full question text lives in the linked PDF page,
    not inline in the table anymore)."""
    text = LEADING_QNUM_RE.sub("", text.strip(), count=1)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split(" ")
    preview = " ".join(words[:num_words])
    if len(words) > num_words:
        preview += "..."
    return preview.replace("|", "\\|").replace("[", "(").replace("]", ")")


def pdf_page_link(source_file: str, page: int) -> str:
    """A link straight to the source PDF, jumped to the page the question is
    on. Prefers a Google Drive link (works for any visitor once deployed,
    since a localhost link only resolves on the machine running the app);
    falls back to the local pdf_server.py link when no Drive mapping exists
    (e.g. drive_file_map.json hasn't been generated yet)."""
    drive_id = DRIVE_FILE_MAP.get(_normalize_path(source_file))
    if drive_id:
        return f"https://drive.google.com/file/d/{drive_id}/view#page={page}"
    url_path = quote(source_file.replace("\\", "/"))
    return f"http://localhost:{PDF_SERVER_PORT}/{url_path}#page={page}"


def format_paper_lookup_answer(
    result: dict, query: str, include_table: bool = True, link_builder=pdf_page_link
) -> str:
    """Render occurrences directly from retrieved data as a table - no LLM
    involved. A small free LLM can't be trusted to transcribe exact question
    text without paraphrasing, truncating, or (observed in practice) even
    switching language, so for this intent we skip generation entirely.

    include_table=False returns just the headline sentence - used by the
    Streamlit app, which renders its own interactive version of the table
    (with a PDF viewer that actually jumps to the right page, unlike a plain
    link) instead of this static markdown one.

    link_builder(source_file, page) -> url lets callers swap in a different
    link style - e.g. api_server.py uses the page-image endpoint instead of
    a plain Drive link, since Drive's own viewer doesn't reliably honor
    #page=N on a direct link (confirmed by testing)."""
    occurrences = result["occurrences"]
    year_limit = result.get("year_limit")
    if not occurrences:
        window_note = f" in the last {year_limit} years" if year_limit else ""
        return f"No matching questions were found{window_note} for this query."

    window_note = f" (last {year_limit} years)" if year_limit else ""
    if not include_table:
        return f"Found **{len(occurrences)}** matching question(s){window_note}."

    lines = [
        f"Found **{len(occurrences)}** matching question(s){window_note}:\n",
        "| # | Year | Session (Month) | Paper | Variant | Question # | Question |",
        "|---|------|------------------|-------|---------|------------|----------|",
    ]
    for i, m in enumerate(occurrences, 1):
        session = (m.get("session") or "").replace("_", "/")
        variant = m.get("variant") if m.get("variant") is not None else "-"
        preview = question_preview(m.get("question_text", ""))
        link = link_builder(m.get("source_file", ""), m.get("page", 1))
        lines.append(
            f"| {i} | {m.get('year')} | {session} | {m.get('paper')} | "
            f"{variant} | {m.get('question_number')} | [{preview}]({link}) |"
        )
    return "\n".join(lines)


def generate(query: str, subject: str | None = None, mode: str | None = None,
             include_table: bool = True, link_builder=pdf_page_link):
    """mode, when given, is an explicit UI choice ("ask" or "find") that
    overrides the auto-classified intent entirely - "Find" always does a
    paper lookup even if the query reads like a general question, and "Ask"
    always gives a web+papers explanation even if it reads like a paper
    lookup. mode=None keeps the previous auto-detection behavior (used by
    the CLI, which has no toggle).

    include_table=False (used by the Streamlit app) keeps result["occurrences"]
    populated as usual but leaves it out of the returned answer text, since
    the app renders its own interactive table instead of this static one.

    link_builder is forwarded to format_paper_lookup_answer - see its
    docstring."""
    retriever = Retriever()
    result = retriever.route(query, subject=subject, top_k=10)
    natural_intent = result["intent"]  # before any mode override, for the check below

    if mode == "find":
        result["intent"] = "paper_lookup"
    elif mode == "ask":
        result["intent"] = "general_qa"

    if result["intent"] == "paper_lookup":
        # route() only computes occurrences when ITS OWN classifier already
        # said paper_lookup; if mode="find" forced this path over a
        # general_qa auto-classification, they still need computing here.
        if "occurrences" not in result:
            year_limit = requested_year_limit(query)
            result["year_limit"] = year_limit
            result["occurrences"] = retriever.paper_lookup_summary(
                result["hits"], year_limit=year_limit
            )
        return format_paper_lookup_answer(
            result, query, include_table=include_table, link_builder=link_builder
        ), result

    web_results = web_search(query)
    context = build_context(result, web_results)

    year_limit = requested_year_limit(query)
    worked_examples = select_worked_examples(result["hits"], year_limit)

    system_prompt = (
        "You are a study assistant for Cambridge O/A Level past exam papers. "
        "Answer the user's question using ONLY the provided context (past-paper "
        "excerpts and, if given, web search results). Always respond in English.\n\n"
        "FORMATTING RULES (the chat UI only renders plain text and Markdown "
        "bold/lists/headings - nothing else renders, so violating these makes "
        "the answer look broken):\n"
        "- Never use LaTeX or math markup of any kind: no \\(...\\), \\[...\\], "
        "\\mathrm{}, \\ldots, ^{}, _{}, or any other backslash command. Write "
        "chemical and math formulas in plain text instead, e.g. '6CO2 + 6H2O -> "
        "C6H12O6 + 6O2' (plain digits, no subscript/superscript markup, '->' or "
        "'→' for arrows).\n"
        "- Never truncate quoted question text with '...' or '\\ldots' - either "
        "quote the relevant part in full or paraphrase it cleanly in your own "
        "words; don't leave a dangling ellipsis.\n"
        "- Use short paragraphs and Markdown bullet points ('- ') for lists of "
        "distinct facts or marking points, not one dense run-on paragraph.\n"
        "- Every distinct marking point in an answer, including alternative "
        "('OR ...') marking points, must be its own Markdown bullet line "
        "starting with '- ' - never write alternatives as plain lines of text, "
        "even in a short answer with only 2-3 points.\n\n"
        "Structure your response in two parts, in this order. These 'PART 1' / "
        "'PART 2' labels below are instructions for YOU only - never print the "
        "words 'PART 1' or 'PART 2' in your actual answer, and never print the "
        "explanation under its own heading either; start straight in with the "
        "explanation text itself:\n\n"
        "PART 1 - Explanation: a clear, accurate explanation of the topic grounded "
        "in the provided context, citing which paper(s) informed it "
        "(e.g. 'Accounting 2023 May_June Paper 1 Q16'). If the context doesn't "
        "contain enough information, say so honestly rather than making things up.\n\n"
        "PART 2 - Worked past-paper examples: this part gets exactly one heading, "
        "'### Worked Past-Paper Examples', immediately followed by the answers - "
        "answer EACH question listed in "
        "'Questions to answer' below, one at a time, in the order given - use "
        "the exact question text provided, do not invent or substitute "
        "different questions, and skip this part entirely if no questions are "
        "listed there. For each one: give the question a level-4 Markdown "
        "heading ('#### ') with ONLY its paper reference, no 'Question 1' / "
        "'Question 2' numbering prefix and NOT bold text "
        "(e.g. '#### Biology 2025 Oct/Nov Paper 2 Q4', NOT "
        "'**Question 1 - Biology 2025 Oct/Nov Paper 2 Q4**'), "
        "restate the question text cleanly below it - bold each sub-part label "
        "TOGETHER WITH the instruction/question sentence that directly follows "
        "it on the same line (e.g. '**(a) Explain how the structure of a leaf "
        "is adapted for photosynthesis.**'); any extra data, passages, or "
        "context given below that sub-part's instruction (not the instruction "
        "itself) stays normal weight - then give a full answer in normal "
        "(non-bold) weight, written the way a top-scoring Cambridge "
        "O/A Level candidate would: correct subject terminology, EVERY marking "
        "point as its own '- ' bullet - never write the answer as flowing "
        "prose/sentences, even a single-sentence answer must be a bullet, e.g.:\n\n"
        "**(a) Describe one adverse effect of increased global warming.**\n"
        "- Rising sea levels cause flooding and loss of coastal land\n"
        "- More extreme weather / droughts / desertification\n"
        "- Loss of habitats / extinction of species\n\n"
        "Never write the word 'OR' at the start of a bullet - each bullet "
        "being its own line already shows it's a separate valid alternative, "
        "so a leading 'OR' is redundant clutter; just start straight with the "
        "point itself.\n\n"
        "NOT the wrong way (an alternative marking point is EVERY bit as much "
        "its own bullet as the first point - it never gets tacked onto the "
        "same line, written without its own leading '- ', or prefixed with "
        "'OR'):\n"
        "**(a) Describe one adverse effect of increased global warming.**\n"
        "One adverse effect of increased global warming is the rise in sea "
        "levels, which can lead to... OR more extreme weather / droughts.\n\n"
        "All working shown step-by-step on its own line for calculations, and "
        "for multiple-choice questions the correct option letter followed by a "
        "one-line justification. Match the depth of the answer to the marks "
        "available - don't pad a 1-mark answer or under-explain a 6-mark one. "
        "If a calculation needs a numeric value (e.g. a population figure) "
        "that isn't present anywhere in the given question text or context, "
        "say plainly that the value isn't given in the retrieved excerpt and "
        "state the method/formula that would be used once it's known - never "
        "invent, guess, or assume a placeholder number for it."
    )
    examples_block = (
        f"\n\nQuestions to answer (most relevant, last {year_limit} years):\n"
        f"{format_worked_examples_block(worked_examples)}"
        if worked_examples else ""
    )
    user_prompt = f"Question: {query}\n\nContext:\n{context}{examples_block}"

    if not any(p["api_key"] for p in PROVIDERS):
        print("ERROR: no LLM provider API key set in .env", file=sys.stderr)
        sys.exit(1)

    answer = strip_redundant_or_prefix(call_llm_with_fallback(system_prompt, user_prompt))

    # After answering, also surface which past-paper questions this topic
    # actually shows up in - same table format as a direct paper_lookup,
    # built from the same retrieved hits so it doesn't need a second search.
    # Skipped when Ask mode is answering a query that's naturally phrased as
    # a paper lookup ("which years was X asked"): showing that table would
    # make Ask mode do Find's job on request, exactly what the mode split is
    # meant to prevent. It only appears for genuinely explanatory queries.
    if not (mode == "ask" and natural_intent == "paper_lookup"):
        occurrences = retriever.paper_lookup_summary(result["hits"], year_limit=year_limit)
        result["occurrences"] = occurrences
        result["year_limit"] = year_limit
        if occurrences and include_table:
            related_table = format_paper_lookup_answer(
                {"occurrences": occurrences, "year_limit": year_limit}, query,
                link_builder=link_builder,
            )
            answer = f"{answer}\n\n---\n\n### Related past-paper questions\n\n{related_table}"

    return answer, result


def call_llm_with_fallback(system_prompt: str, user_prompt: str, retries_per_model: int = 2):
    """Try each provider/model in PROVIDERS in order; within one, retry a
    couple of times on 429 (rate limit) with a short backoff before moving
    on to the next. Providers without a configured API key are skipped.

    Any failure (429, 402 payment required, auth error, network error, etc.)
    moves on to the next provider rather than crashing - a model that turns
    out to need billing, or an outage on one provider, must not take down
    the whole chain when other configured providers could still answer."""
    last_error = None
    for provider in PROVIDERS:
        if not provider["api_key"]:
            continue
        for attempt in range(retries_per_model):
            try:
                response = requests.post(
                    provider["url"],
                    headers={"Authorization": f"Bearer {provider['api_key']}"},
                    json={
                        "model": provider["model"],
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                    },
                    timeout=60,
                )
                if response.status_code == 429:
                    last_error = f"{provider['name']}/{provider['model']} rate-limited (429)"
                    if attempt < retries_per_model - 1:
                        time.sleep(3 * (attempt + 1))
                    continue
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            except requests.exceptions.RequestException as e:
                last_error = f"{provider['name']}/{provider['model']} failed: {e}"
                break  # this provider is broken (not just rate-limited) - skip straight to the next
        # exhausted retries (or hit a non-retryable failure) for this provider, move to the next

    raise RuntimeError(
        f"All providers are currently unavailable, please try again shortly. "
        f"Last error: {last_error}"
    )


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "which years was depreciation asked in accounting"
    answer, result = generate(query)
    print(f"Query: {query!r}")
    print(f"Intent: {result['intent']}\n")
    print("Answer:")
    print(answer)


if __name__ == "__main__":
    main()
