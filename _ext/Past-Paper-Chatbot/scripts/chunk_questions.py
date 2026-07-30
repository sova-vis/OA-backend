"""
Step 2 (chunking): Walk data/processed, split each Question Paper (QP) into
individual question-level chunks, and save results to data/chunks/.

Question detection: Cambridge O Level papers number their main questions
sequentially at the start of a line (e.g. "1 Which statement is correct?").
Structured papers also contain nested enumerated sub-lists (e.g. observations
"1 ... 2 ... 3 ..." inside part (c) of a question) that look identical to a
question boundary. To avoid false splits, a candidate boundary is only
accepted if its number equals the expected next question number in sequence
(1, 2, 3, ...) — nested lists restart at 1 mid-document and get rejected
because they don't match the running expectation.

Non-QP documents (MS/ER/GT/IN/SY, topicals, syllabus) are not question-split;
they're stored as a single whole-document chunk each, since they aren't the
target of "which years was this question asked" lookups.

Each question chunk also records which PDF page it starts on, so results can
link directly to the source PDF at the right page.

Run:
    venv\\Scripts\\python.exe scripts\\chunk_questions.py
"""

import bisect
import json
import re
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
CHUNKS_DIR = PROJECT_DIR / "data" / "chunks"
JSONL_OUT = PROJECT_DIR / "data" / "chunks.jsonl"

# Lines that are pure page noise (headers/footers) — stripped before chunking
# so they can't be mistaken for question boundaries or pollute chunk text.
NOISE_LINE_PATTERNS = [
    re.compile(r"^\s*©\s*UCLES\s*\d{4}\s*$", re.IGNORECASE),
    re.compile(r"^\s*\d{4}/\d{2}/[A-Z]/[A-Z]/\d{2}\s*$"),
    re.compile(r"^\s*\[Turn over\]?\s*$", re.IGNORECASE),
    re.compile(r"^\s*PLEASE TURN OVER\s*$", re.IGNORECASE),
    re.compile(r"^\s*This document has \d+ pages.*$", re.IGNORECASE),
]
BARE_PAGE_NUMBER = re.compile(r"^\s*\d+\s*$")

QUESTION_BOUNDARY = re.compile(r"(?m)^\s*(?P<num>\d{1,2})[.)]?\s+(?=[A-Z(])")


def clean_page_text(text: str) -> str:
    lines = text.split("\n")
    kept = []
    for i, ln in enumerate(lines):
        if any(p.match(ln) for p in NOISE_LINE_PATTERNS):
            continue
        # A bare digit is only a page-number footer/header when it's the
        # very first line of the page; mid-page it's a question number.
        if i == 0 and BARE_PAGE_NUMBER.match(ln):
            continue
        kept.append(ln)
    return "\n".join(kept)


def build_full_text_with_page_offsets(cleaned_pages):
    """Join cleaned pages into one text blob, tracking the character offset
    each page starts at, so a later character position can be mapped back to
    the 1-indexed PDF page it came from (used to build page-specific links)."""
    full_text = ""
    page_offsets = []  # page_offsets[i] = start offset of the (i+1)-th page
    for page_text in cleaned_pages:
        page_offsets.append(len(full_text))
        full_text += page_text + "\n"
    return full_text, page_offsets


def offset_to_page(offset: int, page_offsets: list) -> int:
    idx = bisect.bisect_right(page_offsets, offset) - 1
    return max(1, idx + 1)  # 1-indexed page number


def split_into_questions(full_text: str, page_offsets: list):
    matches = list(QUESTION_BOUNDARY.finditer(full_text))
    boundaries = []  # list of (char_offset, question_number, digit_offset)
    expected = 1
    for m in matches:
        num = int(m.group("num"))
        if num == expected:
            # m.start() (the whole match, including leading \s*) is used for
            # slicing chunk text - but NOT for locating the page: \s* can
            # span across the artificial "\n" inserted between pages when a
            # page ends and the next begins with blank lines, so the match
            # start can land one page early even though the digit itself is
            # always physically on the correct page. m.start("num") (the
            # digit's own offset) is used for the page lookup instead.
            boundaries.append((m.start(), num, m.start("num")))
            expected += 1

    if not boundaries:
        return []

    chunks = []
    for i, (offset, qnum, digit_offset) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(full_text)
        text = full_text[offset:end].strip()
        if text:
            page = offset_to_page(digit_offset, page_offsets)
            chunks.append((qnum, text, page))
    return chunks


def main():
    json_files = list(PROCESSED_DIR.rglob("*.json"))
    print(f"Found {len(json_files)} processed files under {PROCESSED_DIR}")

    total_chunks = 0
    no_questions_found = []

    with open(JSONL_OUT, "w", encoding="utf-8") as jsonl_f:
        for jf in json_files:
            data = json.loads(jf.read_text(encoding="utf-8"))
            meta = data["metadata"]
            cleaned_pages = [clean_page_text(p) for p in data["pages"]]
            full_text, page_offsets = build_full_text_with_page_offsets(cleaned_pages)

            rel_out = jf.relative_to(PROCESSED_DIR)
            out_path = CHUNKS_DIR / rel_out
            out_path.parent.mkdir(parents=True, exist_ok=True)

            if meta.get("doc_category") == "exam_paper" and meta.get("doc_type") == "QP":
                question_chunks = split_into_questions(full_text, page_offsets)
                if not question_chunks:
                    no_questions_found.append(str(rel_out))
                records = [
                    {
                        "source_file": data["source_file"],
                        "metadata": {**meta, "question_number": qnum, "page": page},
                        "chunk_text": text,
                    }
                    for qnum, text, page in question_chunks
                ]
            else:
                records = [
                    {
                        "source_file": data["source_file"],
                        "metadata": {**meta, "page": 1},
                        "chunk_text": full_text.strip(),
                    }
                ]

            out_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
            for rec in records:
                jsonl_f.write(json.dumps(rec) + "\n")
            total_chunks += len(records)

    print(f"Total chunks written: {total_chunks}")
    print(f"QP files with zero detected questions: {len(no_questions_found)}")
    for f in no_questions_found[:20]:
        print(f"  - {f}")


if __name__ == "__main__":
    main()
