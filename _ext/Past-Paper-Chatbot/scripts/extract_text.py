"""
Step 1: Walk data/raw, parse metadata from each PDF's filename/path, extract
text from the PDF, and save one JSON per PDF into data/processed/.

Filename patterns seen:
  Subject_Year_Session_Paper_P_Variant_V_TYPE(_N).pdf   (QP, MS, IN)
  Subject_Year_Session_Paper_P_TYPE(_N).pdf             (no variant)
  Subject_Year_Session_TYPE(_N).pdf                     (ER, GT, PDF, SY)
  Subject_NNNN_Topical_...pdf                           (topical study material)
  Subject_NNNN_Syllabus_Reference_...pdf                (syllabus documents)

Files that don't match any of the above (mostly Physics "Topicals" with very
irregular freeform names) fall back to folder-path-based parsing: anything
under a "Topicals" folder is categorized from its folder structure instead.

Run:
    venv\\Scripts\\python.exe scripts\\extract_text.py
"""

import json
import re
from pathlib import Path

import fitz  # PyMuPDF

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
OUT_DIR = PROJECT_DIR / "data" / "processed"

DOC_TYPES = r"QP|MS|IN|ER|GT|PDF|SY|IR"
SUFFIX = r"(?:_\d+)?"  # optional trailing _1, _2 ... for duplicate/resit copies

FULL_PATTERN = re.compile(
    r"^(?P<subject>.+?)_(?P<year>\d{4})_(?P<session>May_June|Oct_Nov)_"
    rf"Paper_(?P<paper>\d+)_Variant_(?P<variant>\d+)_(?P<doc_type>{DOC_TYPES}){SUFFIX}\.pdf$",
    re.IGNORECASE,
)
PAPER_ONLY_PATTERN = re.compile(
    r"^(?P<subject>.+?)_(?P<year>\d{4})_(?P<session>May_June|Oct_Nov)_"
    rf"Paper_(?P<paper>\d+)_(?P<doc_type>{DOC_TYPES}){SUFFIX}\.pdf$",
    re.IGNORECASE,
)
SIMPLE_PATTERN = re.compile(
    r"^(?P<subject>.+?)_(?P<year>\d{4})_(?P<session>May_June|Oct_Nov)_"
    rf"(?P<doc_type>{DOC_TYPES}){SUFFIX}\.pdf$",
    re.IGNORECASE,
)
TOPICAL_PATTERN = re.compile(
    r"^(?P<subject>.+?)_\d+_Topical_(?P<year>\d{4})_(?P<topic>.+?)_O_Level.*\.pdf$",
    re.IGNORECASE,
)
SYLLABUS_PATTERN = re.compile(
    r"^(?P<subject>.+?)_\d+_Syllabus_Reference_.*\.pdf$",
    re.IGNORECASE,
)


def parse_by_path(rel_path: Path):
    """Fallback for files whose names don't match a known pattern but that
    live under a '<Subject>/Topicals/...' folder - derive metadata from the
    folder structure instead (Topicals/<Category>/<Topic>/file.pdf)."""
    parts = rel_path.parts
    if "Topicals" not in parts:
        return None
    subject = parts[0]
    idx = parts.index("Topicals")
    folder_parts = parts[idx + 1 : -1]  # category/topic folders between Topicals/ and the file
    topic = " / ".join(folder_parts) if folder_parts else rel_path.stem
    return {
        "doc_category": "topical",
        "subject": subject,
        "year": None,
        "session": None,
        "paper": None,
        "variant": None,
        "topic": topic,
        "doc_type": "TOPICAL",
    }


def parse_filename(name: str):
    m = FULL_PATTERN.match(name)
    if m:
        d = m.groupdict()
        return {
            "doc_category": "exam_paper",
            "subject": d["subject"].replace("_", " "),
            "year": int(d["year"]),
            "session": d["session"],
            "paper": int(d["paper"]),
            "variant": int(d["variant"]),
            "doc_type": d["doc_type"].upper(),
        }
    m = PAPER_ONLY_PATTERN.match(name)
    if m:
        d = m.groupdict()
        return {
            "doc_category": "exam_paper",
            "subject": d["subject"].replace("_", " "),
            "year": int(d["year"]),
            "session": d["session"],
            "paper": int(d["paper"]),
            "variant": None,
            "doc_type": d["doc_type"].upper(),
        }
    m = SIMPLE_PATTERN.match(name)
    if m:
        d = m.groupdict()
        return {
            "doc_category": "session_doc",
            "subject": d["subject"].replace("_", " "),
            "year": int(d["year"]),
            "session": d["session"],
            "paper": None,
            "variant": None,
            "doc_type": d["doc_type"].upper(),
        }
    m = TOPICAL_PATTERN.match(name)
    if m:
        d = m.groupdict()
        return {
            "doc_category": "topical",
            "subject": d["subject"].replace("_", " "),
            "year": int(d["year"]),
            "session": None,
            "paper": None,
            "variant": None,
            "topic": d["topic"].replace("_", " "),
            "doc_type": "TOPICAL",
        }
    m = SYLLABUS_PATTERN.match(name)
    if m:
        d = m.groupdict()
        return {
            "doc_category": "syllabus",
            "subject": d["subject"].replace("_", " "),
            "year": None,
            "session": None,
            "paper": None,
            "variant": None,
            "doc_type": "SYLLABUS",
        }
    return None


def extract_pdf_text(pdf_path: Path):
    pages = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            pages.append(page.get_text())
    return pages


def main():
    pdf_files = list(RAW_DIR.rglob("*.pdf"))
    print(f"Found {len(pdf_files)} PDFs under {RAW_DIR}")

    ok, unparsed, empty_text = 0, [], []

    for pdf_path in pdf_files:
        rel_path = pdf_path.relative_to(RAW_DIR)
        meta = parse_filename(pdf_path.name) or parse_by_path(rel_path)
        if meta is None:
            unparsed.append(str(rel_path))
            continue

        pages = extract_pdf_text(pdf_path)
        total_chars = sum(len(p) for p in pages)
        if total_chars < 20:
            empty_text.append(str(rel_path))

        record = {
            "source_file": str(rel_path),
            "metadata": meta,
            "num_pages": len(pages),
            "total_chars": total_chars,
            "pages": pages,
        }

        rel_out = rel_path.with_suffix(".json")
        out_path = OUT_DIR / rel_out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        ok += 1

    print(f"Processed OK: {ok}")
    print(f"Filename did not match expected pattern: {len(unparsed)}")
    for f in unparsed[:20]:
        print(f"  - {f}")
    print(f"Likely scanned/image-based (near-empty extracted text): {len(empty_text)}")
    for f in empty_text[:20]:
        print(f"  - {f}")


if __name__ == "__main__":
    main()
