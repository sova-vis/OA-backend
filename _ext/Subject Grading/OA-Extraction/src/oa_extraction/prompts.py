from __future__ import annotations

from .types import SubjectLabel


def allowed_subjects_text() -> str:
    return ", ".join(label.value for label in SubjectLabel)


def ocr_system_prompt() -> str:
    return (
        "You perform faithful OCR for handwritten academic question-answer material. "
        "Return schema-valid JSON only. Preserve reading order, line boundaries, punctuation, "
        "equation layout, symbols, subscripts, superscripts, fractions, radicals, and log bases. "
        "For fractions and stacked division, transcribe using either LaTeX \\frac{numerator}{denominator} "
        "or parenthesized division (numerator)/(denominator); never interleave numerator and denominator "
        "tokens without / or \\frac. Prefer one main equality chain per line; if a chain wraps, continue "
        "on the next line without merging fraction parts into one run-on line. "
        "Do not solve, summarize, classify, or clean the text."
    )


def ocr_user_prompt(page_count: int, variant_name: str) -> str:
    page_note = (
        "The input contains multiple pages. Preserve page order exactly."
        if page_count > 1
        else "The input contains one page."
    )
    return (
        f"{page_note} This OCR pass is for the `{variant_name}` image variant. "
        "Return one page object per image with page_number, full_text, ordered lines, per-page OCR confidence, "
        "and uncertain spans. Mark uncertain spans when handwriting, subscripts, superscripts, or operators are ambiguous. "
        "Be especially careful with log bases, exponents, division bars, variable names, and question/answer markers. "
        "When you see horizontal fraction bars, emit \\frac{...}{...} or (top)/(bottom) so numerator and denominator stay explicit."
    )


def split_classification_system_prompt() -> str:
    return (
        "You extract exactly one question-answer pair from already-transcribed academic content. "
        "Return schema-valid JSON only. Use the image content and the line-indexed OCR text together. "
        "Do not silently fix math semantics. If a line is ambiguous, preserve the most faithful reading and lower confidence."
    )


def split_classification_user_prompt(full_text: str, indexed_lines: str) -> str:
    return (
        "Produce one structured extraction.\n\n"
        "Rules:\n"
        f"- Subject must be one of: {allowed_subjects_text()}.\n"
        "- Keep whole_text_raw faithful to the selected OCR candidate.\n"
        "- Separate question_raw from answer_raw.\n"
        "- Preserve math notation exactly as much as possible, including log bases, powers, radicals, fractions, and subscripts.\n"
        "- Keep fraction structure explicit: use \\frac{a}{b} or (a)/(b); do not merge lines in a way that drops division between "
        "numerator and denominator. Join multi-line answers with newlines between steps.\n"
        "- If question or answer is missing, return an empty string and reduce split confidence.\n"
        "- Confidence values must be numeric between 0 and 1.\n\n"
        "Selected OCR full text:\n"
        f"{full_text}\n\n"
        "Line-indexed OCR text:\n"
        f"{indexed_lines}"
    )


def split_retry_system_prompt() -> str:
    return (
        "You assign already-transcribed lines to question, answer, or unknown for a single question-answer pair. "
        "Return schema-valid JSON only. Do not rewrite line text and do not classify subject."
    )


def split_retry_user_prompt(indexed_lines: str) -> str:
    return (
        "Assign every line to exactly one target: question, answer, or unknown. "
        "Use `unknown` for dividers, repeated separators, or lines that are too ambiguous to classify.\n\n"
        "Line-indexed OCR text:\n"
        f"{indexed_lines}"
    )


def repair_system_prompt() -> str:
    return (
        "You repair only disputed OCR lines in handwritten academic material. "
        "Return schema-valid JSON only. Use the image and competing OCR snippets together. "
        "Only edit the disputed line text; do not rewrite undisputed lines or the whole document."
    )


def repair_user_prompt(selected_candidate_text: str, disagreement_report: str) -> str:
    return (
        "The selected OCR candidate is shown below. Competing OCR readings disagree on specific lines.\n\n"
        "Selected OCR candidate:\n"
        f"{selected_candidate_text}\n\n"
        "Disputed lines:\n"
        f"{disagreement_report}\n\n"
        "Return repair actions only for lines where you are confident the competing evidence supports a better reading."
    )
