"""Shared content normalization helpers for matching and Mode A confirmation."""

from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional

ContentTypeLabel = str
QuestionFamilyLabel = str


@dataclass(frozen=True)
class MathNormalizationResult:
    original_text: str
    display_text: str
    canonical_text: Optional[str]
    matcher_text: str
    warning_codes: List[str]


@dataclass(frozen=True)
class ContentTextResult:
    original_text: str
    display_text: str
    canonical_text: Optional[str]
    matcher_text: str
    warning_codes: List[str]
    content_type: ContentTypeLabel


_MOJIBAKE_MAP = {
    "\u00c2\u00b2": "\u00b2",
    "\u00c2\u00b3": "\u00b3",
}
_SUBSCRIPT_DIGITS = {
    "\u2080": "0",
    "\u2081": "1",
    "\u2082": "2",
    "\u2083": "3",
    "\u2084": "4",
    "\u2085": "5",
    "\u2086": "6",
    "\u2087": "7",
    "\u2088": "8",
    "\u2089": "9",
}
_SUPERSCRIPT_DIGITS = {
    "\u2070": "0",
    "\u00b9": "1",
    "\u00b2": "2",
    "\u00b3": "3",
    "\u2074": "4",
    "\u2075": "5",
    "\u2076": "6",
    "\u2077": "7",
    "\u2078": "8",
    "\u2079": "9",
}
_UNICODE_DIGIT_FOR_MATCH: dict[str, str] = {**_SUBSCRIPT_DIGITS, **_SUPERSCRIPT_DIGITS}
# Unicode hyphens/minus signs → ASCII hyphen-minus (matcher / JSON alignment).
_UNICODE_HYPHEN_MINUS_CHARS: frozenset[str] = frozenset(
    "\u2010\u2011\u2012\u2013\u2014\u2015\u2212\uFE58\uFE63\uFF0D"
)
# NBSP and similar → normal space for tokenization.
_UNICODE_SPACE_LIKE_CHARS: frozenset[str] = frozenset(
    "\u00A0\u202F\u2007\u2009\u200A\u2060\uFEFF"
)
_DISPLAY_LOG_BASE_RE = re.compile(r"\blog_(\d+)\b", re.IGNORECASE)
_DISPLAY_LOG_INLINE_RE = re.compile(r"log_(\d+)(?=([A-Za-z(]))", re.IGNORECASE)
_ASCII_BASE_COLLISION_RE = re.compile(r"\blog_(\d+)(?=(\d))", re.IGNORECASE)
_UNICODE_BASE_COLLISION_RE = re.compile(r"log[\u2080-\u2089]+(?=\d)", re.IGNORECASE)
_CANONICAL_LOG_INPUT_RE = re.compile(
    r"log\s*\(\s*base\s*=\s*([^,()]+?)\s*,\s*argz?\s*=\s*([^)]+?)\s*\)",
    re.IGNORECASE,
)
_CANONICAL_LOG_EXTRACT_RE = re.compile(r"log\(base=([^,()]+),arg=([^)]+)\)", re.IGNORECASE)
_EXPLICIT_LOG_PAREN_RE = re.compile(r"\blog_([A-Za-z0-9]+)\s*\(\s*([^)]+?)\s*\)", re.IGNORECASE)
_EXPLICIT_LOG_TOKEN_RE = re.compile(r"\blog_([A-Za-z0-9]+)\s+([A-Za-z0-9]+(?:\^[0-9]+)?)\b", re.IGNORECASE)
_AMBIGUOUS_INLINE_LOG_RE = re.compile(r"\blog\d+[A-Za-z(]", re.IGNORECASE)
_AMBIGUOUS_SPACED_LOG_RE = re.compile(r"\blog\s+\d+[A-Za-z(]", re.IGNORECASE)
_MATCHER_POW_RE = re.compile(r"\^(\d+)")
_MATCHER_LOG_BASE_RE = re.compile(r"\blog_(\d+)\b", re.IGNORECASE)
_MATCHER_SYMBOL_RE = re.compile(r"[^A-Za-z0-9\s]+")
_MATCHER_SPACE_RE = re.compile(r"\s+")
_QUESTION_LABEL_RE = re.compile(r"^\s*(?:q(?:uestion)?\.?\s*)", re.IGNORECASE)
_ANSWER_LABEL_RE = re.compile(r"^\s*(?:ans(?:wer)?\.?\s*[:\-]?|a\.?\s*)", re.IGNORECASE)
_MATH_SUBJECTS = {"mathematics 1014"}

_MCQ_RE = re.compile(r"\b(?:option|choose|which of the following|[ABCD][.)])\b", re.IGNORECASE)
_SYMBOLIC_RE = re.compile(r"(?:=|\^|\bsqrt\b|\blog\b|\bsin\b|\bcos\b|\btan\b|\bln\b|[+\-*/()])", re.IGNORECASE)
_SCIENCE_TOKEN_RE = re.compile(r"\b(?:[A-Z][a-z]?\d+|[A-Z][a-z]?[+-]|\d+(?:cm|mm|km|kg|g|mol|m/s|ms|s|N|J|W|Pa|V|A)\b)", re.IGNORECASE)
_PROSE_WORD_RE = re.compile(r"[A-Za-z]{4,}")
_GENERIC_MATCHER_RE = re.compile(r"[^A-Za-z0-9\s]+")
_SPACE_RE = re.compile(r"\s+")
_FUNCTION_EXPR_RE = re.compile(r"\b(?:f|g|h|fg|gf)\s*\(", re.IGNORECASE)
_LOG_FAMILY_RE = re.compile(r"(?<![A-Za-z])(?:log|ln)", re.IGNORECASE)
# Canonical log(base=10,arg=…) in matcher strings → spoken-style log base 10 <arg> (Mathematics only).
_MATCHER_CANONICAL_LOG_ANY_BASE_RE = re.compile(
    r"log\s*\(\s*base\s*=\s*([^,()]+?)\s*,\s*argz?\s*=\s*([^)]+?)\s*\)",
    re.IGNORECASE,
)


def _env_matcher_fold_bare_log_default() -> bool:
    raw = (os.getenv("OA_MATCHER_FOLD_BARE_LOG_TO_BASE10") or "").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    if raw in {"1", "true", "yes", "on"}:
        return True
    return True


def _apply_mathematics_common_log_matcher_fold(
    matcher_text: str,
    *,
    fold_bare_log: bool | None = None,
) -> str:
    """Mathematics matcher only: align lg, bare log (optional), and log(base=10,…) with log base 10 … tokens.

    Does not alter ln. Does not map log_a / log_3 / non-10 bases to base 10. Idempotent on log base 10.
    """
    if fold_bare_log is None:
        fold_bare_log = _env_matcher_fold_bare_log_default()

    matcher = str(matcher_text or "")
    if not matcher:
        return matcher

    def _canonical_base10_sub(match: re.Match[str]) -> str:
        base = re.sub(r"\s+", "", match.group(1))
        if base != "10":
            return match.group(0)
        arg = _compact_math_fragment(match.group(2))
        return f"log base 10 {arg}" if arg else "log base 10"

    matcher = _MATCHER_CANONICAL_LOG_ANY_BASE_RE.sub(_canonical_base10_sub, matcher)
    matcher = re.sub(r"(?<![A-Za-z])lg\b", "log base 10", matcher, flags=re.IGNORECASE)

    if fold_bare_log:
        matcher = re.sub(
            r"(?<![A-Za-z])log(?!\s+base\b)(?!_)(?=\s|\(|$)(?!\s*\(\s*base\s*=)",
            "log base 10",
            matcher,
            flags=re.IGNORECASE,
        )

    return _MATCHER_SPACE_RE.sub(" ", matcher).strip()


def _coalesce_spaces(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def fold_unicode_numeric_forms(text: str) -> str:
    """Map Unicode subscript/superscript digits to ASCII for token matching (e.g. H₂O vs H2O, cm³ vs cm3)."""
    if not text:
        return ""
    return "".join(_UNICODE_DIGIT_FOR_MATCH.get(ch, ch) for ch in text)


def _fold_hyphens_and_unicode_spaces(text: str) -> str:
    """Map Unicode hyphens/minus and space-like characters to ASCII (safe for math input pretreatment)."""
    if not text:
        return ""
    out: List[str] = []
    for ch in text:
        if ch in _UNICODE_HYPHEN_MINUS_CHARS:
            out.append("-")
        elif ch in _UNICODE_SPACE_LIKE_CHARS:
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


def fold_plaintext_science_symbols(text: str) -> str:
    """Map common Unicode science/operators to ASCII for matching OCR to JSON (idempotent on ASCII).

    Middle dot (·) → period (hydrate dot in chemistry). May rarely affect prose; v1 tradeoff.
    """
    if not text:
        return ""
    out: List[str] = []
    for ch in text:
        if ch in _UNICODE_HYPHEN_MINUS_CHARS:
            out.append("-")
        elif ch in _UNICODE_SPACE_LIKE_CHARS:
            out.append(" ")
        elif ch == "\u00d7":  # ×
            out.append("*")
        elif ch == "\u00f7":  # ÷
            out.append("/")
        elif ch == "\u2192":  # →
            out.extend(("-", ">",))
        elif ch == "\u00b7":  # ·
            out.append(".")
        else:
            out.append(ch)
    return "".join(out)


def fold_math_matcher_input(text: str) -> str:
    """Normalize hyphens/spaces before math visual pipeline (does not map ×÷→· to protect log/superscript logic)."""
    return _fold_hyphens_and_unicode_spaces(_coalesce_spaces(text or ""))


def _replace_common_mojibake(text: str) -> str:
    out = str(text or "")
    if any(token in out for token in ("\u00e2", "\u00c2", "\u00c3")):
        try:
            repaired = out.encode("cp1252").decode("utf-8")
            old_score = sum(out.count(token) for token in ("\u00e2", "\u00c2", "\u00c3"))
            new_score = sum(repaired.count(token) for token in ("\u00e2", "\u00c2", "\u00c3"))
            if new_score < old_score:
                out = repaired
        except Exception:
            pass
    for bad, good in _MOJIBAKE_MAP.items():
        out = out.replace(bad, good)
    return out


def is_mathematics_subject(subject: Optional[str]) -> bool:
    return _coalesce_spaces(subject or "").casefold() in _MATH_SUBJECTS


def strip_leading_question_label(text: str) -> str:
    cleaned = _coalesce_spaces(text or "")
    return _QUESTION_LABEL_RE.sub("", cleaned, count=1).strip()


def strip_leading_answer_label(text: str) -> str:
    cleaned = _coalesce_spaces(text or "")
    return _ANSWER_LABEL_RE.sub("", cleaned, count=1).strip()


def _compact_math_fragment(text: str) -> str:
    return re.sub(r"\s+", "", _coalesce_spaces(text or ""))


def _format_canonical_log(base: str, arg: str) -> str:
    return f"log(base={_compact_math_fragment(base)},arg={_compact_math_fragment(arg)})"


def _normalize_visual_math_text(text: str) -> str:
    source = _replace_common_mojibake(text)
    out: List[str] = []
    index = 0
    while index < len(source):
        if source[index : index + 3].lower() == "log":
            out.append(source[index : index + 3])
            index += 3
            base_digits: List[str] = []
            while index < len(source) and source[index] in _SUBSCRIPT_DIGITS:
                base_digits.append(_SUBSCRIPT_DIGITS[source[index]])
                index += 1
            if base_digits:
                out.append("_")
                out.append("".join(base_digits))
                if index < len(source) and source[index] not in " \t\r\n+-=*/^)],":
                    out.append(" ")
                continue
        char = source[index]
        if char in _SUPERSCRIPT_DIGITS:
            digits: List[str] = []
            while index < len(source) and source[index] in _SUPERSCRIPT_DIGITS:
                digits.append(_SUPERSCRIPT_DIGITS[source[index]])
                index += 1
            out.append("^")
            out.append("".join(digits))
            continue
        out.append(char)
        index += 1

    display = "".join(out)
    display = _DISPLAY_LOG_INLINE_RE.sub(lambda m: f"log_{m.group(1)} ", display)
    display = re.sub(r"\s*([=+\-*/()])\s*", r" \1 ", display)
    display = re.sub(r"\s+\^", "^", display)
    display = re.sub(r"\(\s+", "(", display)
    display = re.sub(r"\s+\)", ")", display)
    return _coalesce_spaces(display)


def _normalize_canonical_math_text(text: str) -> Optional[str]:
    raw = _coalesce_spaces(_replace_common_mojibake(text or ""))
    if not raw:
        return None
    normalized = raw.replace("argz", "arg")
    pieces: List[str] = []
    last_end = 0
    found = False
    for match in _CANONICAL_LOG_INPUT_RE.finditer(normalized):
        pieces.append(normalized[last_end:match.start()])
        base = _coalesce_spaces(match.group(1))
        arg = _coalesce_spaces(match.group(2))
        if not base or not arg:
            return None
        pieces.append(_format_canonical_log(base, arg))
        last_end = match.end()
        found = True
    pieces.append(normalized[last_end:])
    out = _coalesce_spaces("".join(pieces))
    if "log" in normalized.casefold() and not found:
        return None
    return out


def _canonicalize_explicit_log_forms(display_text: str) -> Optional[str]:
    display = str(display_text or "")
    if not display:
        return None
    transformed = display
    transformed = _EXPLICIT_LOG_PAREN_RE.sub(
        lambda match: _format_canonical_log(match.group(1), match.group(2)),
        transformed,
    )
    transformed = _EXPLICIT_LOG_TOKEN_RE.sub(
        lambda match: _format_canonical_log(match.group(1), match.group(2)),
        transformed,
    )
    transformed = _coalesce_spaces(transformed)
    return transformed if transformed != display else None


def _has_ambiguous_log_notation(original_text: str, display_text: str) -> bool:
    original = _replace_common_mojibake(original_text or "")
    display = str(display_text or "")
    return bool(
        _AMBIGUOUS_INLINE_LOG_RE.search(original)
        or _AMBIGUOUS_SPACED_LOG_RE.search(original)
        or _AMBIGUOUS_INLINE_LOG_RE.search(display)
        or _AMBIGUOUS_SPACED_LOG_RE.search(display)
    )


def _build_matcher_text(display_text: str) -> str:
    matcher = str(display_text or "")
    matcher = _MATCHER_LOG_BASE_RE.sub(lambda m: f"log base {m.group(1)}", matcher)
    matcher = _MATCHER_POW_RE.sub(lambda m: f" power {m.group(1)}", matcher)
    matcher = matcher.replace("\u2248", " approx ")
    matcher = matcher.replace("\u2264", " le ")
    matcher = matcher.replace("\u2265", " ge ")
    matcher = matcher.replace("\u221a", " sqrt ")
    matcher = matcher.replace("\u00d7", " times ")
    matcher = matcher.replace("\u00f7", " divide ")
    matcher = _MATCHER_SYMBOL_RE.sub(" ", matcher)
    return _MATCHER_SPACE_RE.sub(" ", matcher).strip()


def _normalization_warning_codes(
    original_text: str,
    display_text: str,
    *,
    canonical_text: Optional[str] = None,
    canonical_requested: bool = False,
) -> List[str]:
    warnings: set[str] = set()
    original = str(original_text or "")
    display = str(display_text or "")

    if "\ufffd" in original or "\u00e2\u201a" in original or any(
        ch in original for ch in ("\u208d", "\u208e", "\u208a", "\u208b", "\u208c")
    ):
        warnings.add("subscript_decoding_uncertain")
    if "\ufffd" in original or "\u00c2\u00b2" in original or "\u00c2\u00b3" in original or "\u00e2\u0081" in original:
        warnings.add("superscript_decoding_uncertain")
    if _UNICODE_BASE_COLLISION_RE.search(_replace_common_mojibake(original)):
        warnings.add("merged_log_base_operand")
    if _ASCII_BASE_COLLISION_RE.search(display):
        warnings.add("merged_log_base_operand")
    if re.search(r"\blog_\d{2,}(?=[A-Za-z])", display, flags=re.IGNORECASE):
        warnings.add("ambiguous_log_base_attachment")
    if _has_ambiguous_log_notation(original, display) and not canonical_text:
        warnings.add("ambiguous_log_base_argument")
    if canonical_requested and "log" in display.casefold() and canonical_text is None:
        warnings.add("canonical_math_missing")
    return sorted(warnings)


def normalize_math_text_result(
    text: str,
    *,
    canonical_text: Optional[str] = None,
    enable_canonical: bool = False,
) -> MathNormalizationResult:
    original = str(text or "")
    display = _normalize_visual_math_text(original)
    normalized_canonical: Optional[str] = None
    if enable_canonical:
        normalized_canonical = _normalize_canonical_math_text(canonical_text) if canonical_text else None
        if normalized_canonical is None:
            normalized_canonical = _normalize_canonical_math_text(original)
        if normalized_canonical is None:
            normalized_canonical = _canonicalize_explicit_log_forms(display)
    return MathNormalizationResult(
        original_text=original,
        display_text=display,
        canonical_text=normalized_canonical,
        matcher_text=normalized_canonical or _build_matcher_text(display),
        warning_codes=_normalization_warning_codes(
            original,
            display,
            canonical_text=normalized_canonical,
            canonical_requested=enable_canonical,
        ),
    )


def build_subject_matcher_text(
    text: str,
    *,
    subject: Optional[str],
    canonical_text: Optional[str] = None,
) -> str:
    if is_mathematics_subject(subject):
        pretreated = fold_math_matcher_input(text)
        matcher = normalize_math_text_result(
            pretreated,
            canonical_text=canonical_text,
            enable_canonical=True,
        ).matcher_text
        return _apply_mathematics_common_log_matcher_fold(matcher)
    raw = _coalesce_spaces(text or "")
    raw = fold_unicode_numeric_forms(raw)
    raw = fold_plaintext_science_symbols(raw)
    return raw


def _build_generic_matcher(display_text: str) -> str:
    cleaned = _GENERIC_MATCHER_RE.sub(" ", str(display_text or ""))
    return _SPACE_RE.sub(" ", cleaned).strip()


def _normalize_prose_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    normalized = normalized.replace("\u2018", "'").replace("\u2019", "'")
    normalized = normalized.replace("\u201c", '"').replace("\u201d", '"')
    normalized = re.sub(r"\s+([,.;:?!])", r"\1", normalized)
    return _coalesce_spaces(normalized)


def _science_warning_codes(original: str, display: str) -> List[str]:
    warnings: set[str] = set()
    if re.search(r"\b(?:Co\d|No\d|So\d)\b", display):
        warnings.add("ambiguous_science_formula_case")
    if re.search(r"\bms\b", display):
        warnings.add("ambiguous_unit_boundary")
    if re.search(r"[A-Z][a-z]?\d+[A-Za-z]", display):
        warnings.add("merged_science_token")
    if "CO2" not in display and re.search(r"\bCo2\b", display):
        warnings.add("ambiguous_science_formula_case")
    if "\ufffd" in original:
        warnings.add("unicode_decoding_uncertain")
    return sorted(warnings)


def classify_content_type(question_text: str, student_answer: str, subject: Optional[str] = None) -> ContentTypeLabel:
    combined = f"{question_text or ''}\n{student_answer or ''}".strip()
    lowered = combined.lower()
    subject_text = str(subject or "").strip().lower()
    if _MCQ_RE.search(combined):
        return "mcq_structured"
    if _SCIENCE_TOKEN_RE.search(combined) and any(token in subject_text for token in ("chemistry", "physics", "biology")):
        return "science_notation"
    if _SYMBOLIC_RE.search(combined):
        if _SCIENCE_TOKEN_RE.search(combined) and not any(
            token in lowered for token in ("log", "sqrt", "sin", "cos", "tan")
        ):
            return "science_notation"
        return "symbolic"
    if _SCIENCE_TOKEN_RE.search(combined):
        return "science_notation"
    prose_words = len(_PROSE_WORD_RE.findall(combined))
    if prose_words >= 6 or len(combined.split()) >= 8:
        return "prose"
    return "mcq_structured" if re.search(r"\b[ABCD]\b", combined) else "prose"


def normalize_content_text_result(
    text: str,
    *,
    content_type: ContentTypeLabel,
    subject: Optional[str] = None,
    canonical_text: Optional[str] = None,
) -> ContentTextResult:
    original = str(text or "")
    if content_type in {"symbolic", "science_notation"}:
        math_result = normalize_math_text_result(
            original,
            canonical_text=canonical_text,
            enable_canonical=is_mathematics_subject(subject),
        )
        warnings = list(math_result.warning_codes)
        if content_type == "science_notation":
            warnings = sorted(set(warnings) | set(_science_warning_codes(original, math_result.display_text)))
        matcher_out = math_result.matcher_text
        if is_mathematics_subject(subject):
            matcher_out = _apply_mathematics_common_log_matcher_fold(matcher_out)
        return ContentTextResult(
            original_text=original,
            display_text=math_result.display_text,
            canonical_text=math_result.canonical_text,
            matcher_text=matcher_out,
            warning_codes=warnings,
            content_type=content_type,
        )

    display = _normalize_prose_text(original)
    warnings: List[str] = []
    if not display:
        warnings.append("empty_text")
    return ContentTextResult(
        original_text=original,
        display_text=display,
        canonical_text=None,
        matcher_text=_build_generic_matcher(display).casefold(),
        warning_codes=warnings,
        content_type=content_type,
    )


def classify_question_family(text: str, *, content_type: ContentTypeLabel) -> QuestionFamilyLabel:
    display = str(text or "").strip()
    lowered = display.casefold()
    if content_type == "science_notation":
        return "science_formula_units"
    if content_type in {"prose", "mcq_structured"}:
        return "prose_or_structured"
    if _LOG_FAMILY_RE.search(display):
        return "logarithmic"
    if _FUNCTION_EXPR_RE.search(display):
        return "function_expression"
    if re.search(r"[=+\-*/^]", display):
        return "generic_equation"
    if _SCIENCE_TOKEN_RE.search(display):
        return "science_formula_units"
    return "generic_equation" if any(token in lowered for token in ("x", "y", "z")) else "prose_or_structured"
