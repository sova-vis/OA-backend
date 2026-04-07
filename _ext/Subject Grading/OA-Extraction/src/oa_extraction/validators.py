from __future__ import annotations

import re

from .config import Settings
from .types import ConfidenceScores, ExtractionDiagnostics, FlagSeverity, SubjectLabel, ValidationFlag


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    lines = [line.rstrip() for line in text.split("\n")]
    normalized = "\n".join(lines).strip()
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized


def build_confidence(ocr: float, split: float, classification: float) -> ConfidenceScores:
    overall = min(ocr, split, classification)
    return ConfidenceScores(
        ocr=round(ocr, 4),
        split=round(split, 4),
        classification=round(classification, 4),
        overall=round(overall, 4),
    )


def validate_extraction(
    *,
    whole_text_raw: str,
    question_raw: str,
    answer_raw: str,
    subject: SubjectLabel,
    confidence: ConfidenceScores,
    settings: Settings,
) -> list[ValidationFlag]:
    flags: list[ValidationFlag] = []

    question_normalized = normalize_text(question_raw)
    answer_normalized = normalize_text(answer_raw)
    whole_normalized = normalize_text(whole_text_raw)

    if not question_normalized:
        flags.append(_flag("missing_question", FlagSeverity.ERROR, "Question text could not be extracted."))

    if not answer_normalized:
        flags.append(_flag("missing_answer", FlagSeverity.ERROR, "Answer text could not be extracted."))

    if _has_excessive_overlap(question_normalized, answer_normalized):
        flags.append(
            _flag(
                "question_answer_overlap",
                FlagSeverity.ERROR,
                "Question and answer look duplicated or insufficiently separated.",
            )
        )

    if confidence.ocr < settings.ocr_confidence_threshold:
        flags.append(
            _flag(
                "low_ocr_confidence",
                FlagSeverity.WARNING,
                f"OCR confidence {confidence.ocr:.2f} is below threshold {settings.ocr_confidence_threshold:.2f}.",
            )
        )

    if confidence.split < settings.split_confidence_threshold:
        flags.append(
            _flag(
                "low_split_confidence",
                FlagSeverity.WARNING,
                f"Split confidence {confidence.split:.2f} is below threshold {settings.split_confidence_threshold:.2f}.",
            )
        )

    if confidence.classification < settings.classification_confidence_threshold:
        flags.append(
            _flag(
                "low_classification_confidence",
                FlagSeverity.WARNING,
                (
                    f"Classification confidence {confidence.classification:.2f} is below threshold "
                    f"{settings.classification_confidence_threshold:.2f}."
                ),
            )
        )

    if _should_run_math_checks(subject, whole_normalized, question_normalized, answer_normalized):
        flags.extend(_math_flags(whole_normalized, question_normalized, answer_normalized))

    flags.extend(_subject_specific_flags(subject, whole_normalized))
    return _dedupe_flags(flags)


def validate_diagnostics(diagnostics: ExtractionDiagnostics, settings: Settings) -> list[ValidationFlag]:
    flags: list[ValidationFlag] = []
    if diagnostics.disagreement_spans and any(
        span.disagreement_score >= settings.engine_disagreement_threshold for span in diagnostics.disagreement_spans
    ):
        flags.append(
            _flag(
                "ocr_engine_disagreement",
                FlagSeverity.WARNING,
                "OCR candidates disagree materially on one or more lines.",
            )
        )

    if diagnostics.disagreement_spans and any(span.unresolved for span in diagnostics.disagreement_spans):
        flags.append(
            _flag(
                "unresolved_ocr_conflicts",
                FlagSeverity.WARNING,
                "Some disputed OCR lines remain unresolved after arbitration.",
            )
        )

    if diagnostics.repair_actions and any(action.accepted for action in diagnostics.repair_actions):
        flags.append(
            _flag(
                "targeted_repair_applied",
                FlagSeverity.WARNING,
                "Targeted OCR repair changed one or more disputed lines.",
            )
        )

    return _dedupe_flags(flags)


def needs_review(
    flags: list[ValidationFlag],
    confidence: ConfidenceScores,
    settings: Settings,
    diagnostics: ExtractionDiagnostics | None = None,
) -> bool:
    if any(flag.severity == FlagSeverity.ERROR for flag in flags):
        return True
    if confidence.ocr < settings.ocr_confidence_threshold:
        return True
    if confidence.split < settings.split_confidence_threshold:
        return True
    if confidence.classification < settings.classification_confidence_threshold:
        return True
    if any(flag.code.startswith("potential_") for flag in flags):
        return True
    if diagnostics:
        if any(
            span.unresolved and span.disagreement_score >= settings.engine_disagreement_threshold
            for span in diagnostics.disagreement_spans
        ):
            return True
        if any(action.accepted for action in diagnostics.repair_actions) and (
            confidence.ocr < settings.grok_fallback_ocr_threshold
            or confidence.split < settings.grok_fallback_split_threshold
        ):
            return True
    return False


def _flag(code: str, severity: FlagSeverity, message: str) -> ValidationFlag:
    return ValidationFlag(code=code, severity=severity, message=message)


def _has_excessive_overlap(question_text: str, answer_text: str) -> bool:
    if not question_text or not answer_text:
        return False

    q = _canonical(question_text)
    a = _canonical(answer_text)
    if not q or not a:
        return False

    if q == a:
        return True
    if len(q) >= 20 and q in a:
        return True
    if len(a) >= 20 and a in q:
        return True
    return False


def _canonical(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def _should_run_math_checks(
    subject: SubjectLabel,
    whole_text_raw: str,
    question_raw: str,
    answer_raw: str,
) -> bool:
    if subject == SubjectLabel.MATH:
        return True
    combined = " ".join(part for part in (whole_text_raw, question_raw, answer_raw) if part)
    return bool(re.search(r"(log|=|\^|sqrt|sin|cos|tan|integral|fraction|\/)", combined, re.IGNORECASE))


def _math_flags(
    whole_text_raw: str,
    question_raw: str,
    answer_raw: str,
) -> list[ValidationFlag]:
    flags: list[ValidationFlag] = []

    question_bases = _extract_log_bases(question_raw)
    answer_bases = _extract_log_bases(answer_raw)
    if question_bases and answer_bases and question_bases.isdisjoint(answer_bases):
        flags.append(
            _flag(
                "potential_log_base_mismatch",
                FlagSeverity.WARNING,
                "Detected different logarithm bases between question and answer.",
            )
        )

    if re.search(r"\blog[0-9A-Za-z]", whole_text_raw):
        flags.append(
            _flag(
                "potential_subscript_loss",
                FlagSeverity.WARNING,
                "Detected compact log notation that may hide a lost or ambiguous base subscript.",
            )
        )

    if re.search(r"\b[a-zA-Z][0-9]\b", whole_text_raw):
        flags.append(
            _flag(
                "potential_superscript_loss",
                FlagSeverity.WARNING,
                "Detected compact variable-digit tokens that may indicate lost superscript formatting.",
            )
        )

    if _has_unbalanced_structure(whole_text_raw):
        flags.append(
            _flag(
                "potential_equation_structure_loss",
                FlagSeverity.WARNING,
                "Detected unbalanced mathematical grouping symbols that may indicate OCR structure loss.",
            )
        )

    if _has_symbol_ambiguity(whole_text_raw):
        flags.append(
            _flag(
                "potential_symbol_ambiguity",
                FlagSeverity.WARNING,
                "Detected visually confusable math symbols such as 0/O or 1/I/l.",
            )
        )

    if _has_potential_fraction_or_chain_ambiguity(answer_raw):
        flags.append(
            _flag(
                "potential_fraction_or_chain_ambiguity",
                FlagSeverity.WARNING,
                "Detected a long log-heavy equality chain on one line without explicit division or \\frac; "
                "OCR may have interleaved numerator and denominator.",
            )
        )

    return flags


def _subject_specific_flags(subject: SubjectLabel, whole_text_raw: str) -> list[ValidationFlag]:
    flags: list[ValidationFlag] = []
    if subject == SubjectLabel.CHEMISTRY:
        if re.search(r"\b[A-Z][a-z]?[0O][A-Za-z0-9()]*\b", whole_text_raw):
            flags.append(
                _flag(
                    "potential_chemistry_formula_ambiguity",
                    FlagSeverity.WARNING,
                    "Detected chemistry formula tokens where 0/O ambiguity may affect the formula.",
                )
            )

    if subject == SubjectLabel.PHYSICS:
        if re.search(r"\b\d+\s*(hz|pa|j|v|w|n|c)\b", whole_text_raw):
            flags.append(
                _flag(
                    "potential_physics_unit_ambiguity",
                    FlagSeverity.WARNING,
                    "Detected SI unit tokens with casing that may be OCR-degraded.",
                )
            )

    if subject == SubjectLabel.BIOLOGY:
        if re.search(r"\b(dna|rna|atp)\b", whole_text_raw):
            flags.append(
                _flag(
                    "potential_biology_term_casing",
                    FlagSeverity.WARNING,
                    "Detected biology acronyms with casing that may be OCR-degraded.",
                )
            )

    if subject == SubjectLabel.ENGLISH:
        if whole_text_raw.count('"') % 2 != 0 or whole_text_raw.count("'") % 2 != 0:
            flags.append(
                _flag(
                    "potential_english_punctuation_loss",
                    FlagSeverity.WARNING,
                    "Detected unbalanced quotes or apostrophes.",
                )
            )

    return flags


def _extract_log_bases(text: str) -> set[str]:
    patterns = [
        r"log\s*_\s*([A-Za-z0-9]+)",
        r"\blog([A-Za-z0-9])(?=[A-Za-z(])",
        r"\blog([0-9]+)\b",
    ]
    matches: set[str] = set()
    for pattern in patterns:
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            matches.add(match.lower())
    return matches


def _has_unbalanced_structure(text: str) -> bool:
    pairs = [("(", ")"), ("[", "]"), ("{", "}")]
    for opener, closer in pairs:
        if text.count(opener) != text.count(closer):
            return True
    return False


def _has_symbol_ambiguity(text: str) -> bool:
    has_zero_family = "0" in text and "O" in text
    has_one_family = "1" in text and bool(re.search(r"[Il]", text))
    return has_zero_family or has_one_family


def _math_log_like_token_count(line: str) -> int:
    """Count log/ln tokens typical of handwritten math OCR (including coefficients like 2log(...))."""
    patterns = [
        r"(?<![\w.])log\s*\(",  # log (9), Answer: log (
        r"(?<![\w.])log\s*_\s*",  # log_2, log _ 2
        r"(?<=\d)log\s*\(",  # 2log (3)
        r"(?<![\w.])ln\s*\(",  # ln(x)
    ]
    total = 0
    for pattern in patterns:
        total += len(re.findall(pattern, line, flags=re.IGNORECASE))
    return total


def _line_suggests_fraction_linearization_issue(line: str) -> bool:
    stripped = line.strip()
    if len(stripped) < 24:
        return False
    if stripped.count("=") < 2:
        return False
    if _math_log_like_token_count(stripped) < 3:
        return False
    lower = stripped.lower()
    if r"\frac" in lower:
        return False
    if "/" in stripped:
        return False
    return True


def _has_potential_fraction_or_chain_ambiguity(answer_raw: str) -> bool:
    if not answer_raw or not answer_raw.strip():
        return False
    return any(_line_suggests_fraction_linearization_issue(line) for line in answer_raw.splitlines())


def answer_may_need_math_structure_refine(answer_raw: str) -> bool:
    """True when the answer matches the same heuristic as potential_fraction_or_chain_ambiguity (Phase 3 refine trigger)."""
    return _has_potential_fraction_or_chain_ambiguity(answer_raw)


def _dedupe_flags(flags: list[ValidationFlag]) -> list[ValidationFlag]:
    seen: set[str] = set()
    deduped: list[ValidationFlag] = []
    for flag in flags:
        if flag.code in seen:
            continue
        seen.add(flag.code)
        deduped.append(flag)
    return deduped
