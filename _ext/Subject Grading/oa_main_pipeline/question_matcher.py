"""Deterministic question matcher for typed question input."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .config import PipelineConfig
from .content_normalization import (
    build_subject_matcher_text,
    fold_plaintext_science_symbols,
    fold_unicode_numeric_forms,
)
from .schemas import MatchAlternative, QuestionRecord, StatusLabel

_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]+", re.IGNORECASE)
_MULTISPACE_RE = re.compile(r"\s+")
_NUMBER_HINT_RE = re.compile(r"\b(?:q(?:uestion)?\s*)?(\d{1,3})\b", re.IGNORECASE)
_MCQ_OPTION_RE = re.compile(
    r"\b(?:correct(?:\s+answer|\s+option)?\s*(?:is|:)?|option)\s*([ABCD])\b",
    re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    folded = fold_unicode_numeric_forms(str(text or ""))
    folded = fold_plaintext_science_symbols(folded)
    lowered = folded.casefold()
    cleaned = _NON_ALNUM_RE.sub(" ", lowered)
    return _MULTISPACE_RE.sub(" ", cleaned).strip()


def tokenize(text: str) -> List[str]:
    return [tok for tok in normalize_text(text).split(" ") if tok]


def matcher_text_for_subject(
    text: str,
    *,
    subject: Optional[str],
    canonical_text: Optional[str] = None,
) -> str:
    return build_subject_matcher_text(text, subject=subject, canonical_text=canonical_text)


def extract_question_number_hint(text: str) -> Optional[str]:
    for hit in _NUMBER_HINT_RE.finditer(str(text or "")):
        value = hit.group(1)
        if not value:
            continue
        try:
            num = int(value)
        except Exception:
            continue
        if 1 <= num <= 200:
            return str(num)
    return None


def _token_overlap_score(query: str, candidate: str) -> float:
    q_tokens = set(tokenize(query))
    if not q_tokens:
        return 0.0
    c_tokens = set(tokenize(candidate))
    return float(len(q_tokens & c_tokens)) / float(len(q_tokens))


def _token_jaccard_score(query: str, candidate: str) -> float:
    q_tokens = set(tokenize(query))
    c_tokens = set(tokenize(candidate))
    if not q_tokens or not c_tokens:
        return 0.0
    union = q_tokens | c_tokens
    if not union:
        return 0.0
    return float(len(q_tokens & c_tokens)) / float(len(union))


def _sequence_similarity_score(query: str, candidate: str) -> float:
    q = normalize_text(query)
    c = normalize_text(candidate)
    if not q or not c:
        return 0.0
    if q in c or c in q:
        return 1.0
    return difflib.SequenceMatcher(None, q, c).ratio()


def _clamp_score(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _extract_student_option(student_answer: str) -> Optional[str]:
    text = normalize_text(student_answer).upper()
    if text in {"A", "B", "C", "D"}:
        return text
    match = re.search(r"\b([ABCD])\b", str(student_answer or "").upper())
    if match:
        return match.group(1)
    return None


def _extract_scheme_option(marking_scheme_answer: str) -> Optional[str]:
    text = str(marking_scheme_answer or "").strip().upper()
    if text in {"A", "B", "C", "D"}:
        return text
    match = _MCQ_OPTION_RE.search(text)
    if match:
        return match.group(1).upper()
    return None


def _non_mcq_answer_hint_score(
    student_answer: str,
    marking_scheme_answer: str,
    *,
    subject: Optional[str] = None,
) -> float:
    if not student_answer or not marking_scheme_answer:
        return 0.0
    normalized_student = matcher_text_for_subject(student_answer, subject=subject)
    normalized_scheme = matcher_text_for_subject(marking_scheme_answer, subject=subject)
    student_tokens = set(tokenize(normalized_student))
    if not student_tokens:
        return 0.0
    scheme_tokens = set(tokenize(normalized_scheme))
    if not scheme_tokens:
        return 0.0
    overlap = float(len(student_tokens & scheme_tokens)) / float(len(student_tokens))
    return _clamp_score(min(overlap, 0.6))


def answer_hint_score(student_answer: str, marking_scheme_answer: str, *, subject: Optional[str] = None) -> float:
    scheme_option = _extract_scheme_option(marking_scheme_answer)
    if scheme_option:
        return 1.0 if _extract_student_option(student_answer) == scheme_option else 0.0
    return _non_mcq_answer_hint_score(student_answer, marking_scheme_answer, subject=subject)


@dataclass(frozen=True)
class MatchResult:
    status: StatusLabel
    match_confidence: float
    best_record: Optional[QuestionRecord]
    top_alternatives: List[MatchAlternative]


def _build_alternatives(
    scored: Sequence[Tuple[float, QuestionRecord]],
    *,
    limit: int,
) -> List[MatchAlternative]:
    return [
        MatchAlternative(
            question_id=rec.question_id,
            match_confidence=round(score, 4),
            question_text=rec.question_text,
            source_paper_reference=rec.source_paper_reference,
        )
        for score, rec in scored[:limit]
    ]


def match_question(
    query_question: str,
    records: Sequence[QuestionRecord],
    query_subject: Optional[str] = None,
    config: Optional[PipelineConfig] = None,
) -> MatchResult:
    cfg = config or PipelineConfig()
    if not records:
        return MatchResult(
            status="failed",
            match_confidence=0.0,
            best_record=None,
            top_alternatives=[],
        )

    qnum_hint = extract_question_number_hint(query_question)
    normalized_query = matcher_text_for_subject(query_question, subject=query_subject)
    scored: List[Tuple[float, QuestionRecord]] = []
    for record in records:
        candidate_text = matcher_text_for_subject(record.question_text, subject=record.subject)
        token_score = _token_overlap_score(normalized_query, candidate_text)
        seq_score = _sequence_similarity_score(normalized_query, candidate_text)
        number_bonus = 0.0
        if qnum_hint and qnum_hint == str(record.question_number):
            number_bonus = cfg.number_bonus_weight
        total = (
            (cfg.token_overlap_weight * token_score)
            + (cfg.sequence_similarity_weight * seq_score)
            + number_bonus
        )
        total = _clamp_score(total)
        scored.append((total, record))

    scored.sort(key=lambda item: (item[0], item[1].question_id), reverse=True)
    best_score, best_record = scored[0]
    if best_score >= cfg.accepted_threshold:
        status: StatusLabel = "accepted"
    elif best_score >= cfg.review_threshold:
        status = "review_required"
    else:
        status = "failed"

    return MatchResult(
        status=status,
        match_confidence=round(best_score, 4),
        best_record=best_record,
        top_alternatives=_build_alternatives(scored, limit=cfg.top_alternatives),
    )


def rerank_search_results(
    query_question: str,
    student_answer: str,
    records: Sequence[QuestionRecord],
    embedding_scores: Dict[str, float],
    use_answer_hint: bool = True,
    query_subject: Optional[str] = None,
    config: Optional[PipelineConfig] = None,
) -> Tuple[MatchResult, List[Dict[str, object]]]:
    cfg = config or PipelineConfig()
    if not records:
        return (
            MatchResult(
                status="failed",
                match_confidence=0.0,
                best_record=None,
                top_alternatives=[],
            ),
            [],
        )

    scored: List[Tuple[float, QuestionRecord]] = []
    debug_rows: List[Dict[str, object]] = []
    normalized_query = matcher_text_for_subject(query_question, subject=query_subject)
    for record in records:
        embed_score = _clamp_score(embedding_scores.get(record.question_id, 0.0))
        candidate_question = matcher_text_for_subject(record.question_text, subject=record.subject)
        lexical_score = _token_jaccard_score(normalized_query, candidate_question)
        hint_score = (
            answer_hint_score(student_answer, record.marking_scheme_answer, subject=record.subject)
            if use_answer_hint
            else 0.0
        )
        final_score = _clamp_score(
            (0.75 * embed_score) + (0.20 * lexical_score) + (0.05 * hint_score)
        )
        scored.append((final_score, record))
        debug_rows.append(
            {
                "question_id": record.question_id,
                "embedding_similarity": round(embed_score, 4),
                "lexical_overlap": round(lexical_score, 4),
                "answer_hint_score": round(hint_score, 4),
                "answer_hint_used": bool(use_answer_hint),
                "final_score": round(final_score, 4),
                "question_text": record.question_text,
                "matcher_text": candidate_question,
                "source_paper_reference": record.source_paper_reference,
            }
        )

    scored.sort(key=lambda item: (item[0], item[1].question_id), reverse=True)
    debug_rows.sort(
        key=lambda item: (
            float(item.get("final_score") or 0.0),
            str(item.get("question_id") or ""),
        ),
        reverse=True,
    )
    best_score, best_record = scored[0]
    if best_score >= cfg.search_accepted_threshold:
        status: StatusLabel = "accepted"
    elif best_score >= cfg.search_review_threshold:
        status = "review_required"
    else:
        status = "failed"

    return (
        MatchResult(
            status=status,
            match_confidence=round(best_score, 4),
            best_record=best_record,
            top_alternatives=_build_alternatives(scored, limit=cfg.top_alternatives),
        ),
        debug_rows,
    )
