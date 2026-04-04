"""Answer evaluator with Grok-first grading and deterministic fallback."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, List, Literal, Sequence

import requests

from .config import PipelineConfig
from .content_normalization import fold_unicode_numeric_forms
from .schemas import GradeLabel

GROK_CHAT_URL = "https://api.x.ai/v1/chat/completions"
_MCQ_RE = re.compile(r"^\s*([ABCD])\s*$", re.IGNORECASE)
_MCQ_TOKEN_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)
_MCQ_SCHEME_PATTERNS = (
    re.compile(r"\bcorrect(?:\s+answer|\s+option)?\s*(?:is|:)?\s*([ABCD])\b", re.IGNORECASE),
    re.compile(r"\banswer\s*[:\-]\s*([ABCD])\b", re.IGNORECASE),
    re.compile(r"\boption\s*([ABCD])\s*(?:is\s+correct|correct)\b", re.IGNORECASE),
)
_SPLIT_RE = re.compile(r"(?:\n+|[.;:])")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]+", re.IGNORECASE)
_SPACE_RE = re.compile(r"\s+")
_ANS_PREFIX_RE = re.compile(r"^\s*(?:a\.|ans(?:wer)?\s*[:\-]|q\.|question\s*[:\-])\s*", re.IGNORECASE)

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
}


@dataclass(frozen=True)
class EvaluationResult:
    score: float
    score_percent: float
    grade_label: GradeLabel
    expected_points: List[str]
    missing_points: List[str]
    student_option: str | None = None
    correct_option: str | None = None
    grading_source: Literal["grok", "deterministic"] = "deterministic"
    grading_model: str | None = None
    grading_error: str | None = None


def _normalize_text(text: str) -> str:
    folded = fold_unicode_numeric_forms(str(text or ""))
    lowered = folded.casefold()
    cleaned = _NON_ALNUM_RE.sub(" ", lowered)
    return _SPACE_RE.sub(" ", cleaned).strip()


def _tokens(text: str) -> List[str]:
    return [t for t in _normalize_text(text).split(" ") if t and t not in _STOPWORDS]


def _extract_expected_points(marking_scheme_answer: str) -> List[str]:
    chunks: List[str] = []
    for raw in _SPLIT_RE.split(str(marking_scheme_answer or "")):
        value = " ".join(raw.split())
        if len(value) < 4:
            continue
        chunks.append(value)
    if not chunks and marking_scheme_answer.strip():
        chunks = [marking_scheme_answer.strip()]
    return chunks


def _extract_student_option(student_answer: str) -> str | None:
    text = str(student_answer or "").strip()
    text = _ANS_PREFIX_RE.sub("", text).strip().upper()
    if not text:
        return None
    exact = _MCQ_RE.match(text)
    if exact:
        return exact.group(1)
    token = _MCQ_TOKEN_RE.search(text)
    if token:
        return token.group(1).upper()
    return None


def _extract_scheme_option(marking_scheme_answer: str) -> str | None:
    text = str(marking_scheme_answer or "").strip()
    if not text:
        return None
    exact = _MCQ_RE.match(text)
    if exact:
        return exact.group(1).upper()
    for pattern in _MCQ_SCHEME_PATTERNS:
        hit = pattern.search(text)
        if hit:
            return hit.group(1).upper()
    return None


def _is_mcq_scheme(marking_scheme_answer: str) -> bool:
    return _extract_scheme_option(marking_scheme_answer) is not None


def _normalize_grade_label(value: object, score: float, config: PipelineConfig) -> GradeLabel:
    text = str(value or "").strip().lower()
    if text in {"fully_correct", "partially_correct", "weak"}:
        return text  # type: ignore[return-value]
    return _grade_from_score(score, config)


def _coerce_points(value: object, *, max_items: int = 8) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for row in value:
        text = " ".join(str(row or "").split()).strip()
        if not text:
            continue
        out.append(text)
        if len(out) >= max_items:
            break
    return out


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = str(text or "").strip()
    if not cleaned:
        raise ValueError("empty_grok_response")
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        candidate = cleaned[start : end + 1]
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("invalid_json_payload")


def _build_grok_messages(
    *,
    question_text: str | None,
    student_answer: str,
    marking_scheme_answer: str,
) -> list[dict[str, str]]:
    system_prompt = (
        "You are an O/A Levels exam answer grader. Grade strictly against the supplied marking scheme. "
        "Return JSON only with keys: score, grade_label, feedback, expected_points, missing_points, "
        "student_option, correct_option. "
        "Rules: score must be a number between 0 and 1. grade_label must be one of "
        "fully_correct, partially_correct, weak. expected_points and missing_points must be arrays of short strings. "
        "For MCQ, extract correct option from the marking scheme and set score to 1 if student option matches else 0."
    )
    payload = {
        "question_text": str(question_text or "").strip(),
        "student_answer": str(student_answer or "").strip(),
        "marking_scheme_answer": str(marking_scheme_answer or "").strip(),
        "mcq_scheme_hint": _extract_scheme_option(marking_scheme_answer),
    }
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def _evaluate_with_grok(
    *,
    question_text: str | None,
    student_answer: str,
    marking_scheme_answer: str,
    config: PipelineConfig,
) -> tuple[EvaluationResult | None, str | None]:
    last_error: str | None = None
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.grok_api_key}",
    }
    payload = {
        "model": config.grok_model,
        "temperature": 0.0,
        "messages": _build_grok_messages(
            question_text=question_text,
            student_answer=student_answer,
            marking_scheme_answer=marking_scheme_answer,
        ),
    }

    attempts = max(1, int(config.grok_max_retries) + 1)
    timeout = max(10, int(config.grok_timeout_seconds))
    for attempt in range(1, attempts + 1):
        try:
            resp = requests.post(
                GROK_CHAT_URL,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            resp.raise_for_status()
            body = resp.json()
            content = (
                (((body.get("choices") or [{}])[0].get("message") or {}).get("content"))
                if isinstance(body, dict)
                else ""
            )
            parsed = _extract_json_object(str(content or ""))

            score_raw = parsed.get("score", 0.0)
            score = max(0.0, min(1.0, float(score_raw)))
            grade_label = _normalize_grade_label(parsed.get("grade_label"), score, config)
            feedback = " ".join(str(parsed.get("feedback") or "").split()).strip()
            expected_points = _coerce_points(parsed.get("expected_points"))
            missing_points = _coerce_points(parsed.get("missing_points"))

            student_option = _extract_student_option(str(parsed.get("student_option") or student_answer))
            correct_option = _extract_student_option(str(parsed.get("correct_option") or "")) or _extract_scheme_option(
                marking_scheme_answer
            )

            # Enforce binary score for clear MCQ keys.
            if correct_option:
                score = 1.0 if student_option == correct_option else 0.0
                grade_label = "fully_correct" if score == 1.0 else "weak"
                if not expected_points:
                    expected_points = [f"Correct option: {correct_option}"]
                if score == 0.0 and not missing_points:
                    missing_points = [f"Select option {correct_option}."]

            if not feedback:
                if score >= config.fully_correct_threshold:
                    feedback = "Answer aligns with the marking scheme."
                elif score >= config.partially_correct_threshold:
                    feedback = "Answer is partially aligned with the marking scheme."
                else:
                    feedback = "Answer is weakly aligned with the marking scheme."

            return (
                EvaluationResult(
                    score=round(score, 4),
                    score_percent=round(score * 100.0, 2),
                    grade_label=grade_label,
                    expected_points=expected_points,
                    missing_points=missing_points,
                    student_option=student_option,
                    correct_option=correct_option,
                    grading_source="grok",
                    grading_model=config.grok_model,
                    grading_error=None,
                ),
                None,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            if attempt < attempts:
                time.sleep(min(0.3 * attempt, 1.0))

    return None, f"grok_grading_failed:{last_error}" if last_error else "grok_grading_failed:unknown"


def _grade_from_score(score: float, config: PipelineConfig) -> GradeLabel:
    if score >= config.fully_correct_threshold:
        return "fully_correct"
    if score >= config.partially_correct_threshold:
        return "partially_correct"
    return "weak"


def _evaluate_mcq(student_answer: str, marking_scheme_answer: str) -> EvaluationResult:
    correct = _extract_scheme_option(marking_scheme_answer) or str(marking_scheme_answer or "").strip().upper()
    chosen = _extract_student_option(student_answer)
    is_correct = chosen == correct
    score = 1.0 if is_correct else 0.0
    expected_points = [f"Correct option: {correct}"]
    scheme_text = str(marking_scheme_answer or "").strip()
    if scheme_text and not _MCQ_RE.match(scheme_text):
        expected_points.append(scheme_text)
    missing_points = [] if is_correct else [f"Select option {correct}."]
    return EvaluationResult(
        score=score,
        score_percent=round(score * 100.0, 2),
        grade_label="fully_correct" if is_correct else "weak",
        expected_points=expected_points,
        missing_points=missing_points,
        student_option=chosen,
        correct_option=correct,
    )


def _coverage_for_point(point_tokens: Sequence[str], student_tokens: Sequence[str]) -> float:
    if not point_tokens:
        return 0.0
    point_set = set(point_tokens)
    student_set = set(student_tokens)
    overlap = point_set & student_set
    return float(len(overlap)) / float(len(point_set))


def _evaluate_non_mcq(
    student_answer: str,
    marking_scheme_answer: str,
    config: PipelineConfig,
) -> EvaluationResult:
    expected_points = _extract_expected_points(marking_scheme_answer)
    if not expected_points:
        return EvaluationResult(
            score=0.0,
            score_percent=0.0,
            grade_label="weak",
            expected_points=[],
            missing_points=[],
        )

    student_tokens = _tokens(student_answer)
    matched_points: List[str] = []
    missing_points: List[str] = []
    for point in expected_points:
        point_tokens = _tokens(point)
        coverage = _coverage_for_point(point_tokens, student_tokens)
        if coverage >= 0.4 or (len(set(point_tokens) & set(student_tokens)) >= 2):
            matched_points.append(point)
        else:
            missing_points.append(point)

    score = float(len(matched_points)) / float(len(expected_points))
    score = max(0.0, min(1.0, score))
    return EvaluationResult(
        score=round(score, 4),
        score_percent=round(score * 100.0, 2),
        grade_label=_grade_from_score(score, config),
        expected_points=expected_points,
        missing_points=missing_points,
    )


def evaluate_answer(
    student_answer: str,
    marking_scheme_answer: str,
    config: PipelineConfig | None = None,
    question_text: str | None = None,
) -> EvaluationResult:
    cfg = config or PipelineConfig()
    cleaned_student_answer = str(student_answer or "").strip()
    # Remove common labels that can trick MCQ detection in free-response.
    cleaned_student_answer = _ANS_PREFIX_RE.sub("", cleaned_student_answer).strip()
    if cfg.use_grok_grading and cfg.grok_api_key:
        grok_result, grok_error = _evaluate_with_grok(
            question_text=question_text,
            student_answer=cleaned_student_answer,
            marking_scheme_answer=marking_scheme_answer,
            config=cfg,
        )
        if grok_result is not None:
            return grok_result
    else:
        grok_error = "grok_grading_skipped_missing_config"

    if _is_mcq_scheme(marking_scheme_answer):
        fallback = _evaluate_mcq(cleaned_student_answer, marking_scheme_answer)
    else:
        fallback = _evaluate_non_mcq(cleaned_student_answer, marking_scheme_answer, cfg)
    return EvaluationResult(
        score=fallback.score,
        score_percent=fallback.score_percent,
        grade_label=fallback.grade_label,
        expected_points=list(fallback.expected_points),
        missing_points=list(fallback.missing_points),
        student_option=fallback.student_option,
        correct_option=fallback.correct_option,
        grading_source="deterministic",
        grading_model=None,
        grading_error=grok_error,
    )
