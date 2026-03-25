"""Data schemas for Phase 1 evaluator pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


StatusLabel = Literal["accepted", "review_required", "failed"]
GradeLabel = Literal["fully_correct", "partially_correct", "weak"]
DataSourceLabel = Literal["oa_main_dataset", "o_level_json", "o_level_main_json"]


@dataclass(frozen=True)
class EvaluateRequest:
    question: str
    student_answer: str
    subject: Optional[str] = None
    year: Optional[int] = None
    session: Optional[str] = None
    paper: Optional[str] = None
    variant: Optional[str] = None
    question_id: Optional[str] = None


@dataclass(frozen=True)
class QuestionRecord:
    question_id: str
    subject: str
    year: int
    session: str
    paper: str
    variant: str
    question_number: str
    sub_question: Optional[str]
    question_text: str
    marking_scheme_answer: str
    page_number: Optional[int]
    source_paper_reference: str


@dataclass(frozen=True)
class MatchAlternative:
    question_id: str
    match_confidence: float
    question_text: str
    source_paper_reference: str


@dataclass(frozen=True)
class EvaluateResponse:
    status: StatusLabel
    matched_question_text: str
    student_answer: str
    score: float
    score_percent: float
    grade_label: GradeLabel
    feedback: str
    missing_points: List[str]
    expected_points: List[str]
    source_paper_reference: str
    marking_scheme_answer: str
    page_number: Optional[int]
    match_confidence: float
    matched_question_id: Optional[str]
    data_source: DataSourceLabel = "o_level_json"
    fallback_used: bool = False
    primary_status: StatusLabel = "failed"
    primary_match_confidence: float = 0.0
    primary_data_source: DataSourceLabel = "o_level_json"
    fallback_data_source: Optional[DataSourceLabel] = "oa_main_dataset"
    debug_trace: Optional[Dict[str, Any]] = None
    top_alternatives: List[MatchAlternative] = field(default_factory=list)


@dataclass(frozen=True)
class ModeAPreviewResponse:
    request_id: str
    extracted_question_text: str
    extracted_student_answer: str
    normalized_question_text: str
    normalized_student_answer: str
    vision_confidence: float
    vision_warnings: List[str]
    match_confidence: float
    top1_top2_margin: float
    auto_accept_eligible: bool
    auto_accept_reason: str
    canonical_question_text: Optional[str] = None
    canonical_student_answer: Optional[str] = None
    math_extraction_profile_used: bool = False
    content_type: Optional[str] = None
    recovery_applied: bool = False
    recovery_reason_codes: List[str] = field(default_factory=list)
    question_candidates: List[Dict[str, Any]] = field(default_factory=list)
    top_alternatives: List[MatchAlternative] = field(default_factory=list)
    debug_run_id: Optional[str] = None
