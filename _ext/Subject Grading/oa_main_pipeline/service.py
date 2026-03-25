"""Phase 2 orchestration service for typed question evaluation + fallback."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Dict, List, Optional

from .answer_evaluator import evaluate_answer
from .config import PipelineConfig
from .dataset_repository import DatasetRepository
from .feedback_builder import build_feedback
from .fallback_repository import FallbackDatasetRepository
from .markscheme_lookup import lookup_markscheme
from .o_level_main_repository import MainJsonRepository
from .question_matcher import MatchResult
from .search_index import SearchIndexManager
from .schemas import (
    DataSourceLabel,
    EvaluateRequest,
    EvaluateResponse,
    MatchAlternative,
    QuestionRecord,
    StatusLabel,
)


@dataclass(frozen=True)
class _SourceSpec:
    repository: object
    data_source: DataSourceLabel
    no_records_feedback: str
    no_match_feedback: str
    debug_stage: str


class OALevelEvaluatorService:
    """Question match + answer evaluation service with configurable source order."""

    def __init__(
        self,
        repository: Optional[DatasetRepository] = None,
        fallback_repository: Optional[FallbackDatasetRepository] = None,
        main_repository: Optional[MainJsonRepository] = None,
        config: Optional[PipelineConfig] = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.repository = repository or DatasetRepository(self.config)
        self.fallback_repository = fallback_repository or FallbackDatasetRepository(self.config)
        self.main_repository = main_repository or MainJsonRepository(self.config)
        self.search_index = SearchIndexManager(
            repository=self.repository,
            fallback_repository=self.fallback_repository,
            main_repository=self.main_repository,
            config=self.config,
        )

    def warmup(self) -> None:
        """Pre-load the embedding model and primary search index so the first evaluate request is fast."""
        self.search_index.warmup_embedder()
        source_specs = self._source_specs()
        if source_specs:
            primary_source = source_specs[0].data_source
            self.search_index.ensure_built(primary_source)

    def evaluate(
        self,
        request: EvaluateRequest,
        debug: bool = False,
        use_answer_hint: bool = True,
        direct_question_id_allowed: bool = True,
    ) -> EvaluateResponse:
        started_total = time.perf_counter()
        debug_trace = self._init_debug_trace(request) if debug else None
        request = self._sanitize_request(request)
        source_specs = self._source_specs()
        primary_spec = source_specs[0]
        secondary_spec = source_specs[1] if self.config.enable_fallback and len(source_specs) > 1 else None
        fallback_source = secondary_spec.data_source if secondary_spec is not None else None
        if debug_trace is not None:
            self._set_sanitized_request_debug(debug_trace, request)
            debug_trace["source_order"] = {
                "source_priority": self.config.source_priority,
                "primary_data_source": primary_spec.data_source,
                "fallback_data_source": fallback_source,
            }
            debug_trace["retrieval"] = {
                "use_answer_hint": bool(use_answer_hint),
                "direct_question_id_allowed": bool(direct_question_id_allowed),
            }
        question = request.question
        student_answer = request.student_answer
        if not question:
            response = self._failed_response(
                student_answer=student_answer,
                feedback="Question is required.",
                data_source=primary_spec.data_source,
                fallback_used=False,
                primary_status="failed",
                primary_match_confidence=0.0,
                primary_data_source=primary_spec.data_source,
                fallback_data_source=fallback_source,
                debug_trace=debug_trace,
            )
            return self._finalize_debug(response, debug_trace, started_total)
        if not student_answer:
            response = self._failed_response(
                student_answer=student_answer,
                feedback="Student answer is required.",
                data_source=primary_spec.data_source,
                fallback_used=False,
                primary_status="failed",
                primary_match_confidence=0.0,
                primary_data_source=primary_spec.data_source,
                fallback_data_source=fallback_source,
                debug_trace=debug_trace,
            )
            return self._finalize_debug(response, debug_trace, started_total)

        primary_started = time.perf_counter()
        primary_response = self._evaluate_from_repository(
            repository=primary_spec.repository,
            request=request,
            student_answer=student_answer,
            data_source=primary_spec.data_source,
            fallback_used=False,
            no_records_feedback=primary_spec.no_records_feedback,
            no_match_feedback=primary_spec.no_match_feedback,
            debug_trace=debug_trace,
            debug_stage=primary_spec.debug_stage,
            primary_data_source=primary_spec.data_source,
            fallback_data_source=fallback_source,
            use_answer_hint=use_answer_hint,
            direct_question_id_allowed=direct_question_id_allowed,
        )
        if debug_trace is not None:
            debug_trace["timings_ms"]["primary"] = int((time.perf_counter() - primary_started) * 1000)
        if primary_response.status != "failed" or secondary_spec is None:
            return self._finalize_debug(primary_response, debug_trace, started_total)

        fallback_started = time.perf_counter()
        fallback_response = self._evaluate_from_repository(
            repository=secondary_spec.repository,
            request=request,
            student_answer=student_answer,
            data_source=secondary_spec.data_source,
            fallback_used=True,
            no_records_feedback=secondary_spec.no_records_feedback,
            no_match_feedback=secondary_spec.no_match_feedback,
            primary_status=primary_response.status,
            primary_match_confidence=primary_response.match_confidence,
            debug_trace=debug_trace,
            debug_stage=secondary_spec.debug_stage,
            primary_data_source=primary_spec.data_source,
            fallback_data_source=secondary_spec.data_source,
            use_answer_hint=use_answer_hint,
            direct_question_id_allowed=direct_question_id_allowed,
        )
        if debug_trace is not None:
            debug_trace["timings_ms"]["fallback"] = int((time.perf_counter() - fallback_started) * 1000)
        if fallback_response.status != "failed":
            return self._finalize_debug(fallback_response, debug_trace, started_total)

        combined_feedback = (
            "Unable to confidently match the provided question in primary and fallback datasets."
        )
        response = self._failed_response(
            student_answer=student_answer,
            feedback=combined_feedback,
            alternatives=fallback_response.top_alternatives or primary_response.top_alternatives,
            match_confidence=max(
                primary_response.match_confidence,
                fallback_response.match_confidence,
            ),
            data_source=secondary_spec.data_source,
            fallback_used=True,
            primary_status=primary_response.status,
            primary_match_confidence=primary_response.match_confidence,
            primary_data_source=primary_spec.data_source,
            fallback_data_source=secondary_spec.data_source,
            debug_trace=debug_trace,
        )
        return self._finalize_debug(response, debug_trace, started_total)

    def _evaluate_from_repository(
        self,
        *,
        repository: object,
        request: EvaluateRequest,
        student_answer: str,
        data_source: DataSourceLabel,
        fallback_used: bool,
        no_records_feedback: str,
        no_match_feedback: str,
        primary_status: Optional[StatusLabel] = None,
        primary_match_confidence: Optional[float] = None,
        debug_trace: Optional[Dict[str, Any]] = None,
        debug_stage: str = "primary",
        primary_data_source: DataSourceLabel = "o_level_json",
        fallback_data_source: Optional[DataSourceLabel] = "oa_main_dataset",
        use_answer_hint: bool = True,
        direct_question_id_allowed: bool = True,
    ) -> EvaluateResponse:
        all_records = repository.get_records()  # type: ignore[attr-defined]
        scoped_records = repository.filter_records(all_records, request)  # type: ignore[attr-defined]
        base_primary_status = primary_status or "failed"
        base_primary_confidence = primary_match_confidence if primary_match_confidence is not None else 0.0
        if debug_trace is not None:
            self._append_stage_debug(
                debug_trace=debug_trace,
                stage=debug_stage,
                data_source=data_source,
                records_total=len(all_records),
                records_scoped=len(scoped_records),
                status="failed" if not scoped_records else "running",
                match_confidence=0.0,
                matched_question_id=None,
                reason="no_records_for_filters" if not scoped_records else "records_scoped",
            )
        if not scoped_records:
            return self._failed_response(
                student_answer=student_answer,
                feedback=no_records_feedback,
                data_source=data_source,
                fallback_used=fallback_used,
                primary_status=base_primary_status,
                primary_match_confidence=base_primary_confidence,
                primary_data_source=primary_data_source,
                fallback_data_source=fallback_data_source,
                debug_trace=debug_trace,
            )

        match_result, search_debug = self._resolve_record(
            request,
            student_answer,
            scoped_records,
            repository,
            data_source,
            use_answer_hint=use_answer_hint,
            direct_question_id_allowed=direct_question_id_allowed,
        )
        if debug_trace is not None and search_debug is not None:
            debug_trace.setdefault("search_diagnostics", []).append(search_debug)
        if debug_trace is not None:
            self._append_stage_debug(
                debug_trace=debug_trace,
                stage=f"{debug_stage}_match",
                data_source=data_source,
                records_total=len(all_records),
                records_scoped=len(scoped_records),
                status=match_result.status,
                match_confidence=match_result.match_confidence,
                matched_question_id=(match_result.best_record.question_id if match_result.best_record else None),
                reason="no_confident_match" if match_result.best_record is None else "matched",
            )
        if match_result.best_record is None:
            return self._failed_response(
                student_answer=student_answer,
                feedback=no_match_feedback,
                alternatives=match_result.top_alternatives,
                match_confidence=match_result.match_confidence,
                data_source=data_source,
                fallback_used=fallback_used,
                primary_status=primary_status or match_result.status,
                primary_match_confidence=(
                    primary_match_confidence
                    if primary_match_confidence is not None
                    else match_result.match_confidence
                ),
                primary_data_source=primary_data_source,
                fallback_data_source=fallback_data_source,
                debug_trace=debug_trace,
            )

        record = match_result.best_record
        lookup = lookup_markscheme(record)
        evaluation = evaluate_answer(
            student_answer=student_answer,
            marking_scheme_answer=lookup.marking_scheme_answer,
            config=self.config,
            question_text=record.question_text,
        )
        if debug_trace is not None:
            debug_trace["grading"] = {
                "source": evaluation.grading_source,
                "model": evaluation.grading_model,
                "error": evaluation.grading_error,
                "correct_option": evaluation.correct_option,
                "student_option": evaluation.student_option,
            }
        feedback = build_feedback(
            grade_label=evaluation.grade_label,
            score_percent=evaluation.score_percent,
            expected_points=evaluation.expected_points,
            missing_points=evaluation.missing_points,
            is_mcq=evaluation.correct_option is not None,
            correct_option=evaluation.correct_option,
        )
        return EvaluateResponse(
            status=match_result.status,
            matched_question_text=record.question_text,
            student_answer=student_answer,
            score=round(evaluation.score, 4),
            score_percent=round(evaluation.score_percent, 2),
            grade_label=evaluation.grade_label,
            feedback=feedback,
            missing_points=list(evaluation.missing_points),
            expected_points=list(evaluation.expected_points),
            source_paper_reference=lookup.source_paper_reference,
            marking_scheme_answer=lookup.marking_scheme_answer,
            page_number=lookup.page_number,
            match_confidence=round(match_result.match_confidence, 4),
            matched_question_id=record.question_id,
            data_source=data_source,
            fallback_used=fallback_used,
            primary_status=primary_status or match_result.status,
            primary_match_confidence=round(
                primary_match_confidence
                if primary_match_confidence is not None
                else match_result.match_confidence,
                4,
            ),
            primary_data_source=primary_data_source,
            fallback_data_source=fallback_data_source,
            debug_trace=debug_trace,
            top_alternatives=list(match_result.top_alternatives),
        )

    def _source_specs(self) -> List[_SourceSpec]:
        o_level_spec = _SourceSpec(
            repository=self.fallback_repository,
            data_source="o_level_json",
            no_records_feedback="No O_LEVEL_JSON entries found for the provided filters.",
            no_match_feedback="Unable to confidently match the provided question in O_LEVEL_JSON.",
            debug_stage="primary",
        )
        oa_spec = _SourceSpec(
            repository=self.repository,
            data_source="oa_main_dataset",
            no_records_feedback="No accepted OA_MAIN_DATASET entries found for the provided filters.",
            no_match_feedback="Unable to confidently match the provided question in OA_MAIN_DATASET.",
            debug_stage="fallback",
        )
        main_spec = _SourceSpec(
            repository=self.main_repository,
            data_source="o_level_main_json",
            no_records_feedback="No O_LEVEL_MAIN_JSON entries found for the provided filters.",
            no_match_feedback="Unable to confidently match the provided question in O_LEVEL_MAIN_JSON.",
            debug_stage="primary",
        )
        priority = str(self.config.source_priority or "").strip().lower()
        if priority == "o_level_main_first":
            oa_as_fallback = _SourceSpec(
                repository=self.repository,
                data_source="oa_main_dataset",
                no_records_feedback="No accepted OA_MAIN_DATASET entries found for the provided filters.",
                no_match_feedback="Unable to confidently match the provided question in OA_MAIN_DATASET.",
                debug_stage="fallback",
            )
            return [main_spec, oa_as_fallback]
        if priority == "oa_main_dataset_first":
            o_level_spec = _SourceSpec(
                repository=self.fallback_repository,
                data_source="o_level_json",
                no_records_feedback="No O_LEVEL_JSON entries found for the provided filters.",
                no_match_feedback="Unable to confidently match the provided question in O_LEVEL_JSON.",
                debug_stage="fallback",
            )
            oa_spec = _SourceSpec(
                repository=self.repository,
                data_source="oa_main_dataset",
                no_records_feedback="No accepted OA_MAIN_DATASET entries found for the provided filters.",
                no_match_feedback="Unable to confidently match the provided question in OA_MAIN_DATASET.",
                debug_stage="primary",
            )
            return [oa_spec, o_level_spec]
        return [o_level_spec, oa_spec]

    def _sanitize_request(self, request: EvaluateRequest) -> EvaluateRequest:
        def _clean_text(value: object) -> Optional[str]:
            text = str(value or "").strip()
            if not text:
                return None
            if text.casefold() in {"string", "none", "null", "undefined", "n/a", "na"}:
                return None
            return text

        year_value: Optional[int] = None
        if request.year is not None:
            try:
                parsed_year = int(request.year)
                if parsed_year > 0:
                    year_value = parsed_year
            except Exception:
                year_value = None

        return EvaluateRequest(
            question=(request.question or "").strip(),
            student_answer=(request.student_answer or "").strip(),
            subject=_clean_text(request.subject),
            year=year_value,
            session=_clean_text(request.session),
            paper=_clean_text(request.paper),
            variant=_clean_text(request.variant),
            question_id=_clean_text(request.question_id),
        )

    def _resolve_record(
        self,
        request: EvaluateRequest,
        student_answer: str,
        records: List[QuestionRecord],
        repository: object,
        data_source: DataSourceLabel,
        use_answer_hint: bool = True,
        direct_question_id_allowed: bool = True,
    ) -> tuple[MatchResult, Optional[Dict[str, Any]]]:
        if direct_question_id_allowed and request.question_id:
            resolved = repository.get_by_question_id(request.question_id)  # type: ignore[attr-defined]
            if resolved and resolved in records:
                return (
                    MatchResult(
                        status="accepted",
                        match_confidence=1.0,
                        best_record=resolved,
                        top_alternatives=[
                            MatchAlternative(
                                question_id=resolved.question_id,
                                match_confidence=1.0,
                                question_text=resolved.question_text,
                                source_paper_reference=resolved.source_paper_reference,
                            )
                        ],
                    ),
                    {
                        "search_method": "question_id_direct",
                        "index_source": data_source,
                        "records_scoped": len(records),
                        "final_search_score": 1.0,
                        "top_search_candidates": [
                            {
                                "question_id": resolved.question_id,
                                "final_score": 1.0,
                                "question_text": resolved.question_text,
                                "source_paper_reference": resolved.source_paper_reference,
                            }
                        ],
                    },
                )
            return (
                MatchResult(
                    status="failed",
                    match_confidence=0.0,
                    best_record=None,
                    top_alternatives=[],
                ),
                {
                    "search_method": "question_id_direct",
                    "index_source": data_source,
                    "records_scoped": len(records),
                    "final_search_score": 0.0,
                    "top_search_candidates": [],
                },
            )
        search_result = self.search_index.search(
            source=data_source,
            query=request.question,
            query_subject=request.subject,
            student_answer=student_answer,
            records=records,
            use_answer_hint=use_answer_hint,
        )
        return search_result.match_result, search_result.debug

    def _failed_response(
        self,
        *,
        student_answer: str,
        feedback: str,
        data_source: DataSourceLabel,
        fallback_used: bool,
        primary_status: StatusLabel,
        primary_match_confidence: float,
        primary_data_source: DataSourceLabel,
        fallback_data_source: Optional[DataSourceLabel],
        alternatives: Optional[List[MatchAlternative]] = None,
        match_confidence: float = 0.0,
        debug_trace: Optional[Dict[str, Any]] = None,
    ) -> EvaluateResponse:
        return EvaluateResponse(
            status="failed",
            matched_question_text="",
            student_answer=student_answer,
            score=0.0,
            score_percent=0.0,
            grade_label="weak",
            feedback=feedback,
            missing_points=[],
            expected_points=[],
            source_paper_reference="",
            marking_scheme_answer="",
            page_number=None,
            match_confidence=round(match_confidence, 4),
            matched_question_id=None,
            data_source=data_source,
            fallback_used=fallback_used,
            primary_status=primary_status,
            primary_match_confidence=round(primary_match_confidence, 4),
            primary_data_source=primary_data_source,
            fallback_data_source=fallback_data_source,
            debug_trace=debug_trace,
            top_alternatives=list(alternatives or []),
        )

    def _init_debug_trace(self, request: EvaluateRequest) -> Dict[str, Any]:
        return {
            "enabled": True,
            "request_raw": {
                "subject": request.subject,
                "year": request.year,
                "session": request.session,
                "paper": request.paper,
                "variant": request.variant,
                "question_id": request.question_id,
                "question_len": len(str(request.question or "")),
                "student_answer_len": len(str(request.student_answer or "")),
            },
            "request_sanitized": {},
            "timings_ms": {},
            "stages": [],
            "final_decision": {},
        }

    def _set_sanitized_request_debug(self, debug_trace: Dict[str, Any], request: EvaluateRequest) -> None:
        debug_trace["request_sanitized"] = {
            "subject": request.subject,
            "year": request.year,
            "session": request.session,
            "paper": request.paper,
            "variant": request.variant,
            "question_id": request.question_id,
            "question_len": len(request.question or ""),
            "student_answer_len": len(request.student_answer or ""),
        }

    def _append_stage_debug(
        self,
        *,
        debug_trace: Dict[str, Any],
        stage: str,
        data_source: DataSourceLabel,
        records_total: int,
        records_scoped: int,
        status: StatusLabel | str,
        match_confidence: float,
        matched_question_id: Optional[str],
        reason: str,
    ) -> None:
        debug_trace.setdefault("stages", []).append(
            {
                "stage": stage,
                "data_source": data_source,
                "records_total": records_total,
                "records_scoped": records_scoped,
                "status": status,
                "match_confidence": round(match_confidence, 4),
                "matched_question_id": matched_question_id,
                "reason": reason,
            }
        )

    def _finalize_debug(
        self,
        response: EvaluateResponse,
        debug_trace: Optional[Dict[str, Any]],
        started_total: float,
    ) -> EvaluateResponse:
        if debug_trace is None:
            return response
        debug_trace["timings_ms"]["total"] = int((time.perf_counter() - started_total) * 1000)
        debug_trace["final_decision"] = {
            "status": response.status,
            "data_source": response.data_source,
            "fallback_used": response.fallback_used,
            "primary_status": response.primary_status,
            "primary_match_confidence": response.primary_match_confidence,
            "primary_data_source": response.primary_data_source,
            "fallback_data_source": response.fallback_data_source,
            "match_confidence": response.match_confidence,
            "matched_question_id": response.matched_question_id,
        }
        return EvaluateResponse(
            status=response.status,
            matched_question_text=response.matched_question_text,
            student_answer=response.student_answer,
            score=response.score,
            score_percent=response.score_percent,
            grade_label=response.grade_label,
            feedback=response.feedback,
            missing_points=list(response.missing_points),
            expected_points=list(response.expected_points),
            source_paper_reference=response.source_paper_reference,
            marking_scheme_answer=response.marking_scheme_answer,
            page_number=response.page_number,
            match_confidence=response.match_confidence,
            matched_question_id=response.matched_question_id,
            data_source=response.data_source,
            fallback_used=response.fallback_used,
            primary_status=response.primary_status,
            primary_match_confidence=response.primary_match_confidence,
            primary_data_source=response.primary_data_source,
            fallback_data_source=response.fallback_data_source,
            debug_trace=debug_trace,
            top_alternatives=list(response.top_alternatives),
        )


def evaluate_request(
    request: EvaluateRequest,
    *,
    service: Optional[OALevelEvaluatorService] = None,
    debug: bool = False,
) -> EvaluateResponse:
    """Small adapter for easy API integration."""
    active = service or OALevelEvaluatorService()
    return active.evaluate(request, debug=debug)
