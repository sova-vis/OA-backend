"""FastAPI transport for O/A Levels typed evaluator."""

from __future__ import annotations

import base64
import logging
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .content_normalization import (
    classify_content_type,
    classify_question_family,
    is_mathematics_subject,
    normalize_content_text_result,
    strip_leading_answer_label,
    strip_leading_question_label,
)
from .debug_persist import save_debug_run
from .mode_a_oa_extraction import (
    SUPPORTED_UPLOAD_CONTENT_TYPES,
    compute_mode_a_request_id,
    derive_recovery_reason_codes,
    extract_mode_a_document,
    map_mode_a_extraction_error,
    recovery_applied,
    serialize_oa_extraction,
)
from .schemas import EvaluateRequest, EvaluateResponse
from .service import OALevelEvaluatorService

logger = logging.getLogger(__name__)
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


class EvaluateRequestModel(BaseModel):
    question: str = Field(..., min_length=1)
    student_answer: str = Field(..., min_length=1)
    subject: Optional[str] = None
    year: Optional[int] = None
    session: Optional[str] = None
    paper: Optional[str] = None
    variant: Optional[str] = None
    question_id: Optional[str] = None
    debug: bool = False


class MatchAlternativeModel(BaseModel):
    question_id: str
    match_confidence: float
    question_text: str
    source_paper_reference: str


class EvaluateResponseModel(BaseModel):
    status: str
    matched_question_text: str
    student_answer: str
    score: float
    score_percent: float
    grade_label: str
    feedback: str
    missing_points: list[str]
    expected_points: list[str]
    source_paper_reference: str
    marking_scheme_answer: str
    page_number: Optional[int]
    match_confidence: float
    matched_question_id: Optional[str]
    data_source: str
    fallback_used: bool
    primary_status: str
    primary_match_confidence: float
    primary_data_source: str
    fallback_data_source: Optional[str]
    debug_trace: Optional[dict] = None
    top_alternatives: list[MatchAlternativeModel]


class ModeAQuestionCandidateModel(BaseModel):
    text: str
    source: str
    source_region: str
    source_variant: str
    retrieval_match_confidence: float
    structure_score: float
    answer_consistency_score: float
    final_score: float
    selected: bool
    family_label: Optional[str] = None
    ocr_distance: float = 0.0
    hard_rejected: bool = False
    rejection_reason_codes: list[str] = Field(default_factory=list)
    answer_signal_score: float = 0.0


class ModeAPreviewResponseModel(BaseModel):
    request_id: str
    extracted_question_text: str
    extracted_student_answer: str
    normalized_question_text: str
    normalized_student_answer: str
    canonical_question_text: Optional[str] = None
    canonical_student_answer: Optional[str] = None
    math_extraction_profile_used: bool = False
    vision_confidence: float
    vision_warnings: list[str]
    match_confidence: float
    top1_top2_margin: float
    auto_accept_eligible: bool
    auto_accept_reason: str
    content_type: Optional[str] = None
    recovery_applied: bool = False
    recovery_reason_codes: list[str] = Field(default_factory=list)
    question_candidates: list[ModeAQuestionCandidateModel] = Field(default_factory=list)
    top_alternatives: list[MatchAlternativeModel]
    debug_run_id: Optional[str] = None


class ModeAConfirmRequestModel(BaseModel):
    question_text: str = Field(..., min_length=1)
    student_answer: str = Field(default="")
    subject: Optional[str] = None
    year: Optional[int] = None
    session: Optional[str] = None
    paper: Optional[str] = None
    variant: Optional[str] = None
    question_id: Optional[str] = None
    debug: bool = False


def _to_response_payload(response: EvaluateResponse) -> dict:
    payload = asdict(response)
    payload["top_alternatives"] = [asdict(alt) for alt in response.top_alternatives]
    return payload


def _request_payload(request: EvaluateRequestModel) -> dict:
    if hasattr(request, "model_dump"):
        return request.model_dump()  # type: ignore[return-value]
    return request.dict()  # type: ignore[no-any-return]


def _normalize_mode_a_pair(
    *,
    question_text: str,
    student_answer: str,
    subject: Optional[str],
) -> Tuple[str, Any, Any]:
    stripped_question = strip_leading_question_label(question_text or "")
    stripped_answer = strip_leading_answer_label(student_answer or "")
    content_type = classify_content_type(stripped_question, stripped_answer, subject=subject)
    question_result = normalize_content_text_result(
        stripped_question,
        content_type=content_type,
        subject=subject,
    )
    answer_result = normalize_content_text_result(
        stripped_answer,
        content_type=content_type,
        subject=subject,
    )
    return content_type, question_result, answer_result


def _evaluate_mode_a_candidate(
    *,
    evaluator_service: OALevelEvaluatorService,
    question_text: str,
    student_answer_text: str,
    subject: Optional[str],
    year: Optional[int],
    session: Optional[str],
    paper: Optional[str],
    variant: Optional[str],
    question_id: Optional[str],
    debug: bool,
    direct_question_id_allowed: bool = False,
) -> dict:
    effective_student_answer = student_answer_text if str(student_answer_text or "").strip() else "[no_answer]"
    effective_question_id = question_id if direct_question_id_allowed else None
    evaluate_request = EvaluateRequest(
        question=question_text,
        student_answer=effective_student_answer,
        subject=subject,
        year=year,
        session=session,
        paper=paper,
        variant=variant,
        question_id=effective_question_id,
    )
    return _to_response_payload(
        evaluator_service.evaluate(
            evaluate_request,
            debug=bool(debug),
            use_answer_hint=False,
            direct_question_id_allowed=direct_question_id_allowed,
        )
    )


def _top1_top2_margin(top_alternatives: List[dict]) -> float:
    if len(top_alternatives) < 2:
        return 1.0 if top_alternatives else 0.0
    top1 = float(top_alternatives[0].get("match_confidence") or 0.0)
    top2 = float(top_alternatives[1].get("match_confidence") or 0.0)
    return max(0.0, top1 - top2)


def _selected_question_candidate(
    *,
    question_text: str,
    content_type: str,
    source_variant: str,
    match_confidence: float,
    vision_confidence: float,
) -> dict[str, Any]:
    return {
        "text": question_text,
        "source": "raw",
        "source_region": "document",
        "source_variant": source_variant,
        "retrieval_match_confidence": round(float(match_confidence), 4),
        "structure_score": round(float(vision_confidence), 4),
        "answer_consistency_score": 0.0,
        "final_score": round(float(vision_confidence), 4),
        "selected": True,
        "family_label": classify_question_family(question_text, content_type=content_type),
        "ocr_distance": 0.0,
        "hard_rejected": False,
        "rejection_reason_codes": [],
        "answer_signal_score": 0.0,
    }


def create_app(service: Optional[OALevelEvaluatorService] = None) -> FastAPI:
    evaluator_service = service or OALevelEvaluatorService()
    warmup_done = threading.Event()
    warmup_started_at = time.time()
    warmup_finished_at: Optional[float] = None
    warmup_error: Optional[str] = None

    @asynccontextmanager
    async def _lifespan(app: FastAPI):  # noqa: ARG001
        def _warmup_job() -> None:
            nonlocal warmup_error, warmup_finished_at
            try:
                logger.info("Warmup started (embedder + primary index build/load)...")
                evaluator_service.warmup()
                logger.info("Warmup complete (embedder + primary index).")
            except Exception as exc:
                warmup_error = str(exc)
                logger.exception("Warmup failed: %s", exc)
            finally:
                warmup_finished_at = time.time()
                warmup_done.set()

        threading.Thread(target=_warmup_job, name="oa-warmup", daemon=True).start()
        logger.info("Startup complete; warmup running in background.")
        yield

    app = FastAPI(title="O/A Levels Evaluator API", version="phase2", lifespan=_lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:3001",
            "http://127.0.0.1:3001",
        ],
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    static_dir = FRONTEND_DIR / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

        @app.get("/", include_in_schema=False)
        def frontend_index() -> FileResponse:
            return FileResponse(FRONTEND_DIR / "index.html")

    @app.get(
        "/oa-level/health",
        tags=["O/A Levels"],
        summary="Liveness check",
    )
    def health() -> dict:
        return {"status": "ok", "service": "oa-levels-evaluator"}

    @app.get(
        "/oa-level/ready",
        tags=["O/A Levels"],
        summary="Readiness check (warmup status)",
    )
    def ready() -> JSONResponse:
        payload = {
            "ready": warmup_done.is_set(),
            "warmup_done": warmup_done.is_set(),
            "warmup_error": warmup_error,
            "warmup_started_at": warmup_started_at,
            "warmup_finished_at": warmup_finished_at,
        }
        if warmup_done.is_set():
            return JSONResponse(status_code=200, content=payload)
        return JSONResponse(status_code=503, content=payload)

    @app.post(
        "/oa-level/evaluate",
        response_model=EvaluateResponseModel,
        tags=["O/A Levels"],
        summary="Evaluate typed question (Mode B)",
    )
    def evaluate_endpoint(request: EvaluateRequestModel) -> EvaluateResponseModel:
        started = time.perf_counter()
        payload = _request_payload(request)
        debug_enabled = bool(payload.pop("debug", False))
        evaluate_request = EvaluateRequest(**payload)
        result = evaluator_service.evaluate(evaluate_request, debug=debug_enabled)
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        answer_text = request.student_answer or ""
        answer_preview = answer_text[:48].replace("\n", " ")
        logger.info(
            "oa_eval status=%s source=%s primary_source=%s fallback_source=%s "
            "fallback_used=%s confidence=%.4f duration_ms=%d answer_len=%d answer_preview=%r",
            result.status,
            result.data_source,
            result.primary_data_source,
            result.fallback_data_source,
            result.fallback_used,
            result.match_confidence,
            elapsed_ms,
            len(answer_text),
            answer_preview,
        )
        return _to_response_payload(result)

    @app.post(
        "/oa-level/evaluate-from-image/preview",
        response_model=ModeAPreviewResponseModel,
        tags=["O/A Levels"],
        summary="Preview extraction from uploaded PDF/image (Mode A)",
    )
    async def evaluate_from_image_preview_endpoint(
        file: UploadFile = File(...),
        subject: Optional[str] = Form(None),
        year: Optional[int] = Form(None),
        session: Optional[str] = Form(None),
        paper: Optional[str] = Form(None),
        variant: Optional[str] = Form(None),
        question_id: Optional[str] = Form(None),
        page_number: int = Form(1),
        debug: bool = Form(False),
    ) -> ModeAPreviewResponseModel:
        if file is None:
            raise HTTPException(status_code=400, detail="file is required.")
        if page_number < 1:
            raise HTTPException(status_code=400, detail="page_number must be >= 1.")

        content_type = (file.content_type or "").lower()
        if content_type not in SUPPORTED_UPLOAD_CONTENT_TYPES:
            raise HTTPException(status_code=400, detail=f"Unsupported content-type: {content_type!r}")

        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Empty upload.")
        if len(raw) > 20 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large (max 20MB).")

        debug_payload: dict[str, Any] = {
            "endpoint": "/oa-level/evaluate-from-image/preview",
            "filename": file.filename,
            "content_type": content_type,
            "size_bytes": len(raw),
            "page_number": page_number,
            "filters": {
                "subject": subject,
                "year": year,
                "session": session,
                "paper": paper,
                "variant": variant,
                "question_id": question_id,
            },
        }
        incoming_question_id = str(question_id or "").strip() or None
        preview_question_id: Optional[str] = None
        question_id_ignored_for_preview = incoming_question_id is not None
        debug_payload["routing"] = {
            "incoming_question_id": incoming_question_id,
            "effective_question_id": preview_question_id,
            "question_id_ignored_for_preview": question_id_ignored_for_preview,
        }

        request_id = compute_mode_a_request_id(raw, content_type=content_type, page_number=page_number)
        try:
            extraction = extract_mode_a_document(
                raw_bytes=raw,
                content_type=content_type,
                filename=file.filename,
                page_number=page_number,
            )
        except Exception as exc:
            status_code, detail = map_mode_a_extraction_error(exc)
            raise HTTPException(status_code=status_code, detail=detail) from exc

        extracted_question = strip_leading_question_label(extraction.question_raw)
        extracted_answer = strip_leading_answer_label(extraction.answer_raw)
        normalized_content_type, normalized_question, normalized_answer = _normalize_mode_a_pair(
            question_text=extracted_question,
            student_answer=extracted_answer,
            subject=subject,
        )
        probe_payload = _evaluate_mode_a_candidate(
            evaluator_service=evaluator_service,
            question_text=normalized_question.display_text,
            student_answer_text=normalized_answer.display_text,
            subject=subject,
            year=year,
            session=session,
            paper=paper,
            variant=variant,
            question_id=preview_question_id,
            debug=bool(debug),
            direct_question_id_allowed=False,
        )

        selected_top_alts = list(probe_payload.get("top_alternatives") or [])
        match_conf = float(probe_payload.get("match_confidence") or 0.0)
        margin = _top1_top2_margin(selected_top_alts)
        vision_conf = float(extraction.confidence.overall)
        vision_warnings = [flag.code for flag in extraction.flags]
        recovery_codes = derive_recovery_reason_codes(extraction)

        vision_gate = vision_conf >= float(evaluator_service.config.vision_auto_accept_confidence)
        if bool(evaluator_service.config.vision_auto_accept_requires_no_warnings) and vision_warnings:
            vision_gate = False
        match_gate = (
            match_conf >= float(evaluator_service.config.match_auto_accept_confidence)
            and margin >= float(evaluator_service.config.match_auto_accept_margin)
        )
        auto_ok = bool(vision_gate and match_gate and normalized_question.display_text)
        reason = (
            "auto_accept_ok"
            if auto_ok
            else (
                f"blocked:vision_gate={vision_gate},match_gate={match_gate},"
                f"warnings={','.join(vision_warnings) if vision_warnings else 'none'}"
            )
        )

        diagnostics = extraction.diagnostics
        source_variant = diagnostics.selected_variant if diagnostics is not None else "original"
        question_candidates = [
            _selected_question_candidate(
                question_text=normalized_question.display_text,
                content_type=normalized_content_type,
                source_variant=source_variant,
                match_confidence=match_conf,
                vision_confidence=vision_conf,
            )
        ]

        preview_payload = {
            "request_id": request_id,
            "extracted_question_text": extracted_question,
            "extracted_student_answer": extracted_answer,
            "normalized_question_text": normalized_question.display_text,
            "normalized_student_answer": normalized_answer.display_text,
            "canonical_question_text": normalized_question.canonical_text,
            "canonical_student_answer": normalized_answer.canonical_text,
            "math_extraction_profile_used": bool(is_mathematics_subject(subject)),
            "vision_confidence": vision_conf,
            "vision_warnings": vision_warnings,
            "match_confidence": match_conf,
            "top1_top2_margin": margin,
            "auto_accept_eligible": auto_ok,
            "auto_accept_reason": reason,
            "content_type": normalized_content_type,
            "recovery_applied": recovery_applied(extraction),
            "recovery_reason_codes": recovery_codes,
            "question_candidates": question_candidates,
            "top_alternatives": selected_top_alts,
            "debug_run_id": None,
        }

        debug_payload["request_id"] = request_id
        if getattr(evaluator_service.config, "debug_store_image", False):
            debug_payload["upload"] = {
                "content_type": content_type,
                "base64": base64.b64encode(raw).decode("ascii"),
            }
        debug_payload["oa_extraction"] = serialize_oa_extraction(extraction)
        debug_payload["normalization"] = {
            "selected_question": {
                "display_text": normalized_question.display_text,
                "canonical_text": normalized_question.canonical_text,
                "matcher_text": normalized_question.matcher_text,
                "content_type": normalized_content_type,
            },
            "selected_student_answer": {
                "display_text": normalized_answer.display_text,
                "canonical_text": normalized_answer.canonical_text,
                "matcher_text": normalized_answer.matcher_text,
                "content_type": normalized_content_type,
            },
        }
        debug_payload["preview"] = preview_payload
        debug_payload["retrieval_policy"] = {
            "mode": "question_only",
            "use_answer_hint": False,
            "direct_question_id_allowed": False,
            "question_id_ignored_for_preview": question_id_ignored_for_preview,
            "math_canonical_enabled": bool(is_mathematics_subject(subject)),
            "question_matcher_mode": "canonical" if normalized_question.canonical_text else "display_fallback",
            "answer_matcher_mode": "canonical" if normalized_answer.canonical_text else "display_fallback",
        }
        debug_payload["probe_response"] = {
            "match_confidence": match_conf,
            "top1_top2_margin": margin,
            "top_alternatives": selected_top_alts,
        }
        if debug and probe_payload.get("debug_trace") is not None:
            debug_payload["debug_trace"] = probe_payload.get("debug_trace")

        try:
            run_id, out_path = save_debug_run(debug_payload, root=evaluator_service.config.debug_runs_dir)
            logger.info("mode_a_debug_saved run_id=%s path=%s", run_id, str(out_path))
            preview_payload["debug_run_id"] = run_id
        except Exception as exc:  # pragma: no cover
            logger.warning("mode_a_debug_save_failed error=%s", str(exc))
        return preview_payload

    @app.post(
        "/oa-level/evaluate-from-image/confirm",
        response_model=EvaluateResponseModel,
        tags=["O/A Levels"],
        summary="Confirm/edit extracted text and run final grading (Mode A)",
    )
    def evaluate_from_image_confirm_endpoint(request: ModeAConfirmRequestModel) -> EvaluateResponseModel:
        _, normalized_question, normalized_answer = _normalize_mode_a_pair(
            question_text=request.question_text or "",
            student_answer=request.student_answer or "",
            subject=request.subject,
        )
        evaluate_request = EvaluateRequest(
            question=normalized_question.display_text,
            student_answer=normalized_answer.display_text,
            subject=request.subject,
            year=request.year,
            session=request.session,
            paper=request.paper,
            variant=request.variant,
            question_id=request.question_id,
        )
        result = evaluator_service.evaluate(evaluate_request, debug=bool(request.debug), use_answer_hint=False)
        return _to_response_payload(result)

    @app.post(
        "/oa-level/evaluate-from-image",
        response_model=EvaluateResponseModel,
        tags=["O/A Levels"],
        summary="Legacy Mode A evaluate endpoint (auto confirm)",
    )
    async def evaluate_from_image_endpoint_legacy(
        file: UploadFile = File(...),
        subject: Optional[str] = Form(None),
        year: Optional[int] = Form(None),
        session: Optional[str] = Form(None),
        paper: Optional[str] = Form(None),
        variant: Optional[str] = Form(None),
        question_id: Optional[str] = Form(None),
        page_number: int = Form(1),
        debug: bool = Form(False),
    ) -> EvaluateResponseModel:
        preview = await evaluate_from_image_preview_endpoint(
            file=file,
            subject=subject,
            year=year,
            session=session,
            paper=paper,
            variant=variant,
            question_id=None,
            page_number=page_number,
            debug=debug,
        )
        confirm_request = ModeAConfirmRequestModel(
            question_text=preview["normalized_question_text"],
            student_answer=preview["normalized_student_answer"],
            subject=subject,
            year=year,
            session=session,
            paper=paper,
            variant=variant,
            question_id=None,
            debug=debug,
        )
        result = evaluate_from_image_confirm_endpoint(confirm_request)
        if not preview.get("auto_accept_eligible", False):
            result["status"] = "review_required"
            result["feedback"] = (
                (result.get("feedback") or "").strip()
                + f"\n\n[Mode A] Preview requires confirmation: {preview.get('auto_accept_reason', 'gate_failed')}."
            ).strip()
        return result

    return app


app = create_app()
