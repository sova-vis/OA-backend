"""Repo-local OA-Extraction adapter for Mode A uploads."""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable


def _ensure_repo_local_oa_extraction() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    local_src = repo_root / "OA-Extraction" / "src"
    local_src_str = str(local_src)

    existing = sys.modules.get("oa_extraction")
    existing_path = Path(getattr(existing, "__file__", "")).resolve() if existing is not None else None
    if existing_path is not None and local_src not in existing_path.parents:
        for name in list(sys.modules):
            if name == "oa_extraction" or name.startswith("oa_extraction."):
                sys.modules.pop(name, None)

    if local_src_str not in sys.path:
        sys.path.insert(0, local_src_str)


_ensure_repo_local_oa_extraction()

from oa_extraction import ConfigurationError, GrokAPIError, InputDocumentError, OAExtractionError, extract_qa
from oa_extraction.types import ExtractionResult

SUPPORTED_UPLOAD_CONTENT_TYPES = {
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/jpg",
}

_CONTENT_TYPE_SUFFIX = {
    "application/pdf": ".pdf",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
}


def compute_mode_a_request_id(raw_bytes: bytes, *, content_type: str, page_number: int) -> str:
    digest = hashlib.sha256()
    digest.update(str(content_type or "").encode("utf-8"))
    digest.update(b"\0")
    digest.update(raw_bytes)
    digest.update(b"\0")
    digest.update(str(page_number).encode("ascii"))
    return digest.hexdigest()


def extract_mode_a_document(
    *,
    raw_bytes: bytes,
    content_type: str,
    filename: str | None = None,
    page_number: int = 1,
) -> ExtractionResult:
    suffix = _suffix_for_upload(filename=filename, content_type=content_type)
    temp_path = _write_temp_upload(raw_bytes, suffix=suffix)
    try:
        selected_page = int(page_number) if content_type == "application/pdf" else None
        return extract_qa(str(temp_path), page_number=selected_page)
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass


def map_mode_a_extraction_error(exc: Exception) -> tuple[int, str]:
    if isinstance(exc, InputDocumentError):
        return 400, str(exc)
    if isinstance(exc, (ConfigurationError, GrokAPIError, OAExtractionError)):
        return 503, str(exc)
    raise exc


def derive_recovery_reason_codes(result: ExtractionResult) -> list[str]:
    codes: list[str] = []
    diagnostics = result.diagnostics
    if diagnostics is not None and str(diagnostics.selected_ocr_engine) == "azure":
        codes.append("azure_fallback_selected")
    if diagnostics is not None and bool(diagnostics.split_retry_applied):
        codes.append("split_retry_applied")
    if diagnostics is not None and any(action.accepted for action in diagnostics.repair_actions):
        codes.append("targeted_repair_applied")
    codes.extend(flag.code for flag in result.flags)
    if result.needs_review:
        codes.append("needs_review")
    return _dedupe(codes)


def recovery_applied(result: ExtractionResult) -> bool:
    recovery_codes = set(derive_recovery_reason_codes(result))
    return bool(
        {
            "azure_fallback_selected",
            "split_retry_applied",
            "targeted_repair_applied",
        }
        & recovery_codes
    )


def serialize_oa_extraction(result: ExtractionResult) -> dict[str, Any]:
    diagnostics = result.diagnostics
    return {
        "input_type": result.input_type,
        "page_count": result.page_count,
        "subject": str(result.subject),
        "needs_review": bool(result.needs_review),
        "whole_text_raw": result.whole_text_raw,
        "question_raw": result.question_raw,
        "answer_raw": result.answer_raw,
        "question_normalized": result.question_normalized,
        "answer_normalized": result.answer_normalized,
        "confidence": result.confidence.model_dump(mode="json"),
        "flags": [flag.model_dump(mode="json") for flag in result.flags],
        "recovery_reason_codes": derive_recovery_reason_codes(result),
        "diagnostics": (
            {
                "selected_ocr_engine": str(diagnostics.selected_ocr_engine),
                "selected_variant": diagnostics.selected_variant,
                "split_retry_applied": bool(diagnostics.split_retry_applied),
                "ocr_candidates": [
                    {
                        "engine": str(candidate.engine),
                        "variant": candidate.variant,
                        "ocr_confidence": float(candidate.ocr_confidence),
                        "selection_score": float(candidate.selection_score or 0.0),
                        "line_count": len(candidate.lines),
                        "uncertain_span_count": len(candidate.uncertain_spans),
                    }
                    for candidate in diagnostics.ocr_candidates
                ],
                "disagreement_spans": [span.model_dump(mode="json") for span in diagnostics.disagreement_spans],
                "repair_actions": [action.model_dump(mode="json") for action in diagnostics.repair_actions],
                "selection_reasons": list(diagnostics.selection_reasons),
                "math_answer_refine": (
                    diagnostics.math_answer_refine.model_dump(mode="json")
                    if diagnostics.math_answer_refine is not None
                    else None
                ),
            }
            if diagnostics is not None
            else None
        ),
    }


def _suffix_for_upload(*, filename: str | None, content_type: str) -> str:
    if content_type in _CONTENT_TYPE_SUFFIX:
        return _CONTENT_TYPE_SUFFIX[content_type]
    suffix = Path(filename or "").suffix
    return suffix or ".bin"


def _write_temp_upload(raw_bytes: bytes, *, suffix: str) -> Path:
    fd, temp_path = tempfile.mkstemp(prefix="oa_mode_a_", suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw_bytes)
    except Exception:
        try:
            os.close(fd)
        except Exception:
            pass
        raise
    return Path(temp_path)


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        cleaned = str(value or "").strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        output.append(cleaned)
    return output
