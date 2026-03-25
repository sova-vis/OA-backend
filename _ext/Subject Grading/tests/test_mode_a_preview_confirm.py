import unittest
from pathlib import Path
from typing import Iterable, List, Optional
from unittest.mock import patch

from fastapi.testclient import TestClient

from oa_main_pipeline.api import create_app
from oa_extraction.types import (
    ConfidenceScores,
    ExtractionDiagnostics,
    ExtractionResult,
    FlagSeverity,
    LineOCR,
    OCRCandidate,
    OCREngine,
    RepairAction,
    SubjectLabel,
    ValidationFlag,
)
from oa_main_pipeline.config import PipelineConfig
from oa_main_pipeline.dataset_repository import filter_question_records
from oa_main_pipeline.schemas import EvaluateRequest, QuestionRecord
from oa_main_pipeline.service import OALevelEvaluatorService


def _record(*, question_text: str, marking_scheme_answer: str, subject: str = "Chemistry 1011") -> QuestionRecord:
    return QuestionRecord(
        question_id="fallback|1",
        subject=subject,
        year=2016,
        session="May_June",
        paper="Paper_1",
        variant="Variant_2",
        question_number="1",
        sub_question=None,
        question_text=question_text,
        marking_scheme_answer=marking_scheme_answer,
        page_number=2,
        source_paper_reference=f"{subject}/2016/May_June/Paper_1/Variant_2",
    )


class _InMemoryRepo:
    def __init__(self, records: List[QuestionRecord]) -> None:
        self._records = list(records)
        self._by_id = {r.question_id: r for r in self._records}

    def get_records(self) -> List[QuestionRecord]:
        return list(self._records)

    def filter_records(self, records: Iterable[QuestionRecord], request: EvaluateRequest) -> List[QuestionRecord]:
        return filter_question_records(records, request)

    def get_by_question_id(self, question_id: str) -> Optional[QuestionRecord]:
        return self._by_id.get(question_id)


def _candidate(
    *,
    engine: OCREngine = OCREngine.GROK,
    variant: str = "original",
    full_text: str,
    ocr_confidence: float = 0.95,
) -> OCRCandidate:
    return OCRCandidate(
        engine=engine,
        variant=variant,
        full_text=full_text,
        lines=[LineOCR(page_number=1, line_index=1, text=full_text, confidence=ocr_confidence)],
        ocr_confidence=ocr_confidence,
        uncertain_spans=[],
        selection_score=ocr_confidence,
    )


def _extraction_result(
    *,
    question_raw: str,
    answer_raw: str,
    overall_confidence: float = 0.95,
    flags: Optional[list[ValidationFlag]] = None,
    selected_engine: OCREngine = OCREngine.GROK,
    selected_variant: str = "original",
    split_retry_applied: bool = False,
    accepted_repairs: int = 0,
    needs_review: Optional[bool] = None,
    subject: SubjectLabel = SubjectLabel.OTHER,
) -> ExtractionResult:
    repair_actions = [
        RepairAction(
            page_number=1,
            line_index=index + 1,
            before_text="before",
            after_text="after",
            source="grok_repair",
            accepted=True,
            confidence=0.95,
            rationale="accepted repair",
        )
        for index in range(accepted_repairs)
    ]
    diagnostics = ExtractionDiagnostics(
        selected_ocr_engine=selected_engine,
        selected_variant=selected_variant,
        ocr_candidates=[
            _candidate(
                engine=selected_engine,
                variant=selected_variant,
                full_text="\n".join(part for part in (question_raw, answer_raw) if part),
                ocr_confidence=overall_confidence,
            )
        ],
        disagreement_spans=[],
        repair_actions=repair_actions,
        selection_reasons=["Selected OCR candidate."],
        split_retry_applied=split_retry_applied,
    )
    confidence = ConfidenceScores(
        ocr=overall_confidence,
        split=overall_confidence,
        classification=overall_confidence,
        overall=overall_confidence,
    )
    resolved_flags = flags or []
    return ExtractionResult(
        input_type="image",
        page_count=1,
        whole_text_raw="\n".join(part for part in (question_raw, answer_raw) if part),
        question_raw=question_raw,
        answer_raw=answer_raw,
        question_normalized=question_raw,
        answer_normalized=answer_raw,
        subject=subject,
        confidence=confidence,
        flags=resolved_flags,
        needs_review=(needs_review if needs_review is not None else bool(resolved_flags or overall_confidence < 0.92)),
        diagnostics=diagnostics,
    )


def _warning(code: str, message: str) -> ValidationFlag:
    return ValidationFlag(code=code, severity=FlagSeverity.WARNING, message=message)


class ModeAPreviewConfirmTests(unittest.TestCase):
    def _client(self, *, record: Optional[QuestionRecord] = None, config: Optional[PipelineConfig] = None) -> TestClient:
        cfg = config or PipelineConfig(use_grok_grading=False, source_priority="o_level_json_first")
        service = OALevelEvaluatorService(
            repository=_InMemoryRepo([]),
            fallback_repository=_InMemoryRepo([record or _record(question_text="Which option is correct?", marking_scheme_answer="D")]),  # type: ignore[arg-type]
            config=cfg,
        )
        return TestClient(create_app(service))

    def test_preview_blocks_auto_accept_on_low_extraction_confidence(self):
        client = self._client()
        extraction = _extraction_result(
            question_raw="Which option is correct?",
            answer_raw="D",
            overall_confidence=0.2,
            flags=[_warning("low_ocr_confidence", "OCR confidence below threshold.")],
            needs_review=True,
        )

        with patch("oa_main_pipeline.api.extract_mode_a_document", return_value=extraction), patch(
            "oa_main_pipeline.api.save_debug_run",
            return_value=("runid", Path("DEBUG_RUNS/run.json")),
        ):
            resp = client.post(
                "/oa-level/evaluate-from-image/preview",
                files={"file": ("q.png", b"fake", "image/png")},
                data={"page_number": "1"},
            )

        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        self.assertFalse(payload["auto_accept_eligible"])
        self.assertEqual(payload["vision_warnings"], ["low_ocr_confidence"])
        self.assertEqual(payload["normalized_question_text"], "Which option is correct?")
        self.assertEqual(len(payload["question_candidates"]), 1)

    def test_confirm_endpoint_uses_edited_values(self):
        cfg = PipelineConfig(use_grok_grading=False, source_priority="o_level_json_first")
        service = OALevelEvaluatorService(
            repository=_InMemoryRepo([]),
            fallback_repository=_InMemoryRepo([_record(question_text="Which option is correct?", marking_scheme_answer="D")]),  # type: ignore[arg-type]
            config=cfg,
        )
        client = TestClient(create_app(service))

        with patch.object(service, "evaluate", wraps=service.evaluate) as evaluate_mock:
            resp = client.post(
                "/oa-level/evaluate-from-image/confirm",
                json={"question_text": "Which option is correct?", "student_answer": "D", "subject": "Chemistry 1011", "debug": True},
            )

        self.assertEqual(resp.status_code, 200)
        self.assertFalse(evaluate_mock.call_args.kwargs["use_answer_hint"])
        self.assertEqual(resp.json()["marking_scheme_answer"], "D")

    def test_preview_passes_pdf_page_number_to_oa_extraction(self):
        client = self._client()
        extraction = _extraction_result(question_raw="Which option is correct?", answer_raw="D")

        with patch("oa_main_pipeline.api.extract_mode_a_document", return_value=extraction) as extract_mock, patch(
            "oa_main_pipeline.api.save_debug_run",
            return_value=("runid", Path("DEBUG_RUNS/run.json")),
        ):
            resp = client.post(
                "/oa-level/evaluate-from-image/preview",
                files={"file": ("q.pdf", b"%PDF-fake", "application/pdf")},
                data={"page_number": "2"},
            )

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(extract_mock.call_args.kwargs["page_number"], 2)
        self.assertEqual(extract_mock.call_args.kwargs["content_type"], "application/pdf")

    def test_preview_ignores_incoming_question_id_for_upload_flow(self):
        client = self._client()
        extraction = _extraction_result(question_raw="Which option is correct?", answer_raw="D")

        def _probe_payload(*, question_id: Optional[str], direct_question_id_allowed: bool, **_: object) -> dict:
            self.assertIsNone(question_id)
            self.assertFalse(direct_question_id_allowed)
            return {
                "match_confidence": 0.91,
                "top_alternatives": [
                    {
                        "question_id": "fallback|1",
                        "match_confidence": 0.91,
                        "question_text": "Which option is correct?",
                        "source_paper_reference": "Chemistry 1011/2016/May_June/Paper_1/Variant_2",
                    }
                ],
            }

        with patch("oa_main_pipeline.api.extract_mode_a_document", return_value=extraction), patch(
            "oa_main_pipeline.api._evaluate_mode_a_candidate",
            side_effect=_probe_payload,
        ) as probe_mock, patch(
            "oa_main_pipeline.api.save_debug_run",
            return_value=("runid", Path("DEBUG_RUNS/run.json")),
        ):
            resp = client.post(
                "/oa-level/evaluate-from-image/preview",
                files={"file": ("q.png", b"fake", "image/png")},
                data={
                    "page_number": "1",
                    "question_id": "main|Chemistry 1011|2018|Oct_Nov|Paper_4|Variant_2|3(c)(v)",
                },
            )

        self.assertEqual(resp.status_code, 200)
        self.assertGreaterEqual(probe_mock.call_count, 1)

    def test_preview_marks_azure_fallback_as_recovery(self):
        client = self._client()
        extraction = _extraction_result(
            question_raw="Solve log_3(x) + log_9(x) = 12",
            answer_raw="x = 6561",
            selected_engine=OCREngine.AZURE,
            selected_variant="original",
            flags=[_warning("potential_log_base_mismatch", "Possible log base mismatch.")],
            subject=SubjectLabel.MATH,
        )

        with patch("oa_main_pipeline.api.extract_mode_a_document", return_value=extraction), patch(
            "oa_main_pipeline.api.save_debug_run",
            return_value=("runid", Path("DEBUG_RUNS/run.json")),
        ):
            resp = client.post(
                "/oa-level/evaluate-from-image/preview",
                files={"file": ("q.png", b"fake", "image/png")},
                data={"page_number": "1", "subject": "Mathematics 1014"},
            )

        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        self.assertTrue(payload["recovery_applied"])
        self.assertIn("azure_fallback_selected", payload["recovery_reason_codes"])

    def test_preview_marks_split_retry_as_recovery(self):
        client = self._client()
        extraction = _extraction_result(
            question_raw="State Newton's second law",
            answer_raw="Force equals mass times acceleration",
            split_retry_applied=True,
        )

        with patch("oa_main_pipeline.api.extract_mode_a_document", return_value=extraction), patch(
            "oa_main_pipeline.api.save_debug_run",
            return_value=("runid", Path("DEBUG_RUNS/run.json")),
        ):
            resp = client.post(
                "/oa-level/evaluate-from-image/preview",
                files={"file": ("q.png", b"fake", "image/png")},
                data={"page_number": "1"},
            )

        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        self.assertTrue(payload["recovery_applied"])
        self.assertIn("split_retry_applied", payload["recovery_reason_codes"])

    def test_preview_marks_targeted_repair_as_recovery(self):
        client = self._client()
        extraction = _extraction_result(
            question_raw="Solve log_3(x) = 9",
            answer_raw="x = 3^2",
            accepted_repairs=1,
        )

        with patch("oa_main_pipeline.api.extract_mode_a_document", return_value=extraction), patch(
            "oa_main_pipeline.api.save_debug_run",
            return_value=("runid", Path("DEBUG_RUNS/run.json")),
        ):
            resp = client.post(
                "/oa-level/evaluate-from-image/preview",
                files={"file": ("q.png", b"fake", "image/png")},
                data={"page_number": "1"},
            )

        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        self.assertTrue(payload["recovery_applied"])
        self.assertIn("targeted_repair_applied", payload["recovery_reason_codes"])

    def test_legacy_mode_a_returns_review_required_when_preview_gate_fails(self):
        client = self._client(
            record=_record(
                question_text="Solve log_3(x) + log_9(x) = 12",
                marking_scheme_answer="x=6561",
                subject="Mathematics 1014",
            )
        )
        extraction = _extraction_result(
            question_raw="Q. Solve log_2x + log_2x =12",
            answer_raw="A. log_3x =8",
            overall_confidence=0.35,
            flags=[_warning("potential_subscript_loss", "Potential subscript loss.")],
            needs_review=True,
            subject=SubjectLabel.MATH,
        )

        with patch("oa_main_pipeline.api.extract_mode_a_document", return_value=extraction), patch(
            "oa_main_pipeline.api.save_debug_run",
            return_value=("runid", Path("DEBUG_RUNS/run.json")),
        ):
            resp = client.post(
                "/oa-level/evaluate-from-image",
                files={"file": ("q.png", b"fake", "image/png")},
                data={"page_number": "1", "subject": "Mathematics 1014"},
            )

        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        self.assertEqual(payload["status"], "review_required")
        self.assertIn("Preview requires confirmation", payload["feedback"])


if __name__ == "__main__":
    unittest.main()
