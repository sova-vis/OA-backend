import unittest
from pathlib import Path
from typing import Iterable, List, Optional
from unittest.mock import patch

try:
    from fastapi.testclient import TestClient
    from oa_main_pipeline.api import create_app
except Exception:  # pragma: no cover
    TestClient = None  # type: ignore[assignment]
    create_app = None  # type: ignore[assignment]

from oa_extraction import GrokAPIError
from oa_extraction.types import (
    ConfidenceScores,
    ExtractionDiagnostics,
    ExtractionResult,
    FlagSeverity,
    LineOCR,
    OCRCandidate,
    OCREngine,
    SubjectLabel,
    ValidationFlag,
)
from oa_main_pipeline.config import PipelineConfig
from oa_main_pipeline.dataset_repository import filter_question_records
from oa_main_pipeline.schemas import EvaluateRequest, QuestionRecord
from oa_main_pipeline.service import OALevelEvaluatorService


_NO_GROK_CONFIG = PipelineConfig(use_grok_grading=False, source_priority="o_level_json_first")


def _record(*, question_text: str, marking_scheme_answer: str) -> QuestionRecord:
    return QuestionRecord(
        question_id="fallback|1",
        subject="Chemistry 1011",
        year=2016,
        session="May_June",
        paper="Paper_1",
        variant="Variant_2",
        question_number="1",
        sub_question=None,
        question_text=question_text,
        marking_scheme_answer=marking_scheme_answer,
        page_number=2,
        source_paper_reference="Chemistry 1011/2016/May_June/Paper_1/Variant_2",
    )


class _InMemoryRepo:
    def __init__(self, records: List[QuestionRecord]) -> None:
        self._records = list(records)
        self._by_id = {r.question_id: r for r in self._records}

    def get_records(self) -> List[QuestionRecord]:
        return list(self._records)

    def filter_records(
        self,
        records: Iterable[QuestionRecord],
        request: EvaluateRequest,
    ) -> List[QuestionRecord]:
        return filter_question_records(records, request)

    def get_by_question_id(self, question_id: str) -> Optional[QuestionRecord]:
        return self._by_id.get(question_id)


def _candidate(full_text: str, *, engine: OCREngine = OCREngine.GROK, variant: str = "original") -> OCRCandidate:
    return OCRCandidate(
        engine=engine,
        variant=variant,
        full_text=full_text,
        lines=[LineOCR(page_number=1, line_index=1, text=full_text, confidence=0.95)],
        ocr_confidence=0.95,
        uncertain_spans=[],
        selection_score=0.95,
    )


def _extraction(question_text: str, answer_text: str) -> ExtractionResult:
    full_text = "\n".join(part for part in (question_text, answer_text) if part)
    flags = []
    if not answer_text:
        flags.append(
            ValidationFlag(
                code="missing_answer",
                severity=FlagSeverity.ERROR,
                message="Answer text could not be extracted.",
            )
        )
    return ExtractionResult(
        input_type="image",
        page_count=1,
        whole_text_raw=full_text,
        question_raw=question_text,
        answer_raw=answer_text,
        question_normalized=question_text,
        answer_normalized=answer_text,
        subject=SubjectLabel.OTHER,
        confidence=ConfidenceScores(ocr=0.95, split=0.95, classification=0.95, overall=0.95),
        flags=flags,
        needs_review=bool(flags),
        diagnostics=ExtractionDiagnostics(
            selected_ocr_engine=OCREngine.GROK,
            selected_variant="original",
            ocr_candidates=[_candidate(full_text)],
            disagreement_spans=[],
            repair_actions=[],
            selection_reasons=["Selected OCR candidate."],
            split_retry_applied=False,
        ),
    )


@unittest.skipIf(TestClient is None, "FastAPI TestClient is not available.")
class OAPhase5APITests(unittest.TestCase):
    def _client(self, *, question_text: str = "Which option is correct?") -> TestClient:
        service = OALevelEvaluatorService(
            repository=_InMemoryRepo([]),
            fallback_repository=_InMemoryRepo([_record(question_text=question_text, marking_scheme_answer="D")]),  # type: ignore[arg-type]
            config=_NO_GROK_CONFIG,
        )
        return TestClient(create_app(service))

    def test_endpoint_rejects_unsupported_content_type(self):
        client = self._client()
        resp = client.post(
            "/oa-level/evaluate-from-image",
            files={"file": ("x.txt", b"hello", "text/plain")},
            data={},
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Unsupported content-type", resp.text)

    def test_endpoint_returns_503_when_oa_extraction_fails(self):
        client = self._client()
        with patch(
            "oa_main_pipeline.api.extract_mode_a_document",
            side_effect=GrokAPIError("Grok extraction failed."),
        ):
            resp = client.post(
                "/oa-level/evaluate-from-image",
                files={"file": ("q.png", b"fake", "image/png")},
                data={},
            )
        self.assertEqual(resp.status_code, 503)

    def test_endpoint_success_path_extracts_then_evaluates(self):
        question_text = "Which option is correct?"
        client = self._client(question_text=question_text)
        with patch(
            "oa_main_pipeline.api.extract_mode_a_document",
            return_value=_extraction(question_text, "D"),
        ), patch(
            "oa_main_pipeline.api.save_debug_run",
            return_value=("runid", Path("DEBUG_RUNS/run.json")),
        ):
            resp = client.post(
                "/oa-level/evaluate-from-image",
                files={"file": ("q.png", b"fake", "image/png")},
                data={"page_number": "1", "debug": "true"},
            )
        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        self.assertEqual(payload["status"], "accepted")
        self.assertEqual(payload["marking_scheme_answer"], "D")
        self.assertEqual(payload["source_paper_reference"], "Chemistry 1011/2016/May_June/Paper_1/Variant_2")

    def test_endpoint_missing_answer_returns_review_required(self):
        question_text = "Which option is correct?"
        client = self._client(question_text=question_text)
        with patch(
            "oa_main_pipeline.api.extract_mode_a_document",
            return_value=_extraction(question_text, ""),
        ), patch(
            "oa_main_pipeline.api.save_debug_run",
            return_value=("runid", Path("DEBUG_RUNS/run.json")),
        ):
            resp = client.post(
                "/oa-level/evaluate-from-image",
                files={"file": ("q.png", b"fake", "image/png")},
                data={"page_number": "1", "debug": "true"},
            )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "review_required")


if __name__ == "__main__":
    unittest.main()
