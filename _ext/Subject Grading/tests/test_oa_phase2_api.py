import unittest
from typing import Iterable, List, Optional

try:
    from fastapi.testclient import TestClient
    from oa_main_pipeline.api import create_app
except Exception:  # pragma: no cover
    TestClient = None  # type: ignore[assignment]
    create_app = None  # type: ignore[assignment]
from oa_main_pipeline.config import PipelineConfig
from oa_main_pipeline.dataset_repository import filter_question_records
from oa_main_pipeline.schemas import EvaluateRequest, QuestionRecord
from oa_main_pipeline.service import OALevelEvaluatorService

_NO_GROK_CONFIG = PipelineConfig(use_grok_grading=False, source_priority="o_level_json_first")


def _record(
    *,
    question_id: str,
    question_text: str,
    marking_scheme_answer: str,
) -> QuestionRecord:
    return QuestionRecord(
        question_id=question_id,
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


@unittest.skipIf(TestClient is None, "FastAPI TestClient is not available.")
class OAPhase2APITests(unittest.TestCase):
    def test_endpoint_primary_source_response(self):
        service = OALevelEvaluatorService(
            repository=_InMemoryRepo([]),
            fallback_repository=_InMemoryRepo(  # type: ignore[arg-type]
                [
                    _record(
                        question_id="fallback|1",
                        question_text="Which option is correct?",
                        marking_scheme_answer="D",
                    )
                ]
            ),
            config=_NO_GROK_CONFIG,
        )
        client = TestClient(create_app(service))
        resp = client.post(
            "/oa-level/evaluate",
            json={
                "question": "Which option is correct?",
                "student_answer": "D",
            },
        )
        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        self.assertEqual(payload["status"], "accepted")
        self.assertEqual(payload["data_source"], "o_level_json")
        self.assertFalse(payload["fallback_used"])
        self.assertEqual(payload["primary_status"], "accepted")
        self.assertEqual(payload["primary_data_source"], "o_level_json")
        self.assertEqual(payload["fallback_data_source"], "oa_main_dataset")
        self.assertIn("primary_match_confidence", payload)

    def test_endpoint_fallback_source_response(self):
        service = OALevelEvaluatorService(
            repository=_InMemoryRepo(
                [
                    _record(
                        question_id="primary|1",
                        question_text="Which option is correct?",
                        marking_scheme_answer="D",
                    )
                ]
            ),
            fallback_repository=_InMemoryRepo([]),  # type: ignore[arg-type]
            config=_NO_GROK_CONFIG,
        )
        client = TestClient(create_app(service))
        resp = client.post(
            "/oa-level/evaluate",
            json={
                "question": "Which option is correct?",
                "student_answer": "D",
            },
        )
        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        self.assertEqual(payload["status"], "accepted")
        self.assertEqual(payload["data_source"], "oa_main_dataset")
        self.assertTrue(payload["fallback_used"])
        self.assertEqual(payload["primary_status"], "failed")
        self.assertEqual(payload["primary_data_source"], "o_level_json")
        self.assertEqual(payload["fallback_data_source"], "oa_main_dataset")

    def test_endpoint_failed_when_both_sources_fail(self):
        service = OALevelEvaluatorService(
            repository=_InMemoryRepo([]),
            fallback_repository=_InMemoryRepo([]),  # type: ignore[arg-type]
            config=_NO_GROK_CONFIG,
        )
        client = TestClient(create_app(service))
        resp = client.post(
            "/oa-level/evaluate",
            json={
                "question": "Unrelated question",
                "student_answer": "No match expected",
            },
        )
        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["data_source"], "oa_main_dataset")
        self.assertTrue(payload["fallback_used"])
        self.assertEqual(payload["primary_data_source"], "o_level_json")
        self.assertEqual(payload["fallback_data_source"], "oa_main_dataset")

    def test_endpoint_validation_errors(self):
        service = OALevelEvaluatorService(
            repository=_InMemoryRepo([]),
            fallback_repository=_InMemoryRepo([]),  # type: ignore[arg-type]
            config=_NO_GROK_CONFIG,
        )
        client = TestClient(create_app(service))
        resp = client.post(
            "/oa-level/evaluate",
            json={"student_answer": "Missing question"},
        )
        self.assertEqual(resp.status_code, 422)

    def test_endpoint_accepts_placeholder_optional_fields(self):
        service = OALevelEvaluatorService(
            repository=_InMemoryRepo([]),
            fallback_repository=_InMemoryRepo(  # type: ignore[arg-type]
                [
                    _record(
                        question_id="fallback|1",
                        question_text="Which option is correct?",
                        marking_scheme_answer="D",
                    )
                ]
            ),
            config=_NO_GROK_CONFIG,
        )
        client = TestClient(create_app(service))
        resp = client.post(
            "/oa-level/evaluate",
            json={
                "question": "Which option is correct?",
                "student_answer": "D",
                "subject": "Chemistry 1011",
                "year": 0,
                "session": "string",
                "paper": "string",
                "variant": "string",
                "question_id": "string",
            },
        )
        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        self.assertEqual(payload["status"], "accepted")
        self.assertEqual(payload["data_source"], "o_level_json")

    def test_endpoint_returns_debug_trace_when_enabled(self):
        service = OALevelEvaluatorService(
            repository=_InMemoryRepo([]),
            fallback_repository=_InMemoryRepo(  # type: ignore[arg-type]
                [
                    _record(
                        question_id="fallback|1",
                        question_text="Which option is correct?",
                        marking_scheme_answer="D",
                    )
                ]
            ),
            config=_NO_GROK_CONFIG,
        )
        client = TestClient(create_app(service))
        resp = client.post(
            "/oa-level/evaluate",
            json={
                "question": "Which option is correct?",
                "student_answer": "D",
                "debug": True,
            },
        )
        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        self.assertIn("debug_trace", payload)
        self.assertIsNotNone(payload["debug_trace"])
        self.assertIn("stages", payload["debug_trace"])
        self.assertEqual(payload["primary_data_source"], "o_level_json")
        self.assertEqual(payload["fallback_data_source"], "oa_main_dataset")


if __name__ == "__main__":
    unittest.main()
