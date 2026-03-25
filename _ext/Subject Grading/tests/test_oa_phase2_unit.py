import json
import tempfile
import unittest
from pathlib import Path
from typing import Iterable, List, Optional
from unittest.mock import patch

from oa_main_pipeline.config import PipelineConfig
from oa_main_pipeline.dataset_repository import filter_question_records
from oa_main_pipeline.fallback_repository import FallbackDatasetRepository
from oa_main_pipeline.question_matcher import MatchResult
from oa_main_pipeline.schemas import EvaluateRequest, QuestionRecord
from oa_main_pipeline.search_index import SearchResult
from oa_main_pipeline.service import OALevelEvaluatorService

_NO_GROK_CONFIG = PipelineConfig(use_grok_grading=False, source_priority="o_level_json_first")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _record(
    *,
    question_id: str,
    subject: str = "Chemistry 1011",
    year: int = 2016,
    session: str = "May_June",
    paper: str = "Paper_1",
    variant: str = "Variant_2",
    question_number: str = "1",
    sub_question: Optional[str] = None,
    question_text: str = "What is the formula of water?",
    marking_scheme_answer: str = "H2O",
    page_number: Optional[int] = 2,
) -> QuestionRecord:
    return QuestionRecord(
        question_id=question_id,
        subject=subject,
        year=year,
        session=session,
        paper=paper,
        variant=variant,
        question_number=question_number,
        sub_question=sub_question,
        question_text=question_text,
        marking_scheme_answer=marking_scheme_answer,
        page_number=page_number,
        source_paper_reference=f"{subject}/{year}/{session}/{paper}/{variant}",
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


class _ExplodingRepo:
    def get_records(self) -> List[QuestionRecord]:
        raise AssertionError("Fallback repository should not be called for this test.")

    def filter_records(self, records: Iterable[QuestionRecord], request: EvaluateRequest) -> List[QuestionRecord]:
        raise AssertionError("Fallback repository should not be called for this test.")

    def get_by_question_id(self, question_id: str) -> Optional[QuestionRecord]:
        raise AssertionError("Fallback repository should not be called for this test.")


class OAPhase2UnitTests(unittest.TestCase):
    def test_fallback_repository_loads_valid_and_skips_malformed_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "O_LEVEL_JSON"
            _write_json(
                root / "Chemistry 1011" / "2015-2019.json",
                {
                    "2016": {
                        "May_June": {
                            "Paper_1": {
                                "Variant_2": [
                                    {
                                        "question_number": 1,
                                        "sub_question": None,
                                        "question_text": "Which option is correct?",
                                        "marking_scheme": "D",
                                        "page_number": 3,
                                    },
                                    {
                                        "question_number": "",
                                        "question_text": "bad row missing number",
                                        "marking_scheme": "A",
                                    },
                                    {
                                        "question_number": 2,
                                        "question_text": "",
                                        "marking_scheme": "B",
                                    },
                                ]
                            }
                        }
                    }
                },
            )
            bad_path = root / "Chemistry 1011" / "bad.json"
            bad_path.write_text("{not-valid-json", encoding="utf-8")

            repo = FallbackDatasetRepository(PipelineConfig(fallback_root=root))
            records = repo.get_records()
            self.assertEqual(len(records), 1)
            self.assertTrue(records[0].question_id.startswith("fallback|Chemistry 1011|2016|May_June|Paper_1|Variant_2|1"))

            scoped = repo.filter_records(
                records,
                EvaluateRequest(
                    question="Which option is correct?",
                    student_answer="D",
                    subject="Chemistry 1011",
                    year=2016,
                    session="May_June",
                    paper="Paper_1",
                    variant="Variant_2",
                ),
            )
            self.assertEqual(len(scoped), 1)

    def test_service_primary_accepted_skips_fallback(self):
        primary = _InMemoryRepo(
            [_record(question_id="fallback|chem|2016|mj|p1|v2|1", marking_scheme_answer="H2O")]
        )
        service = OALevelEvaluatorService(
            repository=_ExplodingRepo(),  # type: ignore[arg-type]
            fallback_repository=primary,  # type: ignore[arg-type]
            config=_NO_GROK_CONFIG,
        )
        response = service.evaluate(
            EvaluateRequest(
                question="What is the formula of water?",
                student_answer="H2O",
            )
        )
        self.assertEqual(response.status, "accepted")
        self.assertEqual(response.data_source, "o_level_json")
        self.assertFalse(response.fallback_used)
        self.assertEqual(response.primary_status, "accepted")
        self.assertEqual(response.primary_data_source, "o_level_json")
        self.assertEqual(response.fallback_data_source, "oa_main_dataset")

    def test_service_primary_review_required_skips_fallback(self):
        primary = _InMemoryRepo(
            [_record(question_id="fallback|chem|2016|mj|p1|v2|1", marking_scheme_answer="H2O")]
        )
        service = OALevelEvaluatorService(
            repository=_ExplodingRepo(),  # type: ignore[arg-type]
            fallback_repository=primary,  # type: ignore[arg-type]
            config=_NO_GROK_CONFIG,
        )
        response = service.evaluate(
            EvaluateRequest(
                question="water formula",
                student_answer="H2O",
            )
        )
        self.assertEqual(response.status, "review_required")
        self.assertEqual(response.data_source, "o_level_json")
        self.assertFalse(response.fallback_used)
        self.assertEqual(response.primary_status, "review_required")
        self.assertEqual(response.primary_data_source, "o_level_json")
        self.assertEqual(response.fallback_data_source, "oa_main_dataset")

    def test_service_fallback_recovers_when_primary_fails(self):
        primary = _InMemoryRepo([])
        fallback = _InMemoryRepo(
            [
                _record(
                    question_id="primary|chem|2016|mj|p1|v2|1",
                    question_text="Which option is correct?",
                    marking_scheme_answer="D",
                )
            ]
        )
        service = OALevelEvaluatorService(
            repository=fallback,
            fallback_repository=primary,  # type: ignore[arg-type]
            config=_NO_GROK_CONFIG,
        )
        response = service.evaluate(
            EvaluateRequest(
                question="Which option is correct?",
                student_answer="D",
            )
        )
        self.assertEqual(response.status, "accepted")
        self.assertEqual(response.data_source, "oa_main_dataset")
        self.assertTrue(response.fallback_used)
        self.assertEqual(response.primary_status, "failed")
        self.assertEqual(response.primary_match_confidence, 0.0)
        self.assertEqual(response.primary_data_source, "o_level_json")
        self.assertEqual(response.fallback_data_source, "oa_main_dataset")

    def test_service_failed_when_both_primary_and_fallback_fail(self):
        service = OALevelEvaluatorService(
            repository=_InMemoryRepo([]),
            fallback_repository=_InMemoryRepo([]),  # type: ignore[arg-type]
            config=_NO_GROK_CONFIG,
        )
        response = service.evaluate(
            EvaluateRequest(
                question="Unknown prompt with no dataset match",
                student_answer="Any answer",
            )
        )
        self.assertEqual(response.status, "failed")
        self.assertEqual(response.data_source, "oa_main_dataset")
        self.assertTrue(response.fallback_used)
        self.assertEqual(response.primary_status, "failed")
        self.assertEqual(response.primary_data_source, "o_level_json")
        self.assertEqual(response.fallback_data_source, "oa_main_dataset")

    def test_service_ignores_placeholder_optional_fields(self):
        primary = _InMemoryRepo(
            [
                _record(
                    question_id="fallback|chem|2016|mj|p1|v2|1",
                    question_text="Which option is correct?",
                    marking_scheme_answer="D",
                )
            ]
        )
        service = OALevelEvaluatorService(
            repository=_ExplodingRepo(),  # type: ignore[arg-type]
            fallback_repository=primary,  # type: ignore[arg-type]
            config=_NO_GROK_CONFIG,
        )
        response = service.evaluate(
            EvaluateRequest(
                question="Which option is correct?",
                student_answer="D",
                subject="Chemistry 1011",
                year=0,
                session="string",
                paper="string",
                variant="string",
                question_id="string",
            )
        )
        self.assertEqual(response.status, "accepted")
        self.assertEqual(response.data_source, "o_level_json")
        self.assertFalse(response.fallback_used)
        self.assertEqual(response.primary_data_source, "o_level_json")
        self.assertEqual(response.fallback_data_source, "oa_main_dataset")

    def test_service_debug_trace_for_primary_only(self):
        primary = _InMemoryRepo(
            [
                _record(
                    question_id="fallback|chem|2016|mj|p1|v2|1",
                    question_text="Which option is correct?",
                    marking_scheme_answer="D",
                )
            ]
        )
        service = OALevelEvaluatorService(
            repository=_ExplodingRepo(),  # type: ignore[arg-type]
            fallback_repository=primary,  # type: ignore[arg-type]
            config=_NO_GROK_CONFIG,
        )
        response = service.evaluate(
            EvaluateRequest(question="Which option is correct?", student_answer="D"),
            debug=True,
        )
        self.assertIsNotNone(response.debug_trace)
        trace = response.debug_trace or {}
        self.assertEqual(trace.get("final_decision", {}).get("data_source"), "o_level_json")
        stages = trace.get("stages", [])
        self.assertTrue(any(stage.get("stage") == "primary_match" for stage in stages))
        self.assertIn("total", trace.get("timings_ms", {}))
        self.assertIn("grading", trace)
        self.assertIn("source", trace.get("grading", {}))
        self.assertEqual(trace.get("source_order", {}).get("primary_data_source"), "o_level_json")
        self.assertEqual(trace.get("source_order", {}).get("fallback_data_source"), "oa_main_dataset")
        search_entries = trace.get("search_diagnostics", [])
        self.assertTrue(search_entries)
        self.assertIn("query_embedding_ms", search_entries[0])
        self.assertIn("similarity_search_ms", search_entries[0])
        self.assertIn("top_search_candidates", search_entries[0])

    def test_service_question_id_direct_path_still_available_when_enabled(self):
        record = _record(
            question_id="fallback|chem|2016|mj|p1|v2|1",
            question_text="Which option is correct?",
            marking_scheme_answer="D",
        )
        primary = _InMemoryRepo([record])
        service = OALevelEvaluatorService(
            repository=_InMemoryRepo([]),
            fallback_repository=primary,  # type: ignore[arg-type]
            config=PipelineConfig(use_grok_grading=False, source_priority="o_level_json_first", enable_fallback=False),
        )
        with patch.object(service.search_index, "search", side_effect=AssertionError("search should not run")):
            response = service.evaluate(
                EvaluateRequest(
                    question="completely unrelated prompt",
                    student_answer="D",
                    question_id=record.question_id,
                ),
                debug=True,
            )
        self.assertEqual(response.status, "accepted")
        trace = response.debug_trace or {}
        self.assertTrue(trace.get("retrieval", {}).get("direct_question_id_allowed"))
        self.assertEqual(trace.get("search_diagnostics", [{}])[0].get("search_method"), "question_id_direct")

    def test_service_can_disable_direct_question_id_lookup(self):
        record = _record(
            question_id="fallback|chem|2016|mj|p1|v2|1",
            question_text="Which option is correct?",
            marking_scheme_answer="D",
        )
        primary = _InMemoryRepo([record])
        service = OALevelEvaluatorService(
            repository=_InMemoryRepo([]),
            fallback_repository=primary,  # type: ignore[arg-type]
            config=PipelineConfig(use_grok_grading=False, source_priority="o_level_json_first", enable_fallback=False),
        )
        with patch.object(
            service.search_index,
            "search",
            return_value=SearchResult(
                match_result=MatchResult(
                    status="failed",
                    match_confidence=0.0,
                    best_record=None,
                    top_alternatives=[],
                ),
                debug={
                    "search_method": "embedding",
                    "final_search_score": 0.0,
                    "top_search_candidates": [],
                },
            ),
        ) as search_mock:
            response = service.evaluate(
                EvaluateRequest(
                    question="completely unrelated prompt",
                    student_answer="D",
                    question_id=record.question_id,
                ),
                debug=True,
                direct_question_id_allowed=False,
            )
        self.assertEqual(response.status, "failed")
        search_mock.assert_called_once()
        trace = response.debug_trace or {}
        self.assertFalse(trace.get("retrieval", {}).get("direct_question_id_allowed"))
        self.assertEqual(trace.get("search_diagnostics", [{}])[0].get("search_method"), "embedding")

    def test_service_source_priority_can_be_reversed(self):
        service = OALevelEvaluatorService(
            repository=_InMemoryRepo(
                [
                    _record(
                        question_id="primary|chem|2016|mj|p1|v2|1",
                        question_text="Which option is correct?",
                        marking_scheme_answer="D",
                    )
                ]
            ),
            fallback_repository=_ExplodingRepo(),  # type: ignore[arg-type]
            config=PipelineConfig(use_grok_grading=False, source_priority="oa_main_dataset_first"),
        )
        response = service.evaluate(
            EvaluateRequest(
                question="Which option is correct?",
                student_answer="D",
            )
        )
        self.assertEqual(response.status, "accepted")
        self.assertEqual(response.data_source, "oa_main_dataset")
        self.assertFalse(response.fallback_used)
        self.assertEqual(response.primary_data_source, "oa_main_dataset")
        self.assertEqual(response.fallback_data_source, "o_level_json")


if __name__ == "__main__":
    unittest.main()
