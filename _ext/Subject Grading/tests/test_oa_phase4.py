"""Phase 4: Separate pipeline module + O_LEVEL_MAIN_JSON as primary."""

import unittest
from typing import Iterable, List, Optional

from oa_main_pipeline.config import PipelineConfig
from oa_main_pipeline.dataset_repository import filter_question_records
from oa_main_pipeline.o_level_main_repository import MainJsonRepository
from oa_main_pipeline.schemas import EvaluateRequest, QuestionRecord
from oa_main_pipeline.service import OALevelEvaluatorService


def _record(
    *,
    question_id: str,
    question_text: str = "What is the formula of water?",
    marking_scheme_answer: str = "H2O",
) -> QuestionRecord:
    return QuestionRecord(
        question_id=question_id,
        subject="Mathematics 1014",
        year=2017,
        session="May_June",
        paper="Paper_1",
        variant="Variant_2",
        question_number="1",
        sub_question=None,
        question_text=question_text,
        marking_scheme_answer=marking_scheme_answer,
        page_number=None,
        source_paper_reference="Mathematics 1014/2017/May_June/Paper_1/Variant_2",
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


class Phase4SourcePriorityTests(unittest.TestCase):
    """Test o_level_main_first: primary = O_LEVEL_MAIN_JSON, fallback = OA_MAIN_DATASET."""

    def test_o_level_main_first_uses_main_repository_as_primary(self):
        main_records = [
            _record(
                question_id="main|Mathematics 1014|2017|May_June|Paper_1|Variant_2|1",
                question_text="What is the formula of water?",
                marking_scheme_answer="H2O",
            )
        ]
        config = PipelineConfig(
            use_grok_grading=False,
            source_priority="o_level_main_first",
        )
        service = OALevelEvaluatorService(
            main_repository=_InMemoryRepo(main_records),
            repository=_InMemoryRepo([]),
            fallback_repository=_InMemoryRepo([]),
            config=config,
        )
        response = service.evaluate(
            EvaluateRequest(
                question="What is the formula of water?",
                student_answer="H2O",
                question_id="main|Mathematics 1014|2017|May_June|Paper_1|Variant_2|1",
            )
        )
        self.assertEqual(response.status, "accepted")
        self.assertEqual(response.data_source, "o_level_main_json")
        self.assertFalse(response.fallback_used)
        self.assertEqual(response.primary_data_source, "o_level_main_json")
        self.assertEqual(response.fallback_data_source, "oa_main_dataset")
        self.assertEqual(response.marking_scheme_answer, "H2O")

    def test_o_level_main_first_fallback_to_oa_main_dataset_when_main_fails(self):
        config = PipelineConfig(
            use_grok_grading=False,
            source_priority="o_level_main_first",
        )
        oa_records = [
            _record(
                question_id="oa|1",
                question_text="Which option is correct?",
                marking_scheme_answer="D",
            )
        ]
        service = OALevelEvaluatorService(
            main_repository=_InMemoryRepo([]),
            repository=_InMemoryRepo(oa_records),
            fallback_repository=_InMemoryRepo([]),
            config=config,
        )
        response = service.evaluate(
            EvaluateRequest(
                question="Which option is correct?",
                student_answer="D",
                question_id="oa|1",
            )
        )
        self.assertEqual(response.status, "accepted")
        self.assertEqual(response.data_source, "oa_main_dataset")
        self.assertTrue(response.fallback_used)
        self.assertEqual(response.primary_data_source, "o_level_main_json")
        self.assertEqual(response.fallback_data_source, "oa_main_dataset")


class Phase4PipelineModuleTests(unittest.TestCase):
    """Test oa_levels_pipeline facade: import and create_app."""

    def test_oa_levels_pipeline_imports_and_create_app(self):
        from oa_levels_pipeline import create_app
        from oa_levels_pipeline import OALevelEvaluatorService, PipelineConfig

        app = create_app()
        self.assertIsNotNone(app)
        self.assertTrue(hasattr(app, "routes"))

    def test_oa_levels_pipeline_api_app_attribute(self):
        from oa_levels_pipeline.api import app

        self.assertIsNotNone(app)
        self.assertTrue(hasattr(app, "routes"))


if __name__ == "__main__":
    unittest.main()
