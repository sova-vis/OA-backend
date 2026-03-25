import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import oa_main_pipeline.answer_evaluator as answer_evaluator
from oa_main_pipeline.answer_evaluator import evaluate_answer
from oa_main_pipeline.config import PipelineConfig
from oa_main_pipeline.dataset_repository import DatasetRepository
from oa_main_pipeline.question_matcher import match_question
from oa_main_pipeline.schemas import EvaluateRequest, QuestionRecord
from oa_main_pipeline.service import OALevelEvaluatorService


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class OAPhase1UnitTests(unittest.TestCase):
    def _build_temp_dataset(self, root: Path) -> None:
        # Accepted pair
        accepted_dir = root / "Chemistry 1011" / "2016" / "May_June" / "Paper_1" / "Variant_2"
        _write_json(
            accepted_dir / "pair_extraction_summary.json",
            {
                "status": "accepted",
                "metadata": {
                    "subject": "Chemistry 1011",
                    "year": 2016,
                    "session": "May_June",
                    "paper": "Paper_1",
                    "variant": "Variant_2",
                },
            },
        )
        _write_json(
            accepted_dir / "qp_extracted.json",
            {
                "questions": [
                    {
                        "question_number": "1",
                        "sub_question": None,
                        "page_number": 2,
                        "question_text": "What is the formula of water?",
                    },
                    {
                        "question_number": "2",
                        "sub_question": None,
                        "page_number": 2,
                        "question_text": "Choose the correct gas test result.",
                    },
                ]
            },
        )
        _write_json(
            accepted_dir / "ms_extracted.json",
            {
                "marking_entries": [
                    {
                        "question_number": "1",
                        "sub_question": None,
                        "page_number": 2,
                        "marking_scheme": "H2O",
                    },
                    {
                        "question_number": "2",
                        "sub_question": None,
                        "page_number": 2,
                        "marking_scheme": "B",
                    },
                ]
            },
        )

        # Non-accepted pair should be ignored.
        skipped_dir = root / "Chemistry 1011" / "2017" / "May_June" / "Paper_1" / "Variant_1"
        _write_json(
            skipped_dir / "pair_extraction_summary.json",
            {
                "status": "review_required",
                "metadata": {
                    "subject": "Chemistry 1011",
                    "year": 2017,
                    "session": "May_June",
                    "paper": "Paper_1",
                    "variant": "Variant_1",
                },
            },
        )
        _write_json(
            skipped_dir / "qp_extracted.json",
            {"questions": [{"question_number": "1", "question_text": "Ignored Q"}]},
        )
        _write_json(
            skipped_dir / "ms_extracted.json",
            {"marking_entries": [{"question_number": "1", "marking_scheme": "A"}]},
        )

    def _build_temp_fallback_dataset(self, root: Path) -> None:
        _write_json(
            root / "Chemistry 1011" / "2015-2019.json",
            {
                "2016": {
                    "May_June": {
                        "Paper_1": {
                            "Variant_2": [
                                {
                                    "question_number": "2",
                                    "sub_question": None,
                                    "question_text": "Choose the correct gas test result.",
                                    "marking_scheme": "B",
                                    "page_number": 2,
                                }
                            ]
                        }
                    }
                }
            },
        )

    def test_repository_loads_only_accepted_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "OA_MAIN_DATASET"
            self._build_temp_dataset(root)
            repo = DatasetRepository(PipelineConfig(dataset_root=root))
            records = repo.get_records()
            self.assertEqual(len(records), 2)
            self.assertTrue(all(r.year == 2016 for r in records))

    def test_matching_exact_near_and_wrong(self):
        records = [
            QuestionRecord(
                question_id="chem|2016|mj|p1|v2|1",
                subject="Chemistry 1011",
                year=2016,
                session="May_June",
                paper="Paper_1",
                variant="Variant_2",
                question_number="1",
                sub_question=None,
                question_text="What is the formula of water?",
                marking_scheme_answer="H2O",
                page_number=2,
                source_paper_reference="Chemistry 1011/2016/May_June/Paper_1/Variant_2",
            ),
            QuestionRecord(
                question_id="chem|2016|mj|p1|v2|2",
                subject="Chemistry 1011",
                year=2016,
                session="May_June",
                paper="Paper_1",
                variant="Variant_2",
                question_number="2",
                sub_question=None,
                question_text="Choose the correct gas test result.",
                marking_scheme_answer="B",
                page_number=2,
                source_paper_reference="Chemistry 1011/2016/May_June/Paper_1/Variant_2",
            ),
        ]

        exact = match_question("What is the formula of water?", records)
        near = match_question("water formula", records)
        wrong = match_question("Explain photosynthesis and chlorophyll", records)

        self.assertEqual(exact.status, "accepted")
        self.assertEqual(near.status, "review_required")
        self.assertEqual(wrong.status, "failed")

    def test_mcq_evaluator_correct_and_incorrect(self):
        correct = evaluate_answer("B", "B")
        wrong = evaluate_answer("A", "B")
        self.assertEqual(correct.grade_label, "fully_correct")
        self.assertEqual(correct.score, 1.0)
        self.assertEqual(wrong.grade_label, "weak")
        self.assertEqual(wrong.score, 0.0)

    def test_mcq_evaluator_handles_verbose_markscheme(self):
        scheme = (
            "The correct answer is B: Award 1 mark for selecting the option "
            "that identifies aqueous ammonia forming a white precipitate soluble in excess."
        )
        correct = evaluate_answer("B", scheme)
        wrong = evaluate_answer("D", scheme)
        self.assertEqual(correct.grade_label, "fully_correct")
        self.assertEqual(correct.score, 1.0)
        self.assertEqual(correct.correct_option, "B")
        self.assertEqual(wrong.grade_label, "weak")
        self.assertEqual(wrong.score, 0.0)
        self.assertEqual(wrong.correct_option, "B")

    def test_grok_grading_path_for_mcq(self):
        class FakeResp:
            status_code = 200

            @staticmethod
            def raise_for_status() -> None:
                return None

            @staticmethod
            def json() -> dict:
                return {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "score": 1,
                                        "grade_label": "fully_correct",
                                        "feedback": "Correct option selected.",
                                        "expected_points": ["Correct option: B"],
                                        "missing_points": [],
                                        "student_option": "B",
                                        "correct_option": "B",
                                    }
                                )
                            }
                        }
                    ]
                }

        cfg = PipelineConfig(
            use_grok_grading=True,
            grok_api_key="xai-test-key",
            grok_model="grok-test-model",
            grok_max_retries=0,
        )
        with patch.object(answer_evaluator.requests, "post", return_value=FakeResp()) as mocked_post:
            result = evaluate_answer(
                student_answer="B",
                marking_scheme_answer="The correct answer is B.",
                config=cfg,
                question_text="Which option is correct?",
            )
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.grade_label, "fully_correct")
        self.assertEqual(result.grading_source, "grok")
        self.assertEqual(result.grading_model, "grok-test-model")
        self.assertIsNone(result.grading_error)
        self.assertEqual(mocked_post.call_count, 1)

    def test_grok_grading_fallbacks_to_deterministic_on_invalid_json(self):
        class FakeResp:
            status_code = 200

            @staticmethod
            def raise_for_status() -> None:
                return None

            @staticmethod
            def json() -> dict:
                return {
                    "choices": [
                        {
                            "message": {
                                "content": "not-json"
                            }
                        }
                    ]
                }

        cfg = PipelineConfig(
            use_grok_grading=True,
            grok_api_key="xai-test-key",
            grok_model="grok-test-model",
            grok_max_retries=0,
        )
        with patch.object(answer_evaluator.requests, "post", return_value=FakeResp()):
            result = evaluate_answer(
                student_answer="B",
                marking_scheme_answer="B",
                config=cfg,
            )
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.grade_label, "fully_correct")
        self.assertEqual(result.grading_source, "deterministic")
        self.assertTrue((result.grading_error or "").startswith("grok_grading_failed:"))

    def test_non_mcq_evaluator_score_bands(self):
        scheme = "photosynthesis needs light energy. carbon dioxide is used. oxygen is produced."
        strong = evaluate_answer(
            "Photosynthesis needs light energy and uses carbon dioxide. Oxygen is produced.",
            scheme,
        )
        partial = evaluate_answer(
            "Photosynthesis needs light energy and uses carbon dioxide.",
            scheme,
        )
        weak = evaluate_answer("It is a process in leaves.", scheme)
        self.assertEqual(strong.grade_label, "fully_correct")
        self.assertEqual(partial.grade_label, "partially_correct")
        self.assertEqual(weak.grade_label, "weak")

    def test_service_response_schema_completeness(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "OA_MAIN_DATASET"
            fallback_root = Path(tmp) / "O_LEVEL_JSON"
            self._build_temp_dataset(root)
            self._build_temp_fallback_dataset(fallback_root)
            config = PipelineConfig(
                dataset_root=root,
                fallback_root=fallback_root,
                use_grok_grading=False,
                source_priority="o_level_json_first",
            )
            service = OALevelEvaluatorService(config=config)
            response = service.evaluate(
                EvaluateRequest(
                    question="Choose the correct gas test result.",
                    student_answer="B",
                    subject="Chemistry 1011",
                    year=2016,
                    session="May_June",
                    paper="Paper_1",
                    variant="Variant_2",
                )
            )
            self.assertIn(response.status, {"accepted", "review_required"})
            self.assertTrue(response.matched_question_text)
            self.assertTrue(response.source_paper_reference)
            self.assertTrue(response.marking_scheme_answer)
            self.assertIsNotNone(response.page_number)
            self.assertIsNotNone(response.matched_question_id)
            self.assertEqual(response.data_source, "o_level_json")
            self.assertFalse(response.fallback_used)
            self.assertEqual(response.primary_data_source, "o_level_json")
            self.assertEqual(response.fallback_data_source, "oa_main_dataset")
            self.assertIn(response.primary_status, {"accepted", "review_required"})
            self.assertGreaterEqual(response.primary_match_confidence, 0.0)
            self.assertLessEqual(response.primary_match_confidence, 1.0)
            self.assertGreaterEqual(response.score, 0.0)
            self.assertLessEqual(response.score, 1.0)
            self.assertGreaterEqual(response.score_percent, 0.0)
            self.assertLessEqual(response.score_percent, 100.0)


if __name__ == "__main__":
    unittest.main()
