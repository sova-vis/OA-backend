import json
import unittest
from pathlib import Path

from oa_main_pipeline.config import PipelineConfig
from oa_main_pipeline.schemas import EvaluateRequest
from oa_main_pipeline.service import OALevelEvaluatorService


PAIR_DIR = Path(
    "OA_MAIN_DATASET/Chemistry 1011/2016/May_June/Paper_1/Variant_2"
)
PAIR_SUMMARY_PATH = PAIR_DIR / "pair_extraction_summary.json"


@unittest.skipUnless(PAIR_SUMMARY_PATH.exists(), "Integration dataset pair not available.")
class OAPhase1IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        payload = json.loads(PAIR_SUMMARY_PATH.read_text(encoding="utf-8"))
        if str(payload.get("status") or "").strip().lower() != "accepted":
            raise unittest.SkipTest("Target integration pair is not accepted.")
        cls.service = OALevelEvaluatorService(
            config=PipelineConfig(use_grok_grading=False, source_priority="o_level_json_first")
        )

    def test_end_to_end_on_chemistry_2016_p1_v2(self):
        request = EvaluateRequest(
            question="Which row correctly identifies the gas?",
            student_answer="D",
            subject="Chemistry 1011",
            year=2016,
            session="May_June",
            paper="Paper_1",
            variant="Variant_2",
        )
        response = self.service.evaluate(request)
        self.assertIn(response.status, {"accepted", "review_required"})
        self.assertIn("D", response.marking_scheme_answer)
        self.assertEqual(response.source_paper_reference, "Chemistry 1011/2016/May_June/Paper_1/Variant_2")
        self.assertIn(response.data_source, {"oa_main_dataset", "o_level_json"})
        self.assertEqual(response.primary_data_source, "o_level_json")
        self.assertEqual(response.fallback_data_source, "oa_main_dataset")
        if response.fallback_used:
            self.assertEqual(response.data_source, "oa_main_dataset")
            self.assertEqual(response.primary_status, "failed")
            self.assertIsNotNone(response.page_number)
        else:
            self.assertEqual(response.data_source, "o_level_json")
            self.assertIn(response.primary_status, {"accepted", "review_required"})
        self.assertGreaterEqual(response.primary_match_confidence, 0.0)
        self.assertLessEqual(response.primary_match_confidence, 1.0)
        self.assertGreaterEqual(response.score, 0.0)
        self.assertLessEqual(response.score, 1.0)
        self.assertTrue(response.feedback)
        self.assertGreaterEqual(response.match_confidence, 0.0)

    def test_unrelated_question_returns_safe_status(self):
        request = EvaluateRequest(
            question="Explain plate tectonics and seismic drift patterns.",
            student_answer="This is about earthquakes and crust shifts.",
            subject="Chemistry 1011",
            year=2016,
            session="May_June",
            paper="Paper_1",
            variant="Variant_2",
        )
        response = self.service.evaluate(request)
        self.assertIn(response.status, {"review_required", "failed"})
        self.assertIn(response.data_source, {"oa_main_dataset", "o_level_json"})
        self.assertGreaterEqual(response.match_confidence, 0.0)
        self.assertLessEqual(response.match_confidence, 1.0)


if __name__ == "__main__":
    unittest.main()
