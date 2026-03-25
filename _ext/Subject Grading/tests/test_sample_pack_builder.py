"""Tests for the O_LEVEL_MAIN_JSON sample pack builder (Phase 3)."""

import json
import tempfile
import unittest
from pathlib import Path
from typing import Iterable, List, Optional

from oa_main_pipeline.config import PipelineConfig
from oa_main_pipeline.dataset_repository import filter_question_records
from oa_main_pipeline.o_level_main_repository import load_records_from_main_json
from oa_main_pipeline.schemas import EvaluateRequest, QuestionRecord
from oa_main_pipeline.service import OALevelEvaluatorService
from oa_main_pipeline.sample_pack_builder import (
    _DEFAULT_COUNT,
    _DEFAULT_INPUT,
    _DEFAULT_OUTPUT,
    _DEFAULT_SEED,
    build_sample_pack,
)

_NO_GROK_CONFIG = PipelineConfig(use_grok_grading=False)


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

_REQUIRED_KEYS = ("question_id", "question_text", "page_number", "marking_scheme_answer", "source_paper_reference")


def _input_exists() -> bool:
    return _DEFAULT_INPUT.exists()


@unittest.skipUnless(_input_exists(), "O_LEVEL_MAIN_JSON Mathematics 1014/2015-2019.json not found")
class SamplePackBuilderTests(unittest.TestCase):
    """Tests that require the real O_LEVEL_MAIN_JSON course file."""

    def test_build_sample_pack_produces_exactly_10_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "sample.json"
            result = build_sample_pack(
                input_path=_DEFAULT_INPUT,
                output_path=out,
                count=_DEFAULT_COUNT,
                seed=_DEFAULT_SEED,
            )
            self.assertEqual(len(result), _DEFAULT_COUNT, "Expected exactly 10 exported items")
            self.assertTrue(out.exists())
            loaded = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(len(loaded), _DEFAULT_COUNT)

    def test_output_has_all_required_keys_and_non_empty_where_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "sample.json"
            result = build_sample_pack(
                input_path=_DEFAULT_INPUT,
                output_path=out,
                count=_DEFAULT_COUNT,
                seed=_DEFAULT_SEED,
            )
            for item in result:
                for key in _REQUIRED_KEYS:
                    self.assertIn(key, item, f"Missing key {key}")
                self.assertIsNotNone(item.get("question_id"), "question_id must be non-empty")
                self.assertIsNotNone(item.get("question_text"), "question_text must be non-empty")
                self.assertIsNotNone(item.get("marking_scheme_answer"), "marking_scheme_answer must be non-empty")
                self.assertIsNotNone(item.get("source_paper_reference"), "source_paper_reference must be non-empty")
                # page_number may be null

    def test_no_duplicate_question_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "sample.json"
            result = build_sample_pack(
                input_path=_DEFAULT_INPUT,
                output_path=out,
                count=_DEFAULT_COUNT,
                seed=_DEFAULT_SEED,
            )
            ids = [item["question_id"] for item in result]
            self.assertEqual(len(ids), len(set(ids)), "Duplicate question_id in pack")

    def test_fixed_seed_produces_stable_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out1 = Path(tmp) / "sample1.json"
            out2 = Path(tmp) / "sample2.json"
            result1 = build_sample_pack(
                input_path=_DEFAULT_INPUT,
                output_path=out1,
                count=_DEFAULT_COUNT,
                seed=_DEFAULT_SEED,
            )
            result2 = build_sample_pack(
                input_path=_DEFAULT_INPUT,
                output_path=out2,
                count=_DEFAULT_COUNT,
                seed=_DEFAULT_SEED,
            )
            ids1 = [r["question_id"] for r in result1]
            ids2 = [r["question_id"] for r in result2]
            self.assertEqual(ids1, ids2, "Same seed must yield same question set")
            self.assertEqual(result1, result2, "Full export must be identical with same seed")


class OLevelMainLoaderTests(unittest.TestCase):
    """Tests for the O_LEVEL_MAIN_JSON loader (no real file required)."""

    def test_load_records_from_missing_file_returns_empty_list(self):
        records = load_records_from_main_json(Path("/nonexistent/file.json"))
        self.assertEqual(records, [])

    def test_load_records_from_empty_json_returns_empty_list(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b"{}")
            path = Path(f.name)
        try:
            records = load_records_from_main_json(path)
            self.assertEqual(records, [])
        finally:
            path.unlink(missing_ok=True)


@unittest.skipUnless(_input_exists(), "O_LEVEL_MAIN_JSON Mathematics 1014/2015-2019.json not found")
class SamplePackEvaluatorSmokeTests(unittest.TestCase):
    """Smoke test: evaluator accepts one sampled question when answer matches scheme (Grok disabled)."""

    def test_evaluator_accepts_sampled_question_with_scheme_answer(self):
        records = load_records_from_main_json(_DEFAULT_INPUT)
        self.assertGreater(len(records), 0, "Need at least one record from main JSON")
        record = records[0]
        service = OALevelEvaluatorService(
            repository=_InMemoryRepo([]),
            fallback_repository=_InMemoryRepo([record]),
            config=_NO_GROK_CONFIG,
        )
        request = EvaluateRequest(
            question=record.question_text,
            student_answer=record.marking_scheme_answer,
            question_id=record.question_id,
        )
        response = service.evaluate(request, debug=False)
        self.assertEqual(response.status, "accepted", f"Expected accepted, got {response.status}: {response.feedback}")
        self.assertEqual(response.matched_question_id, record.question_id)
        self.assertEqual(response.source_paper_reference, record.source_paper_reference)
        self.assertEqual(response.marking_scheme_answer, record.marking_scheme_answer)


if __name__ == "__main__":
    unittest.main()
