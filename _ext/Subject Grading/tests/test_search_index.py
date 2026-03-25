import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from oa_main_pipeline.config import PipelineConfig
from oa_main_pipeline.schemas import QuestionRecord
from oa_main_pipeline.search_index import SearchIndexManager


def _record(
    *,
    question_id: str,
    question_text: str,
    marking_scheme_answer: str,
    subject: str = "Chemistry 1011",
    year: int = 2016,
    session: str = "May_June",
    paper: str = "Paper_1",
    variant: str = "Variant_2",
) -> QuestionRecord:
    return QuestionRecord(
        question_id=question_id,
        subject=subject,
        year=year,
        session=session,
        paper=paper,
        variant=variant,
        question_number="1",
        sub_question=None,
        question_text=question_text,
        marking_scheme_answer=marking_scheme_answer,
        page_number=2,
        source_paper_reference=f"{subject}/{year}/{session}/{paper}/{variant}",
    )


class _InMemoryRepo:
    def __init__(self, records: list[QuestionRecord]) -> None:
        self._records = list(records)

    def get_records(self) -> list[QuestionRecord]:
        return list(self._records)


class SearchIndexTests(unittest.TestCase):
    def test_index_build_writes_expected_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "SEARCH_INDEX_CACHE"
            config = PipelineConfig(
                search_cache_dir=cache_dir,
                use_grok_grading=False,
            )
            manager = SearchIndexManager(
                repository=_InMemoryRepo(
                    [_record(question_id="oa|1", question_text="What is water?", marking_scheme_answer="H2O")]
                ),
                fallback_repository=_InMemoryRepo(
                    [_record(question_id="fallback|1", question_text="What is water?", marking_scheme_answer="H2O")]
                ),
                config=config,
            )
            with patch("oa_main_pipeline.search_index.SentenceTransformer", None):
                first = manager.ensure_built("o_level_json")
                second = manager.ensure_built("oa_main_dataset")

            self.assertTrue(first["index_rebuilt"])
            self.assertTrue(second["index_rebuilt"])
            self.assertTrue((cache_dir / "o_level_json" / "manifest.json").exists())
            self.assertTrue((cache_dir / "o_level_json" / "records.jsonl").exists())
            self.assertTrue((cache_dir / "o_level_json" / "embeddings.npy").exists())
            manifest = json.loads((cache_dir / "o_level_json" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["record_count"], 1)
            self.assertEqual(manifest["embedding_backend"], "hash_fallback")

    def test_index_rebuilds_when_source_priority_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "SEARCH_INDEX_CACHE"
            records = [_record(question_id="fallback|1", question_text="What is water?", marking_scheme_answer="H2O")]
            with patch("oa_main_pipeline.search_index.SentenceTransformer", None):
                manager = SearchIndexManager(
                    repository=_InMemoryRepo([]),
                    fallback_repository=_InMemoryRepo(records),
                    config=PipelineConfig(search_cache_dir=cache_dir, use_grok_grading=False),
                )
                first = manager.ensure_built("o_level_json")
                manager = SearchIndexManager(
                    repository=_InMemoryRepo([]),
                    fallback_repository=_InMemoryRepo(records),
                    config=PipelineConfig(
                        search_cache_dir=cache_dir,
                        use_grok_grading=False,
                        source_priority="oa_main_dataset_first",
                    ),
                )
                second = manager.ensure_built("o_level_json")

            self.assertTrue(first["index_rebuilt"])
            self.assertTrue(second["index_rebuilt"])
            self.assertEqual(second["rebuild_reason"], "source_priority_changed")

    def test_search_returns_exact_and_wrong_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "SEARCH_INDEX_CACHE"
            records = [
                _record(question_id="fallback|1", question_text="What is the formula of water?", marking_scheme_answer="H2O"),
                _record(question_id="fallback|2", question_text="Choose the correct gas test result.", marking_scheme_answer="B"),
            ]
            manager = SearchIndexManager(
                repository=_InMemoryRepo([]),
                fallback_repository=_InMemoryRepo(records),
                config=PipelineConfig(search_cache_dir=cache_dir, use_grok_grading=False),
            )
            with patch("oa_main_pipeline.search_index.SentenceTransformer", None):
                exact = manager.search(
                    source="o_level_json",
                    query="What is the formula of water?",
                    student_answer="H2O",
                    records=records,
                )
                wrong = manager.search(
                    source="o_level_json",
                    query="Explain plate tectonics and crust motion.",
                    student_answer="Something about earthquakes",
                    records=records,
                )

            self.assertEqual(exact.match_result.status, "accepted")
            self.assertEqual(exact.match_result.best_record.question_id, "fallback|1")
            self.assertEqual(wrong.match_result.status, "failed")

    def test_answer_rerank_can_break_ties_for_mcq(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "SEARCH_INDEX_CACHE"
            records = [
                _record(
                    question_id="fallback|1",
                    question_text="Which statement about zinc chloride is correct?",
                    marking_scheme_answer="A",
                ),
                _record(
                    question_id="fallback|2",
                    question_text="Which statement about zinc chloride is correct?",
                    marking_scheme_answer="B",
                ),
            ]
            manager = SearchIndexManager(
                repository=_InMemoryRepo([]),
                fallback_repository=_InMemoryRepo(records),
                config=PipelineConfig(search_cache_dir=cache_dir, use_grok_grading=False),
            )
            with patch("oa_main_pipeline.search_index.SentenceTransformer", None):
                result = manager.search(
                    source="o_level_json",
                    query="Which statement about zinc chloride is correct?",
                    student_answer="B",
                    records=records,
                )

            self.assertEqual(result.match_result.status, "accepted")
            self.assertEqual(result.match_result.best_record.question_id, "fallback|2")

    def test_embed_backend_hash_uses_hash_fallback_without_hf(self):
        """With embed_backend=hash, SearchIndexManager uses _HashingEmbedder only (no Hugging Face)."""
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "SEARCH_INDEX_CACHE"
            records = [_record(question_id="fallback|1", question_text="What is water?", marking_scheme_answer="H2O")]
            config = PipelineConfig(
                search_cache_dir=cache_dir,
                use_grok_grading=False,
                embed_backend="hash",
            )
            manager = SearchIndexManager(
                repository=_InMemoryRepo([]),
                fallback_repository=_InMemoryRepo(records),
                config=config,
            )
            build_state = manager.ensure_built("o_level_json")
            self.assertEqual(build_state.get("embedding_backend"), "hash_fallback")
            self.assertEqual(build_state.get("embedding_model"), "hash-token-384")
            manifest = json.loads((cache_dir / "o_level_json" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["embedding_backend"], "hash_fallback")

    def test_search_uses_canonical_math_text_for_mathematics(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "SEARCH_INDEX_CACHE"
            records = [
                _record(
                    question_id="fallback|1",
                    question_text="Solve log_3(x) + log_9(x) = 12",
                    marking_scheme_answer="x=6561",
                    subject="Mathematics 1014",
                ),
                _record(
                    question_id="fallback|2",
                    question_text="Solve log(3x) + log(9x) = 12",
                    marking_scheme_answer="x=12",
                    subject="Mathematics 1014",
                ),
            ]
            manager = SearchIndexManager(
                repository=_InMemoryRepo([]),
                fallback_repository=_InMemoryRepo(records),
                config=PipelineConfig(search_cache_dir=cache_dir, use_grok_grading=False),
            )
            with patch("oa_main_pipeline.search_index.SentenceTransformer", None):
                result = manager.search(
                    source="o_level_json",
                    query="Solve log(base=3,arg=x) + log(base=9,arg=x) = 12",
                    query_subject="Mathematics 1014",
                    student_answer="x=6561",
                    records=records,
                )

            self.assertEqual(result.match_result.status, "accepted")
            self.assertEqual(result.match_result.best_record.question_id, "fallback|1")


if __name__ == "__main__":
    unittest.main()
