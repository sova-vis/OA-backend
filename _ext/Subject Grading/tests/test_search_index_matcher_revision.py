"""Tests for search index invalidation when OA_MATCHER_REVISION / matcher pipeline changes."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from oa_main_pipeline.config import PipelineConfig
from oa_main_pipeline.schemas import QuestionRecord
from oa_main_pipeline.search_index import SearchIndexManager


def _record(
    *,
    question_id: str,
    question_text: str,
    marking_scheme_answer: str,
    subject: str = "Chemistry 1011",
) -> QuestionRecord:
    return QuestionRecord(
        question_id=question_id,
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
    def __init__(self, records: list[QuestionRecord]) -> None:
        self._records = list(records)

    def get_records(self) -> list[QuestionRecord]:
        return list(self._records)


def _hash_config(tmp: str, *, matcher_revision: str) -> PipelineConfig:
    return PipelineConfig(
        search_cache_dir=Path(tmp) / "SEARCH_INDEX_CACHE",
        embed_backend="hash",
        matcher_revision=matcher_revision,
        use_grok_grading=False,
    )


class SearchIndexMatcherRevisionTests(unittest.TestCase):
    def test_manifest_contains_matcher_revision_after_build(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            records = [
                _record(question_id="fb|1", question_text="What is water?", marking_scheme_answer="H2O"),
            ]
            config = _hash_config(tmp, matcher_revision="1")
            manager = SearchIndexManager(
                repository=_InMemoryRepo([]),
                fallback_repository=_InMemoryRepo(records),
                config=config,
            )
            out = manager.ensure_built("o_level_json")
            self.assertEqual(out.get("matcher_revision"), "1")

            manifest_path = Path(tmp) / "SEARCH_INDEX_CACHE" / "o_level_json" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest.get("matcher_revision"), "1")

    def test_matcher_revision_change_triggers_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            records = [
                _record(question_id="fb|1", question_text="What is water?", marking_scheme_answer="H2O"),
            ]
            cache_dir = Path(tmp) / "SEARCH_INDEX_CACHE"
            cfg1 = _hash_config(tmp, matcher_revision="1")
            manager1 = SearchIndexManager(
                repository=_InMemoryRepo([]),
                fallback_repository=_InMemoryRepo(records),
                config=cfg1,
            )
            first = manager1.ensure_built("o_level_json")
            self.assertTrue(first["index_rebuilt"])

            cfg2 = PipelineConfig(
                search_cache_dir=cache_dir,
                embed_backend="hash",
                matcher_revision="2",
                use_grok_grading=False,
            )
            manager2 = SearchIndexManager(
                repository=_InMemoryRepo([]),
                fallback_repository=_InMemoryRepo(records),
                config=cfg2,
            )
            second = manager2.ensure_built("o_level_json")
            self.assertTrue(second["index_rebuilt"])
            self.assertEqual(second.get("rebuild_reason"), "matcher_revision_changed")
            self.assertEqual(second.get("matcher_revision"), "2")

            manifest = json.loads((cache_dir / "o_level_json" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest.get("matcher_revision"), "2")

    def test_same_matcher_revision_skips_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            records = [
                _record(question_id="fb|1", question_text="What is water?", marking_scheme_answer="H2O"),
            ]
            config = _hash_config(tmp, matcher_revision="1")
            manager = SearchIndexManager(
                repository=_InMemoryRepo([]),
                fallback_repository=_InMemoryRepo(records),
                config=config,
            )
            self.assertTrue(manager.ensure_built("o_level_json")["index_rebuilt"])
            self.assertFalse(manager.ensure_built("o_level_json")["index_rebuilt"])

    def test_should_rebuild_matcher_revision_direct(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            records = [
                _record(question_id="fb|1", question_text="What is water?", marking_scheme_answer="H2O"),
            ]
            config = _hash_config(tmp, matcher_revision="1")
            manager = SearchIndexManager(
                repository=_InMemoryRepo([]),
                fallback_repository=_InMemoryRepo(records),
                config=config,
            )
            manager.ensure_built("o_level_json")
            artifacts = manager._artifacts_for("o_level_json")

            stale = PipelineConfig(
                search_cache_dir=config.search_cache_dir,
                embed_backend="hash",
                matcher_revision="99",
                use_grok_grading=False,
            )
            stale_manager = SearchIndexManager(
                repository=_InMemoryRepo([]),
                fallback_repository=_InMemoryRepo(records),
                config=stale,
            )
            should, reason = stale_manager._should_rebuild("o_level_json", artifacts)
            self.assertTrue(should)
            self.assertEqual(reason, "matcher_revision_changed")


if __name__ == "__main__":
    unittest.main()
