"""Tests for Mathematics-only lg / base-10 log matcher alignment."""

from __future__ import annotations

import os
import unittest

from oa_main_pipeline.content_normalization import (
    _apply_mathematics_common_log_matcher_fold,
    build_subject_matcher_text,
    normalize_content_text_result,
)
from oa_main_pipeline.question_matcher import tokenize


class CommonLogMatcherFoldTests(unittest.TestCase):
    def test_lg_and_bare_log_match_for_mathematics_1014(self) -> None:
        lg_m = build_subject_matcher_text("lg y", subject="Mathematics 1014")
        log_m = build_subject_matcher_text("log y", subject="Mathematics 1014")
        self.assertEqual(lg_m, log_m)
        self.assertIn("log base 10", lg_m)
        self.assertIn("y", lg_m)

    def test_ln_unchanged_and_distinct_from_log(self) -> None:
        ln_m = build_subject_matcher_text("ln y", subject="Mathematics 1014")
        log_m = build_subject_matcher_text("log y", subject="Mathematics 1014")
        self.assertIn("ln", ln_m.casefold())
        self.assertNotEqual(tokenize(ln_m), tokenize(log_m))

    def test_log_3_unchanged(self) -> None:
        m = build_subject_matcher_text("Solve log_3(x) + log_9(x) = 12", subject="Mathematics 1014")
        ml = m.casefold()
        self.assertIn("log(base=3", ml)
        self.assertIn("log(base=9", ml)
        self.assertNotIn("log base 10", ml)

    def test_canonical_log_base_10_collapses(self) -> None:
        m = build_subject_matcher_text("Solve log_10(x) = 2", subject="Mathematics 1014")
        self.assertIn("log base 10", m.casefold())
        self.assertNotIn("log(base=10", m.casefold())

    def test_normalize_content_text_result_applies_fold_for_math(self) -> None:
        res = normalize_content_text_result(
            "lg t = 3",
            content_type="symbolic",
            subject="Mathematics 1014",
        )
        self.assertIn("log base 10", res.matcher_text.casefold())

    def test_non_math_subject_no_fold(self) -> None:
        chem = build_subject_matcher_text("lg y", subject="Chemistry 1011")
        self.assertIn("lg", chem.casefold())

    def test_fold_bare_log_disabled_keeps_log_token(self) -> None:
        prev = os.environ.get("OA_MATCHER_FOLD_BARE_LOG_TO_BASE10")
        try:
            os.environ["OA_MATCHER_FOLD_BARE_LOG_TO_BASE10"] = "false"
            log_m = build_subject_matcher_text("log y", subject="Mathematics 1014")
            self.assertRegex(log_m.casefold(), r"\blog\b")
            self.assertNotIn("log base 10", log_m.casefold())
        finally:
            if prev is None:
                os.environ.pop("OA_MATCHER_FOLD_BARE_LOG_TO_BASE10", None)
            else:
                os.environ["OA_MATCHER_FOLD_BARE_LOG_TO_BASE10"] = prev

    def test_apply_fold_idempotent(self) -> None:
        once = _apply_mathematics_common_log_matcher_fold("log base 10 x")
        twice = _apply_mathematics_common_log_matcher_fold(once)
        self.assertEqual(once, twice)


if __name__ == "__main__":
    unittest.main()
