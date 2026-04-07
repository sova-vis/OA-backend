"""Tests for fold_plaintext_science_symbols and build_subject_matcher_text wiring."""

import unittest

from oa_main_pipeline.content_normalization import (
    build_subject_matcher_text,
    fold_math_matcher_input,
    fold_plaintext_science_symbols,
    fold_unicode_numeric_forms,
    normalize_math_text_result,
)
from oa_main_pipeline.question_matcher import normalize_text, tokenize


class ScienceSymbolFoldTests(unittest.TestCase):
    def test_hyphen_minus_and_nbsp(self) -> None:
        self.assertEqual(fold_plaintext_science_symbols("a\u2212b"), "a-b")
        self.assertEqual(fold_plaintext_science_symbols("x\u00a0y"), "x y")

    def test_operators_and_arrow_and_dot(self) -> None:
        self.assertEqual(fold_plaintext_science_symbols("a\u00d7b"), "a*b")
        self.assertEqual(fold_plaintext_science_symbols("a\u00f7b"), "a/b")
        self.assertEqual(fold_plaintext_science_symbols("A\u2192B"), "A->B")
        self.assertEqual(fold_plaintext_science_symbols("H\u00b7O"), "H.O")

    def test_chain_with_digit_fold(self) -> None:
        raw = fold_unicode_numeric_forms("Na\u2082CO\u2083")
        self.assertEqual(fold_plaintext_science_symbols(raw), "Na2CO3")

    def test_idempotent_ascii(self) -> None:
        s = "Mg + CuSO4 -> Cu + MgSO4"
        self.assertEqual(fold_plaintext_science_symbols(s), s)

    def test_build_subject_matcher_text_chemistry(self) -> None:
        u = "10 cm\u00b3 of propane \u2192 3CO2"
        out = build_subject_matcher_text(u, subject="Chemistry 1011")
        self.assertIn("10", out)
        self.assertIn("cm3", out.replace(" ", ""))
        self.assertIn("->", out.replace(" ", ""))

    def test_normalize_text_converges(self) -> None:
        self.assertEqual(
            normalize_text("H\u2082O \u2192 gas"),
            normalize_text("H2O -> gas"),
        )

    def test_tokenize_idempotent_double_fold(self) -> None:
        once = "Na\u2082SO\u2084"
        folded = fold_unicode_numeric_forms(once)
        folded = fold_plaintext_science_symbols(folded)
        self.assertEqual(tokenize(folded), tokenize("Na2SO4"))

    def test_math_matcher_input_coalesces_hyphen(self) -> None:
        out = fold_math_matcher_input("ln\u00a0y = 2 \u2212 x")
        self.assertNotIn("\u00a0", out)
        self.assertNotIn("\u2212", out)
        self.assertIn("-", out)

    def test_math_subject_matcher_regression(self) -> None:
        s = "Variables x and y are such that ln y = 2 + 1/x - 3x^2."
        result = normalize_math_text_result(s, enable_canonical=True)
        self.assertIn("ln", result.matcher_text.casefold())
        self.assertIn("x", result.matcher_text.casefold())

    def test_build_subject_matcher_text_mathematics_uses_pretreatment(self) -> None:
        s = "Solve\u00a0ln(x)\u00a0=\u00a01"
        out = build_subject_matcher_text(s, subject="Mathematics 1014")
        self.assertNotIn("\u00a0", out)
        self.assertIn("ln", out.casefold())


if __name__ == "__main__":
    unittest.main()
