import unittest

from oa_main_pipeline.answer_evaluator import _normalize_text, _tokens
from oa_main_pipeline.content_normalization import fold_unicode_numeric_forms
from oa_main_pipeline.question_matcher import (
    _sequence_similarity_score,
    normalize_text,
    tokenize,
)


class UnicodeFormulaFoldTests(unittest.TestCase):
    def test_fold_maps_subscripts_and_superscripts(self):
        self.assertEqual(fold_unicode_numeric_forms("H₂O"), "H2O")
        self.assertEqual(fold_unicode_numeric_forms("SO₄²-"), "SO42-")
        self.assertEqual(fold_unicode_numeric_forms("cm³"), "cm3")

    def test_tokenize_unicode_formula_matches_ascii(self):
        self.assertEqual(tokenize("H₂O"), tokenize("H2O"))
        self.assertEqual(tokenize("Na₂CO₃"), tokenize("Na2CO3"))

    def test_normalize_text_preserves_digits_after_fold(self):
        self.assertEqual(normalize_text("H₂O"), "h2o")
        self.assertEqual(normalize_text("H2O"), "h2o")

    def test_sequence_similarity_unicode_vs_ascii(self):
        q = "What is the formula of water H₂O?"
        c = "What is the formula of water H2O?"
        self.assertGreaterEqual(_sequence_similarity_score(q, c), 0.8)

    def test_answer_evaluator_normalize_matches_question_matcher(self):
        u = "The product is H₂SO₄"
        a = "The product is H2SO4"
        self.assertEqual(_normalize_text(u), normalize_text(u))
        self.assertEqual(_normalize_text(u), _normalize_text(a))
        self.assertEqual(_tokens(u), _tokens(a))


if __name__ == "__main__":
    unittest.main()
