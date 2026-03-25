import unittest

from oa_main_pipeline.content_normalization import build_subject_matcher_text, normalize_math_text_result


class MathCanonicalizationTests(unittest.TestCase):
    def test_explicit_log_forms_are_canonicalized(self):
        result = normalize_math_text_result(
            "Solve log_3(x) + log₉(x) = 12",
            enable_canonical=True,
        )

        self.assertEqual(result.display_text, "Solve log_3 (x) + log_9 (x) = 12")
        self.assertEqual(
            result.canonical_text,
            "Solve log(base=3,arg=x) + log(base=9,arg=x) = 12",
        )
        self.assertEqual(result.matcher_text, "Solve log(base=3,arg=x) + log(base=9,arg=x) = 12")

    def test_ambiguous_log3x_is_not_canonicalized(self):
        result = normalize_math_text_result(
            "Solve log3x + log9x = 12",
            enable_canonical=True,
        )

        self.assertIsNone(result.canonical_text)
        self.assertIn("ambiguous_log_base_argument", result.warning_codes)
        self.assertEqual(result.matcher_text, "Solve log3x log9x 12")

    def test_math_subject_matcher_prefers_canonical_text(self):
        matcher_text = build_subject_matcher_text(
            "Solve log_3(x) + log_9(x) = 12",
            subject="Mathematics 1014",
        )
        non_math_text = build_subject_matcher_text(
            "Which option is correct?",
            subject="Chemistry 1011",
        )

        self.assertEqual(matcher_text, "Solve log(base=3,arg=x) + log(base=9,arg=x) = 12")
        self.assertEqual(non_math_text, "Which option is correct?")


if __name__ == "__main__":
    unittest.main()
