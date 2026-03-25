import argparse
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).resolve().parent.parent / "tools" / "ocr_extract_pipeline.py"
SPEC = importlib.util.spec_from_file_location("ocr_extract_pipeline", MODULE_PATH)
ocr_pipeline = importlib.util.module_from_spec(SPEC)
sys.modules["ocr_extract_pipeline"] = ocr_pipeline
SPEC.loader.exec_module(ocr_pipeline)


def make_args(**overrides):
    base = {
        "index_path": "OA_MAIN_DATASET/index.json",
        "run_report_path": "OA_MAIN_DATASET/ocr_run_report.json",
        "review_queue_path": "OA_MAIN_DATASET/ocr_review_queue.json",
        "subject": [],
        "year": None,
        "session": None,
        "paper": None,
        "variant": None,
        "limit": None,
        "resume": True,
        "force": False,
        "no_grok_normalization": False,
        "no_raw_ocr": False,
        "profile": "auto",
        "expected_max_question": 40,
        "ocr_grok_model": None,
        "grok_max_retries": None,
        "grok_timeout_normalize": None,
        "grok_timeout_repair": None,
        "debug": None,
        "debug_dir": None,
        "debug_level": None,
        "progress_log": None,
        "progress_log_path": None,
        "dry_run": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def build_settings(debug_enabled=False, debug_level="basic", debug_dir="OCR_DEBUG_OUTPUT"):
    return ocr_pipeline.PipelineSettings(
        grok_api_key="grok",
        grok_model="grok-4-1-fast-reasoning",
        grok_max_retries=2,
        grok_timeout_normalize=90,
        grok_timeout_repair=90,
        azure_endpoint="https://example.com",
        azure_key="secret",
        ocr_per_page_timeout=120.0,
        ocr_overall_timeout=900.0,
        ocr_max_retries=3,
        ocr_retry_base_delay=1.0,
        ocr_retry_max_delay=30.0,
        ocr_concurrent_pages=2,
        review_conf_threshold=0.75,
        review_match_threshold=0.80,
        use_grok_normalization=True,
        parser_profile="auto",
        expected_max_question=40,
        debug_enabled=debug_enabled,
        debug_dir=Path(debug_dir),
        debug_level=debug_level,
        debug_run_id="20260312T000000Z",
        progress_log_enabled=True,
        progress_log_path=Path(debug_dir) / "20260312T000000Z" / "pipeline_progress.txt",
    )


def sample_ocr_payload():
    return {
        "source_pdf": "dummy.pdf",
        "processed_at_utc": "2026-03-12T00:00:00+00:00",
        "metadata": {
            "page_count": 1,
            "ocr_duration_seconds": 1.0,
            "avg_confidence": 0.98,
            "errors": [],
            "error_count": 0,
        },
        "full_text": "Question text",
        "pages": [
            {
                "page_number": 1,
                "avg_confidence": 0.98,
                "lines": [{"text": "1 Which statement is correct? A one B two C three D four"}],
                "words": [{"text": "1", "confidence": 0.99}],
            }
        ],
    }


class OCRExtractPipelineTests(unittest.TestCase):
    def test_parse_anchor_question_with_sub(self):
        got = ocr_pipeline.parse_anchor("12(a) Explain the process")
        self.assertEqual(got, ("12", "(a)", "Explain the process"))

    def test_load_settings_uses_required_env_keys(self):
        args = make_args()
        env = {
            "Grok_API": "grok-key",
            "AZURE_ENDPOINT": "https://ocr.example.com",
            "AZURE_KEY": "azure-key",
            "OCR_GROK_MODEL": "grok-custom",
            "CHATBOT_LLM_MODEL": "should-not-be-used",
            "MCQ_LLM_MODEL": "should-not-be-used",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = ocr_pipeline.load_settings(args)

        self.assertEqual(settings.grok_api_key, "grok-key")
        self.assertEqual(settings.azure_endpoint, "https://ocr.example.com")
        self.assertEqual(settings.azure_key, "azure-key")
        self.assertEqual(settings.grok_model, "grok-custom")
        self.assertTrue(settings.progress_log_enabled)
        self.assertIn("pipeline_progress.txt", str(settings.progress_log_path))

    def test_load_settings_raises_when_azure_missing(self):
        args = make_args()
        with patch.dict(os.environ, {"Grok_API": "grok-key"}, clear=True):
            with patch.object(ocr_pipeline, "_load_env_file_fallback", return_value=None):
                with self.assertRaises(EnvironmentError):
                    ocr_pipeline.load_settings(args)

    def test_load_settings_progress_log_override(self):
        args = make_args(progress_log=False, progress_log_path="tmp/custom_progress.txt")
        env = {
            "Grok_API": "grok-key",
            "AZURE_ENDPOINT": "https://ocr.example.com",
            "AZURE_KEY": "azure-key",
            "OCR_PROGRESS_LOG": "true",
        }
        with patch.dict(os.environ, env, clear=True):
            with patch.object(ocr_pipeline, "_load_env_file_fallback", return_value=None):
                settings = ocr_pipeline.load_settings(args)
        self.assertFalse(settings.progress_log_enabled)
        self.assertEqual(str(settings.progress_log_path).replace("\\", "/"), "tmp/custom_progress.txt")

    def test_load_settings_progress_log_dir_override_from_env(self):
        args = make_args()
        env = {
            "Grok_API": "grok-key",
            "AZURE_ENDPOINT": "https://ocr.example.com",
            "AZURE_KEY": "azure-key",
            "OCR_PROGRESS_LOG_PATH": "tmp/progress_logs",
        }
        with patch.dict(os.environ, env, clear=True):
            with patch.object(ocr_pipeline, "_load_env_file_fallback", return_value=None):
                settings = ocr_pipeline.load_settings(args)
        path_text = str(settings.progress_log_path).replace("\\", "/")
        self.assertIn("tmp/progress_logs/", path_text)
        self.assertTrue(path_text.endswith("/pipeline_progress.txt"))

    def test_load_settings_grok_tuning_from_env(self):
        args = make_args()
        env = {
            "Grok_API": "grok-key",
            "AZURE_ENDPOINT": "https://ocr.example.com",
            "AZURE_KEY": "azure-key",
            "OCR_GROK_MAX_RETRIES": "4",
            "OCR_GROK_TIMEOUT_NORMALIZE": "77",
            "OCR_GROK_TIMEOUT_REPAIR": "88",
        }
        with patch.dict(os.environ, env, clear=True):
            with patch.object(ocr_pipeline, "_load_env_file_fallback", return_value=None):
                settings = ocr_pipeline.load_settings(args)
        self.assertEqual(settings.grok_max_retries, 4)
        self.assertEqual(settings.grok_timeout_normalize, 77)
        self.assertEqual(settings.grok_timeout_repair, 88)

    def test_extract_qp_items_mcq_rejects_header_footer_and_periodic_table(self):
        ocr_payload = {
            "pages": [
                {
                    "page_number": 1,
                    "ocr_page_text": "READ THESE INSTRUCTIONS FIRST",
                    "lines": [
                        {"text": "1 hour READ THESE INSTRUCTIONS FIRST"},
                        {"text": "5070/12/M/J/16 [Turn over"},
                    ],
                },
                {
                    "page_number": 2,
                    "ocr_page_text": "Paper body",
                    "lines": [
                        {
                            "text": "1 Which row is correct? A first B second C third D fourth"
                        },
                        {
                            "text": "2 What happens next? A alpha B beta C gamma D delta"
                        },
                        {"text": "5070/12/M/J/16"},
                    ],
                },
                {
                    "page_number": 16,
                    "ocr_page_text": "The Periodic Table of Elements",
                    "lines": [
                        {"text": "3 Li lithium"},
                        {"text": "11 Na sodium"},
                    ],
                },
            ]
        }

        items, meta, candidates = ocr_pipeline.extract_qp_items_mcq(
            ocr_payload,
            expected_max_question=40,
        )
        question_numbers = [it["question_number"] for it in items]

        self.assertEqual(question_numbers, ["1", "2"])
        self.assertGreater(meta["candidate_summary"]["rejected"], 0)
        self.assertTrue(any(c.get("reason", "").startswith("page_excluded") for c in candidates))

    def test_extract_qp_items_mcq_handles_page_number_header_and_split_anchor(self):
        ocr_payload = {
            "pages": [
                {
                    "page_number": 2,
                    "ocr_page_text": "Body",
                    "lines": [
                        {"text": "2"},  # page number header
                        {"text": "1"},
                        {"text": "A student plans two experiments."},
                        {"text": "2 Copper wire is used to complete an electrical circuit."},
                    ],
                }
            ]
        }
        items, meta, candidates = ocr_pipeline.extract_qp_items_mcq(
            ocr_payload,
            expected_max_question=40,
        )
        q_numbers = [it["question_number"] for it in items]
        self.assertEqual(q_numbers, ["1", "2"])
        first = items[0]["question_text"].lower()
        self.assertIn("student plans", first)
        self.assertTrue(any(c.get("reason") == "page_number_header" for c in candidates))

    def test_extract_ms_items_mcq_parses_split_key_table(self):
        ocr_payload = {
            "pages": [
                {
                    "page_number": 2,
                    "lines": [
                        {"text": "Question Number Key"},
                        {"text": "1"},
                        {"text": "D"},
                        {"text": "2"},
                        {"text": "C"},
                        {"text": "11"},
                        {"text": "12"},
                        {"text": "B"},
                        {"text": "C"},
                        {"text": "Syllabus 5070 Paper 12"},
                        {"text": "13"},
                        {"text": "A"},
                    ],
                }
            ]
        }

        entries, parser_meta, _events = ocr_pipeline.extract_ms_items_mcq(
            ocr_payload,
            expected_max_question=40,
        )

        by_q = {int(e["question_number"]): e["marking_scheme"] for e in entries}
        self.assertEqual(by_q[1], "D")
        self.assertEqual(by_q[2], "C")
        self.assertEqual(by_q[11], "B")
        self.assertEqual(by_q[12], "C")
        self.assertEqual(by_q[13], "A")
        self.assertTrue(parser_meta["table_started"])

    def test_extract_ms_items_mcq_tracks_orphan_answers_without_backfill(self):
        ocr_payload = {
            "pages": [
                {
                    "page_number": 2,
                    "lines": [
                        {"text": "Question Number Key"},
                        {"text": "12"},
                        {"text": "C"},
                        {"text": "D"},
                        {"text": "12"},
                    ],
                }
            ]
        }
        entries, parser_meta, events = ocr_pipeline.extract_ms_items_mcq(
            ocr_payload,
            expected_max_question=40,
        )
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["question_number"], "12")
        self.assertEqual(entries[0]["marking_scheme"], "C")
        self.assertEqual(parser_meta.get("conflicts") or [], [])
        self.assertTrue(
            any(e.get("event") == "orphan_answer" for e in events)
        )
        self.assertEqual(parser_meta.get("pending_unanswered"), [12])

    def test_extract_ms_items_mcq_starts_from_question_header_without_key(self):
        ocr_payload = {
            "pages": [
                {
                    "page_number": 2,
                    "lines": [
                        {"text": "Question"},
                        {"text": "1"},
                        {"text": "C"},
                        {"text": "2"},
                        {"text": "D"},
                    ],
                }
            ]
        }
        entries, parser_meta, _events = ocr_pipeline.extract_ms_items_mcq(
            ocr_payload,
            expected_max_question=40,
        )
        by_q = {int(e["question_number"]): e["marking_scheme"] for e in entries}
        self.assertTrue(parser_meta["table_started"])
        self.assertEqual(by_q[1], "C")
        self.assertEqual(by_q[2], "D")

    def test_extract_ms_items_mcq_tracks_discounted_question(self):
        ocr_payload = {
            "pages": [
                {
                    "page_number": 2,
                    "lines": [
                        {"text": "Question"},
                        {"text": "25"},
                        {"text": "Question discounted"},
                        {"text": "26"},
                        {"text": "D"},
                    ],
                }
            ]
        }
        entries, parser_meta, _events = ocr_pipeline.extract_ms_items_mcq(
            ocr_payload,
            expected_max_question=40,
        )
        questions = [int(e["question_number"]) for e in entries]
        self.assertEqual(questions, [26])
        self.assertEqual(parser_meta.get("discounted_questions"), [25])

    def test_extract_items_structured_supports_top_level_anchors_and_page_suppression(self):
        ocr_payload = {
            "pages": [
                {
                    "page_number": 1,
                    "ocr_page_text": "READ THESE INSTRUCTIONS FIRST",
                    "lines": [{"text": "READ THESE INSTRUCTIONS FIRST"}],
                },
                {
                    "page_number": 2,
                    "ocr_page_text": "Question body",
                    "lines": [
                        {"text": "A1 Explain why methane is unreactive."},
                        {"text": "(a) State one use of methane."},
                        {"text": "A2 Calculate the moles of gas."},
                    ],
                },
            ]
        }
        items, meta, _candidates = ocr_pipeline.extract_items_structured(
            ocr_payload,
            item_text_field="question_text",
            allow_blank_rest=False,
            kind="qp",
        )
        self.assertEqual([it["question_number"] for it in items], ["1", "1", "2"])
        self.assertTrue(meta["page_filters"][0]["excluded"])
        self.assertEqual(meta["page_filters"][0]["reason"], "instructions_page")

    def test_extract_ms_items_structured_ignores_noise_rows(self):
        ocr_payload = {
            "pages": [
                {
                    "page_number": 2,
                    "ocr_page_text": "mark scheme body",
                    "lines": [
                        {"text": "Question Answer Marks"},
                        {"text": "1(a) Titration Measurements (1)"},
                        {"text": "12"},
                        {"text": "Page 2 of 4"},
                        {"text": "1(b) Concentration = 0.302 (1)"},
                        {"text": "General points"},
                        {"text": "2(a) white ppt (1)"},
                    ],
                }
            ]
        }
        items, meta, _candidates = ocr_pipeline.extract_ms_items_structured(ocr_payload)
        keys = [(it["question_number"], it.get("sub_question")) for it in items]
        self.assertIn(("1", "(a)"), keys)
        self.assertIn(("1", "(b)"), keys)
        self.assertIn(("2", "(a)"), keys)
        self.assertGreaterEqual(meta["candidate_summary"]["rejected"], 1)

    def test_compute_quality_metrics_uses_balanced_coverage(self):
        qp_items = [
            {"question_number": "1", "sub_question": None, "question_text": "Q1", "page_number": 1, "marks": 1},
            {"question_number": "2", "sub_question": None, "question_text": "Q2", "page_number": 1, "marks": 1},
        ]
        ms_items = [
            {"question_number": "1", "sub_question": None, "marking_scheme": "A", "page_number": 2, "marks": 1},
            {"question_number": "2", "sub_question": None, "marking_scheme": "B", "page_number": 2, "marks": 1},
            {"question_number": "3", "sub_question": None, "marking_scheme": "C", "page_number": 2, "marks": 1},
            {"question_number": "4", "sub_question": None, "marking_scheme": "D", "page_number": 2, "marks": 1},
        ]
        pairing = ocr_pipeline.pair_qp_ms(qp_items, ms_items, profile="mcq")
        metrics = ocr_pipeline.compute_quality_metrics(
            profile="mcq",
            qp_items=qp_items,
            ms_items=ms_items,
            pairing=pairing,
            expected_max_question=4,
            qp_parser_meta={"conflicts": []},
            ms_parser_meta={"conflicts": []},
        )
        self.assertEqual(metrics["coverage_qp"], 1.0)
        self.assertEqual(metrics["coverage_ms"], 0.5)
        self.assertEqual(metrics["coverage_balanced"], 0.5)

    def test_compute_quality_metrics_excludes_discounted_missing_questions(self):
        qp_items = [
            {"question_number": "1", "sub_question": None, "question_text": "Q1", "page_number": 1, "marks": 1},
            {"question_number": "2", "sub_question": None, "question_text": "Q2", "page_number": 1, "marks": 1},
            {"question_number": "3", "sub_question": None, "question_text": "Q3", "page_number": 1, "marks": 1},
            {"question_number": "4", "sub_question": None, "question_text": "Q4", "page_number": 1, "marks": 1},
        ]
        ms_items = [
            {"question_number": "1", "sub_question": None, "marking_scheme": "A", "page_number": 2, "marks": 1},
            {"question_number": "3", "sub_question": None, "marking_scheme": "C", "page_number": 2, "marks": 1},
            {"question_number": "4", "sub_question": None, "marking_scheme": "D", "page_number": 2, "marks": 1},
        ]
        pairing = ocr_pipeline.pair_qp_ms(qp_items, ms_items, profile="mcq")
        metrics = ocr_pipeline.compute_quality_metrics(
            profile="mcq",
            qp_items=qp_items,
            ms_items=ms_items,
            pairing=pairing,
            expected_max_question=4,
            qp_parser_meta={"conflicts": []},
            ms_parser_meta={"conflicts": [], "discounted_questions": [2]},
        )
        self.assertEqual(metrics.get("discounted_questions"), [2])
        self.assertNotIn(2, metrics.get("missing_ms_questions", []))

    def test_determine_status_rejects_duplicate_qp_ids(self):
        settings = build_settings()
        quality_metrics = {
            "profile": "structured",
            "counts": {
                "invalid_qp_ids": 0,
                "invalid_ms_ids": 0,
                "duplicate_qp_ids": 1,
                "duplicate_ms_ids": 0,
                "pairing_issues": 0,
                "parser_conflicts": 0,
                "missing_qp_questions": 0,
                "missing_ms_questions": 0,
            },
            "coverage_qp": 1.0,
            "coverage_ms": 1.0,
            "coverage_balanced": 1.0,
        }
        status, _score, reasons, _validation = ocr_pipeline.determine_status(
            qp_confidence=0.95,
            ms_confidence=0.95,
            matched_count=10,
            qp_count=10,
            ms_count=10,
            issues=[],
            quality_metrics=quality_metrics,
            settings=settings,
        )
        self.assertEqual(status, "review_required")
        self.assertIn("duplicate_qp_ids_detected", reasons)

    def test_pair_qp_ms_no_positional_fallback(self):
        qp_items = [
            {
                "question_number": "1",
                "sub_question": None,
                "question_text": "Q1",
                "page_number": 1,
                "marks": 1,
            },
            {
                "question_number": "2",
                "sub_question": None,
                "question_text": "Q2",
                "page_number": 1,
                "marks": 1,
            },
        ]
        ms_items = [
            {
                "question_number": "1",
                "sub_question": None,
                "marking_scheme": "A",
                "page_number": 2,
                "marks": 1,
            },
            {
                "question_number": "5",
                "sub_question": None,
                "marking_scheme": "B",
                "page_number": 2,
                "marks": 1,
            },
        ]

        paired = ocr_pipeline.pair_qp_ms(qp_items, ms_items, profile="mcq")
        self.assertEqual(paired["matched_count"], 1)
        self.assertEqual(len(paired["unmatched_qp"]), 1)
        self.assertEqual(paired["unmatched_qp"][0]["question_number"], "2")
        self.assertEqual(len(paired["unmatched_ms"]), 1)
        self.assertEqual(paired["unmatched_ms"][0]["question_number"], "5")

    def test_repair_trigger_reasons_mcq(self):
        quality_metrics = {
            "counts": {
                "missing_qp_questions": 2,
                "missing_ms_questions": 1,
                "parser_conflicts": 1,
            },
            "unmatched_qp_count": 3,
            "unmatched_ms_count": 0,
        }
        reasons = ocr_pipeline._repair_trigger_reasons("mcq", quality_metrics)
        self.assertIn("missing_qp_questions", reasons)
        self.assertIn("missing_ms_questions", reasons)
        self.assertIn("parser_conflicts", reasons)
        self.assertIn("unmatched_qp", reasons)

    def test_sanitize_mcq_ms_items_handles_noisy_values(self):
        raw_items = [
            {"question_number": "1", "page_number": 2, "marking_scheme": "A C Cambridge"},
            {"question_number": "2", "page_number": 2, "marking_scheme": "B"},
            {"question_number": "2", "page_number": 2, "marking_scheme": "C"},
            {"question_number": "507", "page_number": 2, "marking_scheme": "D"},
        ]
        items, conflicts, duplicates = ocr_pipeline._sanitize_mcq_ms_items(
            raw_items,
            expected_max_question=40,
        )
        by_q = {int(e["question_number"]): e["marking_scheme"] for e in items}
        self.assertEqual(by_q[1], "A")
        self.assertEqual(by_q[2], "B")
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(len(duplicates), 0)

    def test_maybe_normalize_with_grok_rejects_mcq_regression(self):
        settings = build_settings()
        settings.expected_max_question = 4
        original_items = [
            {"question_number": "1", "sub_question": None, "page_number": 1, "marks": 1, "question_text": "Q1 A B C D"},
            {"question_number": "2", "sub_question": None, "page_number": 1, "marks": 1, "question_text": "Q2 A B C D"},
            {"question_number": "3", "sub_question": None, "page_number": 1, "marks": 1, "question_text": "Q3 A B C D"},
            {"question_number": "4", "sub_question": None, "page_number": 1, "marks": 1, "question_text": "Q4 A B C D"},
        ]
        grok_payload = {
            "items": [
                {
                    "question_number": "1",
                    "sub_question": None,
                    "page_number": 1,
                    "marks": 1,
                    "question_text": "Q1 shrunk",
                }
            ]
        }
        with patch.object(
            ocr_pipeline,
            "_call_grok_json",
            return_value=(grok_payload, {"success": True, "attempts": [{"attempt": 1}]}),
        ):
            items, issues, debug = ocr_pipeline.maybe_normalize_with_grok(
                kind="qp",
                profile="mcq",
                items=list(original_items),
                ocr_payload=sample_ocr_payload(),
                settings=settings,
            )
        self.assertEqual(len(items), 4)
        self.assertIn("grok_normalization_rejected_regression", issues)
        self.assertEqual(debug.get("result", {}).get("before_metrics", {}).get("unique_count"), 4)
        self.assertEqual(debug.get("result", {}).get("after_metrics", {}).get("unique_count"), 1)

    def test_maybe_repair_pair_with_grok_applies_improved_candidate(self):
        pair = ocr_pipeline.PairMeta(
            subject="Chemistry 1011",
            year=2016,
            session="May_June",
            paper="Paper_1",
            variant="Variant_2",
            qp_path=Path("OA_MAIN_DATASET/Chemistry 1011/2016/May_June/Paper_1/Variant_2/qp.pdf"),
            ms_path=Path("OA_MAIN_DATASET/Chemistry 1011/2016/May_June/Paper_1/Variant_2/ms.pdf"),
        )
        settings = build_settings()
        settings.expected_max_question = 4

        qp_items = [
            {"question_number": "3", "sub_question": None, "page_number": 2, "marks": 1, "question_text": "Q3 A B C D"},
            {"question_number": "4", "sub_question": None, "page_number": 2, "marks": 1, "question_text": "Q4 A B C D"},
        ]
        ms_items = [
            {"question_number": "1", "sub_question": None, "page_number": 2, "marks": 1, "marking_scheme": "A"},
            {"question_number": "2", "sub_question": None, "page_number": 2, "marks": 1, "marking_scheme": "B"},
            {"question_number": "3", "sub_question": None, "page_number": 2, "marks": 1, "marking_scheme": "C"},
            {"question_number": "4", "sub_question": None, "page_number": 2, "marks": 1, "marking_scheme": "D"},
        ]
        baseline_pairing = ocr_pipeline.pair_qp_ms(qp_items, ms_items, profile="mcq")
        baseline_quality = ocr_pipeline.compute_quality_metrics(
            profile="mcq",
            qp_items=qp_items,
            ms_items=ms_items,
            pairing=baseline_pairing,
            expected_max_question=4,
            qp_parser_meta={"conflicts": []},
            ms_parser_meta={"conflicts": []},
        )

        repaired_payload = {
            "qp_items": [
                {"question_number": "1", "page_number": 1, "question_text": "Q1 stem A B C D", "marks": 1},
                {"question_number": "2", "page_number": 1, "question_text": "Q2 stem A B C D", "marks": 1},
                {"question_number": "3", "page_number": 2, "question_text": "Q3 stem A B C D", "marks": 1},
                {"question_number": "4", "page_number": 2, "question_text": "Q4 stem A B C D", "marks": 1},
            ],
            "ms_items": [
                {"question_number": "1", "page_number": 2, "marking_scheme": "A", "marks": 1},
                {"question_number": "2", "page_number": 2, "marking_scheme": "B", "marks": 1},
                {"question_number": "3", "page_number": 2, "marking_scheme": "C", "marks": 1},
                {"question_number": "4", "page_number": 2, "marking_scheme": "D", "marks": 1},
            ],
        }

        with patch.object(
            ocr_pipeline,
            "_call_grok_json",
            return_value=(repaired_payload, {"success": True}),
        ):
            result = ocr_pipeline.maybe_repair_pair_with_grok(
                pair=pair,
                profile="mcq",
                qp_items=qp_items,
                ms_items=ms_items,
                qp_ocr_payload=sample_ocr_payload(),
                ms_ocr_payload=sample_ocr_payload(),
                baseline_pairing=baseline_pairing,
                baseline_quality_metrics=baseline_quality,
                settings=settings,
            )

        self.assertTrue(result["debug"]["applied"])
        self.assertIn("grok_pair_repair_applied", result["issues"])
        self.assertEqual(len(result["qp_items"]), 4)
        self.assertEqual(result["pairing"]["matched_count"], 4)

    def test_maybe_repair_pair_with_grok_structured_preserves_qp_ms_conflicts_on_skip(self):
        pair = ocr_pipeline.PairMeta(
            subject="Chemistry 1011",
            year=2015,
            session="May_June",
            paper="Paper_2",
            variant="Variant_1",
            qp_path=Path("OA_MAIN_DATASET/Chemistry 1011/2015/May_June/Paper_2/Variant_1/qp.pdf"),
            ms_path=Path("OA_MAIN_DATASET/Chemistry 1011/2015/May_June/Paper_2/Variant_1/ms.pdf"),
        )
        settings = build_settings()
        settings.use_grok_normalization = False

        qp_items = [
            {
                "question_number": "1",
                "sub_question": "(a)",
                "page_number": 1,
                "marks": 1,
                "question_text": "State one property.",
            }
        ]
        ms_items = [
            {
                "question_number": "1",
                "sub_question": "(a)",
                "page_number": 2,
                "marks": 1,
                "marking_scheme": "Any valid property.",
            }
        ]
        baseline_pairing = ocr_pipeline.pair_qp_ms(qp_items, ms_items, profile="structured")
        baseline_quality = ocr_pipeline.compute_quality_metrics(
            profile="structured",
            qp_items=qp_items,
            ms_items=ms_items,
            pairing=baseline_pairing,
            expected_max_question=40,
            qp_parser_meta={"conflicts": [{"type": "qp_conflict"}]},
            ms_parser_meta={"conflicts": [{"type": "ms_conflict"}]},
        )

        result = ocr_pipeline.maybe_repair_pair_with_grok(
            pair=pair,
            profile="structured",
            qp_items=qp_items,
            ms_items=ms_items,
            qp_ocr_payload=sample_ocr_payload(),
            ms_ocr_payload=sample_ocr_payload(),
            baseline_pairing=baseline_pairing,
            baseline_quality_metrics=baseline_quality,
            settings=settings,
        )

        self.assertEqual(result["qp_conflicts"], [{"type": "qp_conflict"}])
        self.assertEqual(result["ms_conflicts"], [{"type": "ms_conflict"}])
        self.assertEqual(result["debug"].get("skipped_reason"), "disabled")

    def test_write_pair_debug_files_switch(self):
        pair = ocr_pipeline.PairMeta(
            subject="Chemistry 1011",
            year=2016,
            session="May_June",
            paper="Paper_1",
            variant="Variant_2",
            qp_path=Path("OA_MAIN_DATASET/Chemistry 1011/2016/May_June/Paper_1/Variant_2/qp.pdf"),
            ms_path=Path("OA_MAIN_DATASET/Chemistry 1011/2016/May_June/Paper_1/Variant_2/ms.pdf"),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            settings_off = build_settings(debug_enabled=False, debug_dir=tmpdir)
            out = ocr_pipeline.write_pair_debug_files(
                settings=settings_off,
                pair=pair,
                qp_ocr_payload=sample_ocr_payload(),
                ms_ocr_payload=sample_ocr_payload(),
                qp_page_filters=[],
                anchor_candidates_qp=[],
                anchor_candidates_ms=[],
                normalization_debug={},
                pairing_debug={},
                validation_debug={},
                grok_debug={},
            )
            self.assertIsNone(out)

            settings_on = build_settings(debug_enabled=True, debug_dir=tmpdir)
            out = ocr_pipeline.write_pair_debug_files(
                settings=settings_on,
                pair=pair,
                qp_ocr_payload=sample_ocr_payload(),
                ms_ocr_payload=sample_ocr_payload(),
                qp_page_filters=[],
                anchor_candidates_qp=[],
                anchor_candidates_ms=[],
                normalization_debug={"x": 1},
                pairing_debug={"x": 1},
                validation_debug={"x": 1},
                grok_debug={"x": 1},
            )
            self.assertIsNotNone(out)
            self.assertTrue((out / "qp_ocr_debug.json").exists())
            self.assertTrue((out / "pair_debug_manifest.json").exists())

    def test_debug_level_basic_vs_full(self):
        pair = ocr_pipeline.PairMeta(
            subject="Chemistry 1011",
            year=2016,
            session="May_June",
            paper="Paper_1",
            variant="Variant_2",
            qp_path=Path("OA_MAIN_DATASET/Chemistry 1011/2016/May_June/Paper_1/Variant_2/qp.pdf"),
            ms_path=Path("OA_MAIN_DATASET/Chemistry 1011/2016/May_June/Paper_1/Variant_2/ms.pdf"),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            basic_settings = build_settings(
                debug_enabled=True,
                debug_level="basic",
                debug_dir=tmpdir,
            )
            basic_out = ocr_pipeline.write_pair_debug_files(
                settings=basic_settings,
                pair=pair,
                qp_ocr_payload=sample_ocr_payload(),
                ms_ocr_payload=sample_ocr_payload(),
                qp_page_filters=[],
                anchor_candidates_qp=[],
                anchor_candidates_ms=[],
                normalization_debug={},
                pairing_debug={},
                validation_debug={},
                grok_debug={},
            )
            with open(basic_out / "qp_ocr_debug.json", "r", encoding="utf-8") as f:
                basic_payload = json.load(f)
            self.assertNotIn("raw_ocr", basic_payload)
            self.assertIn("page_summaries", basic_payload)

            full_settings = build_settings(
                debug_enabled=True,
                debug_level="full",
                debug_dir=tmpdir,
            )
            full_settings.debug_run_id = "20260312T010000Z"
            full_out = ocr_pipeline.write_pair_debug_files(
                settings=full_settings,
                pair=pair,
                qp_ocr_payload=sample_ocr_payload(),
                ms_ocr_payload=sample_ocr_payload(),
                qp_page_filters=[],
                anchor_candidates_qp=[],
                anchor_candidates_ms=[],
                normalization_debug={},
                pairing_debug={},
                validation_debug={},
                grok_debug={},
            )
            with open(full_out / "qp_ocr_debug.json", "r", encoding="utf-8") as f:
                full_payload = json.load(f)
            self.assertIn("raw_ocr", full_payload)

    def test_progress_logger_writes_realtime_and_final_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "progress.txt"
            logger = ocr_pipeline.RunProgressLogger(
                enabled=True,
                log_path=log_path,
                run_id="run123",
            )
            logger.write_header({"a": 1})
            with logger.step("step_a", pair_id="pair1", message="testing"):
                pass
            logger.warn("pair1", "step_a", "warn-msg")
            logger.error("pair1", "step_a", "err-msg")
            logger.record_pair_status("accepted")
            logger.record_pair_timing("pair1", 1.23)
            logger.write_final_report({"summary": {"accepted": 1}})
            logger.close()

            text = log_path.read_text(encoding="utf-8")
            self.assertIn("STEP_START", text)
            self.assertIn("STEP_END", text)
            self.assertIn("STEP_WARN", text)
            self.assertIn("STEP_ERROR", text)
            self.assertIn("OCR PIPELINE FINAL REPORT", text)
            self.assertIn("total_elapsed_seconds=", text)

    def test_call_grok_json_reports_json_decode_metadata(self):
        class FakeResp:
            status_code = 200
            text = "ok"

            def json(self):
                return {"choices": [{"message": {"content": "{\"items\": [}"}}]}

        with patch.object(ocr_pipeline.requests, "post", return_value=FakeResp()):
            with self.assertRaises(ocr_pipeline.GrokCallError) as ctx:
                ocr_pipeline._call_grok_json(
                    grok_api_key="xai-key",
                    model="grok-4-1-fast-reasoning",
                    system_prompt="sys",
                    user_payload={"x": 1},
                    timeout=1,
                    max_retries=1,
                )
        debug = ctx.exception.debug
        self.assertIn("attempts", debug)
        self.assertEqual(len(debug["attempts"]), 1)
        self.assertEqual(debug["attempts"][0].get("error_class"), "json_decode_error")
        self.assertIn("json_error", debug["attempts"][0])

    def test_call_grok_json_repairs_trailing_comma_payload(self):
        class FakeResp:
            status_code = 200
            text = "ok"

            def json(self):
                return {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"items":[{"question_number":"1","page_number":1,"marks":1,'
                                    '"question_text":"ok"},],}'
                                )
                            }
                        }
                    ]
                }

        with patch.object(ocr_pipeline.requests, "post", return_value=FakeResp()):
            parsed, debug = ocr_pipeline._call_grok_json(
                grok_api_key="xai-key",
                model="grok-4-1-fast-reasoning",
                system_prompt="sys",
                user_payload={"x": 1},
                timeout=1,
                max_retries=1,
            )
        self.assertIn("items", parsed)
        self.assertEqual(parsed["items"][0]["question_number"], "1")
        self.assertTrue(debug["attempts"][0].get("json_repaired"))


if __name__ == "__main__":
    unittest.main()
