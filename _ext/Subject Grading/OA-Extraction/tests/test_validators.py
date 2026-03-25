from __future__ import annotations

from oa_extraction.config import Settings
from oa_extraction.types import (
    DisagreementSpan,
    ExtractionDiagnostics,
    OCRCandidate,
    OCREngine,
    RepairAction,
    SubjectLabel,
)
from oa_extraction.validators import build_confidence, validate_diagnostics, validate_extraction


def _settings() -> Settings:
    return Settings(
        api_key="test-key",
        base_url="https://api.x.ai/v1",
        model="grok-test",
        timeout_seconds=30,
        max_retries=0,
        ocr_confidence_threshold=0.85,
        split_confidence_threshold=0.90,
        classification_confidence_threshold=0.80,
        azure_endpoint="https://example.cognitiveservices.azure.com",
        azure_api_key="azure-test-key",
        azure_api_version="2024-11-30",
        enable_azure_fallback=True,
        grok_fallback_ocr_threshold=0.90,
        grok_fallback_split_threshold=0.92,
        enable_image_variants=True,
        enable_targeted_repair=True,
        engine_disagreement_threshold=0.08,
        repair_confidence_threshold=0.85,
        selection_score_margin=0.05,
    )


def test_math_log_base_mismatch_is_flagged() -> None:
    flags = validate_extraction(
        whole_text_raw="Solve log_3 x = 9\nx = log_2 9",
        question_raw="Solve log_3 x = 9",
        answer_raw="x = log_2 9",
        subject=SubjectLabel.MATH,
        confidence=build_confidence(0.95, 0.95, 0.95),
        settings=_settings(),
    )

    assert "potential_log_base_mismatch" in {flag.code for flag in flags}


def test_subject_specific_english_punctuation_loss_is_flagged() -> None:
    flags = validate_extraction(
        whole_text_raw='Explain the phrase "to be or not to be',
        question_raw='Explain the phrase "to be or not to be',
        answer_raw="It expresses indecision.",
        subject=SubjectLabel.ENGLISH,
        confidence=build_confidence(0.96, 0.96, 0.96),
        settings=_settings(),
    )

    assert "potential_english_punctuation_loss" in {flag.code for flag in flags}


def test_validate_diagnostics_flags_engine_disagreement_and_repair() -> None:
    diagnostics = ExtractionDiagnostics(
        selected_ocr_engine=OCREngine.GROK,
        selected_variant="original",
        ocr_candidates=[
            OCRCandidate(
                engine=OCREngine.GROK,
                variant="original",
                full_text="Solve log_3 x = 9",
                lines=[],
                ocr_confidence=0.95,
                selection_score=0.92,
            )
        ],
        disagreement_spans=[
            DisagreementSpan(
                page_number=1,
                line_index=1,
                selected_text="Solve log_3 x = 9",
                alternate_texts=["Solve log3 x = 9"],
                disagreement_score=0.25,
                unresolved=True,
            )
        ],
        repair_actions=[
            RepairAction(
                page_number=1,
                line_index=1,
                before_text="Solve log3 x = 9",
                after_text="Solve log_3 x = 9",
                source="grok_repair",
                accepted=True,
                confidence=0.94,
                rationale="Restore the base subscript.",
            )
        ],
        selection_reasons=["Selected Grok original candidate."],
    )

    flags = validate_diagnostics(diagnostics, _settings())
    codes = {flag.code for flag in flags}

    assert "ocr_engine_disagreement" in codes
    assert "unresolved_ocr_conflicts" in codes
    assert "targeted_repair_applied" in codes
