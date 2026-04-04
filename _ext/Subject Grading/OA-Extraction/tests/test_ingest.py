"""Tests for PDF page selection in ingest."""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from oa_extraction.config import Settings
from oa_extraction.ingest import load_document
from oa_extraction.types import InputDocumentError


def _minimal_settings() -> Settings:
    return Settings(
        api_key="k",
        base_url="https://api.x.ai/v1",
        model="m",
        timeout_seconds=30.0,
        max_retries=0,
        ocr_confidence_threshold=0.85,
        split_confidence_threshold=0.9,
        classification_confidence_threshold=0.8,
        azure_endpoint=None,
        azure_api_key=None,
        azure_api_version="2024-11-30",
        enable_azure_fallback=False,
        grok_fallback_ocr_threshold=0.9,
        grok_fallback_split_threshold=0.92,
        enable_image_variants=False,
        enable_targeted_repair=False,
        engine_disagreement_threshold=0.08,
        repair_confidence_threshold=0.85,
        selection_score_margin=0.05,
    )


def test_load_pdf_all_pages_when_page_number_omitted(tmp_path: Path) -> None:
    pdf_path = tmp_path / "two.pdf"
    doc = fitz.open()
    try:
        doc.new_page().insert_text((72, 72), "P1")
        doc.new_page().insert_text((72, 72), "P2")
        doc.save(pdf_path)
    finally:
        doc.close()

    loaded = load_document(pdf_path, _minimal_settings())
    assert loaded.input_type == "pdf"
    assert loaded.page_count == 2


def test_load_pdf_single_page_by_number(tmp_path: Path) -> None:
    pdf_path = tmp_path / "two.pdf"
    doc = fitz.open()
    try:
        doc.new_page().insert_text((72, 72), "P1")
        doc.new_page().insert_text((72, 72), "P2")
        doc.save(pdf_path)
    finally:
        doc.close()

    loaded = load_document(pdf_path, _minimal_settings(), page_number=2)
    assert loaded.page_count == 1
    assert loaded.pages[0].page_number == 2


def test_load_pdf_page_number_out_of_range(tmp_path: Path) -> None:
    pdf_path = tmp_path / "one.pdf"
    doc = fitz.open()
    try:
        doc.new_page()
        doc.save(pdf_path)
    finally:
        doc.close()

    with pytest.raises(InputDocumentError, match="between 1 and 1"):
        load_document(pdf_path, _minimal_settings(), page_number=3)

