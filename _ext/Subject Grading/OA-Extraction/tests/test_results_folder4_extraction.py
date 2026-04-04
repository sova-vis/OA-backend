"""Live extraction check for Results/4 handwritten fixture (requires Grok API)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from oa_extraction import extract_qa
from oa_extraction.config import _load_dotenv_if_present
from oa_extraction.types import ExtractionResult


def _find_results_folder_4() -> Path:
    start = Path(__file__).resolve()
    for parent in start.parents:
        case = parent / "Results" / "4"
        expected = case / "extraction_expected.json"
        if expected.is_file():
            return case
    pytest.skip("Results/4/extraction_expected.json not found (expected Propelify workspace root).")


def _normalize_corrupted_frac(answer: str) -> str:
    """Some model outputs use U+000C (form feed) where LaTeX \\f starts, yielding \\x0crac{...} instead of \\frac{...}."""
    return answer.replace("\x0crac", "\\frac")


def _write_results_4_artifact(
    case_dir: Path, spec: dict, image_name: str, result: ExtractionResult
) -> Path:
    """Persist last run to Results/4 for inspection (same folder as the image)."""
    out_path = case_dir / "extraction_last_run.json"
    payload = {
        "meta": {
            "case_id": spec.get("case_id"),
            "image_file": image_name,
            "artifact": "written by OA-Extraction integration test",
        },
        "extraction": result.model_dump(mode="json"),
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_path


@pytest.mark.integration
def test_extraction_results_folder4_matches_expected_json() -> None:
    oa_extraction_root = Path(__file__).resolve().parent.parent
    _load_dotenv_if_present(oa_extraction_root)
    if not (os.getenv("Grok_API") or os.getenv("XAI_API_KEY")):
        pytest.skip(
            "Set Grok_API or XAI_API_KEY (or add them to OA-Extraction/.env) to run live extraction."
        )

    case_dir = _find_results_folder_4()
    spec = json.loads((case_dir / "extraction_expected.json").read_text(encoding="utf-8"))
    image_name = spec["image_file"]
    image_path = case_dir / image_name
    if not image_path.is_file():
        pytest.fail(f"Missing image file: {image_path}")

    expect = spec["expect"]
    result = extract_qa(str(image_path))
    _write_results_4_artifact(case_dir, spec, image_name, result)

    assert result.input_type == expect["input_type"]
    assert result.page_count == expect["page_count"]
    assert str(result.subject) == expect["subject"]

    q_lower = result.question_raw.lower()
    for needle in expect["question_contains"]:
        assert needle.lower() in q_lower, f"question_raw missing {needle!r}: {result.question_raw!r}"

    assert len(result.answer_raw.strip()) >= expect["answer_min_length"]

    rules = expect["quality_rules"]
    answer = result.answer_raw
    answer_for_markers = _normalize_corrupted_frac(answer)
    markers = rules["answer_contains_substrings_any_of"]
    has_explicit_division = any(m in answer_for_markers for m in markers)
    flag_codes = {f.code for f in result.flags}
    wanted_flags = set(rules["or_flag_codes_any_of"])
    has_wanted_flag = bool(flag_codes & wanted_flags)

    assert has_explicit_division or has_wanted_flag, (
        "Expected either explicit / or \\frac in answer, or flag "
        f"{wanted_flags!r}; got answer={answer!r} flags={flag_codes!r}"
    )
