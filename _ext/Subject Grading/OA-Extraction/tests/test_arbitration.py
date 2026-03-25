from __future__ import annotations

from oa_extraction.arbitration import detect_disagreement_spans, score_candidates
from oa_extraction.types import LineOCR, OCRCandidate, OCREngine


def _candidate(engine: OCREngine, variant: str, full_text: str, confidence: float) -> OCRCandidate:
    lines = [
        LineOCR(page_number=1, line_index=index, text=line, confidence=confidence)
        for index, line in enumerate([part for part in full_text.splitlines() if part.strip()], start=1)
    ]
    return OCRCandidate(
        engine=engine,
        variant=variant,
        full_text=full_text,
        lines=lines,
        ocr_confidence=confidence,
    )


def test_score_candidates_prefers_better_symbol_preservation() -> None:
    weak = _candidate(OCREngine.GROK, "original", "Solve log3 x 9 x 81", 0.86)
    strong = _candidate(OCREngine.AZURE, "original", "Solve log_3 x = 9\nx = 3^2", 0.90)

    ranked, _ = score_candidates([weak, strong])

    assert ranked[0].full_text == strong.full_text
    assert (ranked[0].selection_score or 0.0) > (ranked[1].selection_score or 0.0)


def test_detect_disagreement_spans_finds_line_level_conflicts() -> None:
    selected = _candidate(OCREngine.GROK, "original", "Q. Solve log_3 x = 9\nA. x = 3^2", 0.95)
    alternate = _candidate(OCREngine.AZURE, "original", "Q. Solve log3 x = 9\nA. x = 3^2", 0.96)

    disagreements = detect_disagreement_spans(selected, [alternate])

    assert len(disagreements) == 1
    assert disagreements[0].line_index == 1
    assert "log3 x = 9" in disagreements[0].alternate_texts[0]
