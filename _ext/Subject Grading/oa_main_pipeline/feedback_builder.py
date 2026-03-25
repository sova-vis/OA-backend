"""Feedback builder for deterministic evaluator outputs."""

from __future__ import annotations

from typing import Sequence

from .schemas import GradeLabel


def build_feedback(
    *,
    grade_label: GradeLabel,
    score_percent: float,
    expected_points: Sequence[str],
    missing_points: Sequence[str],
    is_mcq: bool,
    correct_option: str | None = None,
) -> str:
    if is_mcq:
        if grade_label == "fully_correct":
            return "Correct. Your selected option matches the marking scheme."
        if correct_option:
            return f"Incorrect option. Correct answer is {correct_option}."
        return "Incorrect option based on the marking scheme."

    if grade_label == "fully_correct":
        return f"Strong answer ({score_percent:.1f}%). You covered the key marking points."
    if grade_label == "partially_correct":
        if missing_points:
            preview = "; ".join(missing_points[:2])
            return (
                f"Partially correct ({score_percent:.1f}%). "
                f"Missing key points: {preview}"
            )
        return f"Partially correct ({score_percent:.1f}%). Include more marking points."
    if missing_points:
        preview = "; ".join(missing_points[:2])
        return f"Weak alignment ({score_percent:.1f}%). Major points missing: {preview}"
    if expected_points:
        preview = "; ".join(expected_points[:2])
        return f"Weak alignment ({score_percent:.1f}%). Expected points include: {preview}"
    return f"Weak alignment ({score_percent:.1f}%)."

