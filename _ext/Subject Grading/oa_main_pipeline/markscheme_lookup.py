"""Markscheme/reference lookup helpers."""

from __future__ import annotations

from dataclasses import dataclass

from .schemas import QuestionRecord


@dataclass(frozen=True)
class MarkschemeLookupResult:
    source_paper_reference: str
    marking_scheme_answer: str
    page_number: int | None


def lookup_markscheme(record: QuestionRecord) -> MarkschemeLookupResult:
    return MarkschemeLookupResult(
        source_paper_reference=record.source_paper_reference,
        marking_scheme_answer=record.marking_scheme_answer,
        page_number=record.page_number,
    )

