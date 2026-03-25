"""Repository for loading accepted OCR-derived question records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .config import PipelineConfig
from .schemas import EvaluateRequest, QuestionRecord


def _norm_qnum(value: object) -> str:
    text = str(value or "").strip()
    if text.isdigit():
        return str(int(text))
    return text


def _norm_sub(value: object) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    if not text.startswith("("):
        text = f"({text})"
    return text


def filter_question_records(
    records: Iterable[QuestionRecord],
    request: EvaluateRequest,
) -> List[QuestionRecord]:
    result = list(records)
    if request.subject:
        subject = request.subject.strip().casefold()
        result = [r for r in result if r.subject.casefold() == subject]
    if request.year is not None:
        result = [r for r in result if int(r.year) == int(request.year)]
    if request.session:
        session = request.session.strip().casefold()
        result = [r for r in result if r.session.casefold() == session]
    if request.paper:
        paper = request.paper.strip().casefold()
        result = [r for r in result if r.paper.casefold() == paper]
    if request.variant:
        variant = request.variant.strip().casefold()
        result = [r for r in result if r.variant.casefold() == variant]
    return result


class DatasetRepository:
    """Loads and caches QuestionRecord items from accepted extracted pairs."""

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self._records: List[QuestionRecord] = []
        self._by_id: Dict[str, QuestionRecord] = {}
        self._loaded = False

    def reload(self) -> None:
        self._records = self._load_records()
        self._by_id = {r.question_id: r for r in self._records}
        self._loaded = True

    def get_records(self) -> List[QuestionRecord]:
        if not self._loaded:
            self.reload()
        return list(self._records)

    def get_by_question_id(self, question_id: str) -> Optional[QuestionRecord]:
        if not self._loaded:
            self.reload()
        return self._by_id.get(question_id)

    def filter_records(
        self,
        records: Iterable[QuestionRecord],
        request: EvaluateRequest,
    ) -> List[QuestionRecord]:
        return filter_question_records(records, request)

    def _load_records(self) -> List[QuestionRecord]:
        dataset_root = self.config.dataset_root
        if not dataset_root.exists():
            return []

        all_records: List[QuestionRecord] = []
        for summary_path in sorted(dataset_root.rglob(self.config.pair_summary_filename)):
            pair_records = self._load_pair_records(summary_path)
            all_records.extend(pair_records)
        return all_records

    def _load_pair_records(self, summary_path: Path) -> List[QuestionRecord]:
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            return []

        if str(summary.get("status") or "").strip().lower() != "accepted":
            return []

        meta = summary.get("metadata") or {}
        subject = str(meta.get("subject") or "").strip()
        session = str(meta.get("session") or "").strip()
        paper = str(meta.get("paper") or "").strip()
        variant = str(meta.get("variant") or "").strip()
        year_raw = meta.get("year")
        try:
            year = int(year_raw)
        except Exception:
            return []

        variant_dir = summary_path.parent
        qp_path = variant_dir / self.config.qp_extracted_filename
        ms_path = variant_dir / self.config.ms_extracted_filename
        if not qp_path.exists() or not ms_path.exists():
            return []

        try:
            qp_payload = json.loads(qp_path.read_text(encoding="utf-8"))
            ms_payload = json.loads(ms_path.read_text(encoding="utf-8"))
        except Exception:
            return []

        qp_questions = qp_payload.get("questions") or []
        ms_entries = ms_payload.get("marking_entries") or []
        if not isinstance(qp_questions, list) or not isinstance(ms_entries, list):
            return []

        ms_by_key: Dict[Tuple[str, str], Dict[str, object]] = {}
        for entry in ms_entries:
            if not isinstance(entry, dict):
                continue
            key = (_norm_qnum(entry.get("question_number")), _norm_sub(entry.get("sub_question")))
            if not key[0]:
                continue
            ms_by_key.setdefault(key, entry)

        source_ref = "/".join([subject, str(year), session, paper, variant])
        records: List[QuestionRecord] = []
        for qp in qp_questions:
            if not isinstance(qp, dict):
                continue

            q_num = _norm_qnum(qp.get("question_number"))
            sub = _norm_sub(qp.get("sub_question"))
            if not q_num:
                continue
            ms_entry = ms_by_key.get((q_num, sub))
            if ms_entry is None and not sub:
                # Common MCQ fallback.
                ms_entry = ms_by_key.get((q_num, ""))
            if not ms_entry:
                continue

            question_text = str(qp.get("question_text") or "").strip()
            marking_scheme = str(ms_entry.get("marking_scheme") or "").strip()
            if not question_text or not marking_scheme:
                continue

            sub_question = sub or None
            page_number = qp.get("page_number")
            try:
                page_number = int(page_number) if page_number is not None else None
            except Exception:
                page_number = None

            suffix = q_num if sub_question is None else f"{q_num}{sub_question}"
            question_id = "|".join(
                [
                    subject,
                    str(year),
                    session,
                    paper,
                    variant,
                    suffix,
                ]
            )

            records.append(
                QuestionRecord(
                    question_id=question_id,
                    subject=subject,
                    year=year,
                    session=session,
                    paper=paper,
                    variant=variant,
                    question_number=q_num,
                    sub_question=sub_question,
                    question_text=question_text,
                    marking_scheme_answer=marking_scheme,
                    page_number=page_number,
                    source_paper_reference=source_ref,
                )
            )
        return records
