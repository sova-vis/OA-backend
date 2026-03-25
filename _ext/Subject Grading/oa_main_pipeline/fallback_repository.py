"""Repository for loading fallback question records from O_LEVEL_JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .config import PipelineConfig
from .dataset_repository import _norm_qnum, _norm_sub, filter_question_records
from .schemas import EvaluateRequest, QuestionRecord


def _to_int(value: object) -> Optional[int]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return int(text)
    except Exception:
        return None


class FallbackDatasetRepository:
    """Loads and caches QuestionRecord items from O_LEVEL_JSON fallback files."""

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
        root = self.config.fallback_root
        if not root.exists():
            return []

        out: List[QuestionRecord] = []
        for subject_dir in sorted(root.iterdir()):
            if not subject_dir.is_dir():
                continue
            subject = subject_dir.name.strip()
            if not subject:
                continue
            for json_path in sorted(subject_dir.glob("*.json")):
                out.extend(self._load_file(subject=subject, json_path=json_path))
        return out

    def _load_file(self, *, subject: str, json_path: Path) -> List[QuestionRecord]:
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        if not isinstance(payload, dict):
            return []

        records: List[QuestionRecord] = []
        for year_key, sessions in payload.items():
            year = _to_int(year_key)
            if year is None or not isinstance(sessions, dict):
                continue
            for session, papers in sessions.items():
                if not isinstance(papers, dict):
                    continue
                for paper, variants in papers.items():
                    if not isinstance(variants, dict):
                        continue
                    for variant, entries in variants.items():
                        if not isinstance(entries, list):
                            continue
                        records.extend(
                            self._parse_entries(
                                subject=subject,
                                year=year,
                                session=str(session),
                                paper=str(paper),
                                variant=str(variant),
                                entries=entries,
                            )
                        )
        return records

    def _parse_entries(
        self,
        *,
        subject: str,
        year: int,
        session: str,
        paper: str,
        variant: str,
        entries: List[object],
    ) -> List[QuestionRecord]:
        source_ref = "/".join([subject, str(year), session, paper, variant])
        out: List[QuestionRecord] = []
        for row in entries:
            if not isinstance(row, dict):
                continue

            q_num = _norm_qnum(row.get("question_number"))
            sub = _norm_sub(row.get("sub_question"))
            if not q_num:
                continue

            question_text = str(row.get("question_text") or "").strip()
            marking_scheme = str(row.get("marking_scheme") or "").strip()
            if not question_text or not marking_scheme:
                continue

            page_number = _to_int(row.get("page_number"))
            sub_question = sub or None
            suffix = q_num if sub_question is None else f"{q_num}{sub_question}"
            question_id = "|".join(
                [
                    "fallback",
                    subject,
                    str(year),
                    session,
                    paper,
                    variant,
                    suffix,
                ]
            )

            out.append(
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
        return out

