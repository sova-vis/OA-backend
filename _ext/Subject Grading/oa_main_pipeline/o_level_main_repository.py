"""Loader for O_LEVEL_MAIN_JSON course files (primary dataset)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .config import PipelineConfig
from .dataset_repository import _norm_qnum, _norm_sub, filter_question_records
from .schemas import EvaluateRequest, QuestionRecord


def load_records_from_main_json(
    json_path: Path,
    subject: Optional[str] = None,
) -> List[QuestionRecord]:
    """
    Load question records from a single O_LEVEL_MAIN_JSON course file.

    File structure: year -> session -> paper -> variant -> list of entries.
    Each entry has question_number, sub_question, question_text, marking_scheme, etc.
    If subject is not provided, it is derived from the parent directory name.
    """
    if not json_path.exists():
        return []

    resolved_subject = subject
    if resolved_subject is None:
        resolved_subject = json_path.parent.name.strip() or "Unknown"

    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    if not isinstance(payload, dict):
        return []

    records: List[QuestionRecord] = []
    for year_key, sessions in payload.items():
        try:
            year = int(year_key)
        except (TypeError, ValueError):
            continue
        if not isinstance(sessions, dict):
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
                    source_ref = "/".join(
                        [resolved_subject, str(year), str(session), str(paper), str(variant)]
                    )
                    for row in entries:
                        if not isinstance(row, dict):
                            continue
                        q_num = _norm_qnum(row.get("question_number"))
                        if not q_num:
                            continue
                        sub = _norm_sub(row.get("sub_question"))
                        sub_question = sub or None
                        question_text = str(row.get("question_text") or "").strip()
                        marking_scheme = str(row.get("marking_scheme") or "").strip()
                        if not question_text or not marking_scheme:
                            continue
                        suffix = q_num if sub_question is None else f"{q_num}{sub_question}"
                        question_id = "|".join(
                            [
                                "main",
                                resolved_subject,
                                str(year),
                                str(session),
                                str(paper),
                                str(variant),
                                suffix,
                            ]
                        )
                        records.append(
                            QuestionRecord(
                                question_id=question_id,
                                subject=resolved_subject,
                                year=year,
                                session=str(session),
                                paper=str(paper),
                                variant=str(variant),
                                question_number=q_num,
                                sub_question=sub_question,
                                question_text=question_text,
                                marking_scheme_answer=marking_scheme,
                                page_number=None,
                                source_paper_reference=source_ref,
                            )
                        )
    return records


class MainJsonRepository:
    """Loads and caches QuestionRecord items from all O_LEVEL_MAIN_JSON course files."""

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
        root = self.config.main_json_root
        if not root.exists():
            return []
        seen: Dict[str, QuestionRecord] = {}
        for subject_dir in sorted(root.iterdir()):
            if not subject_dir.is_dir():
                continue
            for json_path in sorted(subject_dir.glob("*.json")):
                for record in load_records_from_main_json(json_path):
                    seen[record.question_id] = record
        return list(seen.values())
