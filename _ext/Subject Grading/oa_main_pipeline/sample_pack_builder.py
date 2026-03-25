"""Build a repeatable sample question pack from O_LEVEL_MAIN_JSON for tests and manual validation."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

from .o_level_main_repository import load_records_from_main_json
from .schemas import QuestionRecord

_DEFAULT_SEED = 1014
_DEFAULT_COUNT = 10
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_INPUT = _REPO_ROOT / "O_LEVEL_MAIN_JSON" / "Mathematics 1014" / "2015-2019.json"
_DEFAULT_OUTPUT = _REPO_ROOT / "tests" / "fixtures" / "oa_sample_questions_math1014_2015_2019.json"


def _is_mcq(record: QuestionRecord) -> bool:
    """Heuristic: single letter A/B/C/D as marking scheme answer."""
    text = (record.marking_scheme_answer or "").strip().upper()
    return text in ("A", "B", "C", "D")


def _record_to_export_item(record: QuestionRecord) -> Dict[str, Any]:
    return {
        "question_id": record.question_id,
        "question_text": record.question_text,
        "page_number": record.page_number,
        "marking_scheme_answer": record.marking_scheme_answer,
        "source_paper_reference": record.source_paper_reference,
    }


def build_sample_pack(
    input_path: Path,
    output_path: Path,
    count: int = _DEFAULT_COUNT,
    seed: int = _DEFAULT_SEED,
) -> List[Dict[str, Any]]:
    """
    Load questions from input_path, sample `count` with stratified mix (MCQ + non-MCQ), write JSON.
    Returns the list of exported items for testing.
    """
    records = load_records_from_main_json(input_path)
    if not records:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("[]", encoding="utf-8")
        return []

    mcq = [r for r in records if _is_mcq(r)]
    non_mcq = [r for r in records if not _is_mcq(r)]
    rng = random.Random(seed)

    half = count // 2
    take_mcq = min(half, len(mcq))
    take_non_mcq = min(count - take_mcq, len(non_mcq))
    if take_mcq + take_non_mcq < count:
        take_mcq = min(count - take_non_mcq, len(mcq))
        take_non_mcq = min(count - take_mcq, len(non_mcq))

    chosen_mcq = list(rng.sample(mcq, take_mcq)) if mcq else []
    chosen_non_mcq = list(rng.sample(non_mcq, take_non_mcq)) if non_mcq else []
    combined = chosen_mcq + chosen_non_mcq
    rng.shuffle(combined)

    while len(combined) < count and (mcq or non_mcq):
        pool = [r for r in records if r not in combined]
        if not pool:
            break
        combined.append(rng.choice(pool))

    exported = [_record_to_export_item(r) for r in combined[:count]]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(exported, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return exported


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a repeatable 10-question sample pack from O_LEVEL_MAIN_JSON"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=_DEFAULT_INPUT,
        help="Path to course JSON file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_DEFAULT_SEED,
        help="Random seed for reproducible sampling",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=_DEFAULT_COUNT,
        help="Number of questions to sample",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help="Output JSON path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_sample_pack(
        input_path=args.input,
        output_path=args.output,
        count=args.count,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
