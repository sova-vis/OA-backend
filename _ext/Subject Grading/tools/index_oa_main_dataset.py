#!/usr/bin/env python3
"""Index OA main dataset and keep only valid qp/ms pairs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

SUBJECTS = {
    "Chemistry 1011",
    "English 1012",
    "Islamiyat 1013",
    "Mathematics 1014",
    "Pakistan Studies 1015",
    "Physics 1016",
}
SESSIONS = {"May_June", "Oct_Nov"}
PAPERS = {"Paper_1", "Paper_2", "Paper_3", "Paper_4"}
VARIANTS = {"Variant_1", "Variant_2", "Variant_3"}


def parse_variant_from_path(path: Path, root: Path) -> Dict[str, str]:
    rel = path.relative_to(root)
    parts = rel.parts
    if len(parts) != 6:
        raise ValueError("Unexpected path depth")

    subject, year, session, paper, variant, _ = parts
    if subject not in SUBJECTS:
        raise ValueError("Unknown subject")
    if session not in SESSIONS:
        raise ValueError("Unknown session")
    if paper not in PAPERS:
        raise ValueError("Unknown paper")
    if variant not in VARIANTS:
        raise ValueError("Unknown variant")
    if not year.isdigit():
        raise ValueError("Invalid year")

    return {
        "subject": subject,
        "year": int(year),
        "session": session,
        "paper": paper,
        "variant": variant,
    }


def build_index(root: Path, start_year: int, end_year: int) -> Dict[str, object]:
    valid_pairs: List[Dict[str, object]] = []
    skipped_missing_ms: List[Dict[str, object]] = []
    skipped_missing_qp: List[Dict[str, object]] = []
    malformed_paths: List[str] = []

    # Walk candidate variant directories by locating qp.pdf or ms.pdf files.
    seen_variants = set()
    for file_path in root.rglob("*.pdf"):
        if file_path.name not in {"qp.pdf", "ms.pdf"}:
            continue

        variant_dir = file_path.parent
        if variant_dir in seen_variants:
            continue
        seen_variants.add(variant_dir)

        qp = variant_dir / "qp.pdf"
        ms = variant_dir / "ms.pdf"

        try:
            meta = parse_variant_from_path(qp if qp.exists() else ms, root)
        except Exception:
            malformed_paths.append(str(variant_dir))
            continue

        year = int(meta["year"])
        if year < start_year or year > end_year:
            continue

        item = {
            **meta,
            "qp_path": str(qp),
            "ms_path": str(ms),
        }

        if qp.exists() and ms.exists():
            valid_pairs.append(item)
        elif qp.exists() and not ms.exists():
            skipped_missing_ms.append(item)
        elif ms.exists() and not qp.exists():
            skipped_missing_qp.append(item)

    valid_pairs.sort(
        key=lambda x: (
            x["subject"],
            int(x["year"]),
            x["session"],
            x["paper"],
            x["variant"],
        )
    )
    skipped_missing_ms.sort(
        key=lambda x: (
            x["subject"],
            int(x["year"]),
            x["session"],
            x["paper"],
            x["variant"],
        )
    )
    skipped_missing_qp.sort(
        key=lambda x: (
            x["subject"],
            int(x["year"]),
            x["session"],
            x["paper"],
            x["variant"],
        )
    )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(root),
        "year_range": {"start": start_year, "end": end_year},
        "summary": {
            "valid_pairs": len(valid_pairs),
            "skipped_missing_ms": len(skipped_missing_ms),
            "skipped_missing_qp": len(skipped_missing_qp),
            "malformed_paths": len(malformed_paths),
        },
        "valid_pairs": valid_pairs,
        "skipped_missing_ms": skipped_missing_ms,
        "skipped_missing_qp": skipped_missing_qp,
        "malformed_paths": malformed_paths,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index OA main dataset with strict qp/ms pairing"
    )
    parser.add_argument(
        "--root",
        default="OA_MAIN_DATASET",
        help="Dataset root folder (default: OA_MAIN_DATASET)",
    )
    parser.add_argument(
        "--output",
        default="OA_MAIN_DATASET/index.json",
        help="Path to write index JSON (default: OA_MAIN_DATASET/index.json)",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2015,
        help="Start year (default: 2015)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2025,
        help="End year (default: 2025)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    output = Path(args.output)

    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    index_data = build_index(root, args.start_year, args.end_year)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(index_data, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = index_data["summary"]
    print(f"Index written to: {output.resolve()}")
    print(
        "Summary -> "
        f"valid_pairs={summary['valid_pairs']}, "
        f"skipped_missing_ms={summary['skipped_missing_ms']}, "
        f"skipped_missing_qp={summary['skipped_missing_qp']}, "
        f"malformed_paths={summary['malformed_paths']}"
    )


if __name__ == "__main__":
    main()
