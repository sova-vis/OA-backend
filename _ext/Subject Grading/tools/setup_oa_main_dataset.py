#!/usr/bin/env python3
"""Create canonical OA main dataset folder scaffold for 2015-2025."""

from __future__ import annotations

import argparse
from pathlib import Path

SUBJECTS = [
    "Chemistry 1011",
    "English 1012",
    "Islamiyat 1013",
    "Mathematics 1014",
    "Pakistan Studies 1015",
    "Physics 1016",
]
SESSIONS = ["May_June", "Oct_Nov"]
PAPERS = ["Paper_1", "Paper_2", "Paper_3", "Paper_4"]
VARIANTS = ["Variant_1", "Variant_2", "Variant_3"]


def build_scaffold(root: Path, start_year: int, end_year: int) -> int:
    created = 0
    for subject in SUBJECTS:
        for year in range(start_year, end_year + 1):
            for session in SESSIONS:
                for paper in PAPERS:
                    for variant in VARIANTS:
                        leaf = root / subject / str(year) / session / paper / variant
                        if not leaf.exists():
                            created += 1
                        leaf.mkdir(parents=True, exist_ok=True)
    return created


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create canonical OA main dataset folder scaffold"
    )
    parser.add_argument(
        "--root",
        default="OA_MAIN_DATASET",
        help="Dataset root folder (default: OA_MAIN_DATASET)",
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
    root.mkdir(parents=True, exist_ok=True)

    created = build_scaffold(root, args.start_year, args.end_year)
    print(f"Scaffold ready at: {root.resolve()}")
    print(f"Leaf directories created: {created}")


if __name__ == "__main__":
    main()
