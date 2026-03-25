#!/usr/bin/env python3
"""Migrate QP/MS PDFs from O_LEVEL_PAST_PAPERS to OA_MAIN_DATASET.

Rules:
- Source: <subject>/Past_Papers/<year>/<session>/<paper>/<variant>/<file>
- Accept only *_QP.pdf and *_MS.pdf
- Year filter defaults to 2015-2025
- Destination: <dest>/<subject>/<year>/<session>/<paper>/<variant>/{qp,ms}.pdf
- Skip malformed paths and inconsistent filename metadata (strict mode)
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SESSION_SET = {"May_June", "Oct_Nov"}
PAPER_SET = {"Paper_1", "Paper_2", "Paper_3", "Paper_4"}
VARIANT_SET = {"Variant_1", "Variant_2", "Variant_3"}

FILENAME_RE = re.compile(
    r"^(?P<subject>.+?)_"
    r"(?P<year>\d{4})_"
    r"(?P<session>May_June|Oct_Nov)_"
    r"(?P<paper>Paper_[1-4])"
    r"(?:_(?P<variant>Variant_[1-3]))?_"
    r"(?P<kind>QP|MS)\.pdf$",
    re.IGNORECASE,
)


@dataclass
class Candidate:
    source: Path
    subject: str
    year: int
    session: str
    paper: str
    variant: str
    kind: str  # qp | ms


@dataclass
class Summary:
    scanned_files: int = 0
    candidates: int = 0
    moved_or_copied: int = 0
    skipped_year_out_of_range: int = 0
    skipped_non_qp_ms: int = 0
    skipped_malformed_path: int = 0
    skipped_invalid_meta: int = 0
    skipped_filename_mismatch: int = 0
    skipped_existing_dest: int = 0
    failed_ops: int = 0


class Migrator:
    def __init__(
        self,
        source_root: Path,
        dest_root: Path,
        start_year: int,
        end_year: int,
        action: str,
        execute: bool,
        overwrite: bool,
        strict_filename_check: bool,
    ) -> None:
        self.source_root = source_root
        self.dest_root = dest_root
        self.start_year = start_year
        self.end_year = end_year
        self.action = action
        self.execute = execute
        self.overwrite = overwrite
        self.strict_filename_check = strict_filename_check
        self.summary = Summary()
        self.samples: Dict[str, List[str]] = {
            "malformed_path": [],
            "invalid_meta": [],
            "filename_mismatch": [],
            "existing_dest": [],
            "failed_ops": [],
        }

    def _add_sample(self, key: str, message: str, limit: int = 20) -> None:
        if len(self.samples[key]) < limit:
            self.samples[key].append(message)

    def _extract_variant_from_filename(self, filename: str) -> Optional[str]:
        match = FILENAME_RE.match(filename)
        if not match:
            return None
        variant = match.group("variant")
        if variant:
            variant = f"Variant_{variant.split('_')[-1]}"
        return variant

    def _parse_candidate(self, file_path: Path) -> Optional[Candidate]:
        self.summary.scanned_files += 1

        if file_path.suffix.lower() != ".pdf":
            return None

        name_upper = file_path.name.upper()
        if not (name_upper.endswith("_QP.PDF") or name_upper.endswith("_MS.PDF")):
            self.summary.skipped_non_qp_ms += 1
            return None

        rel = file_path.relative_to(self.source_root)
        parts = rel.parts

        # Expected patterns:
        # 7 parts: subject/Past_Papers/year/session/paper/variant/file
        # 6 parts: subject/Past_Papers/year/session/paper/file (legacy; infer variant)
        if len(parts) not in (6, 7):
            self.summary.skipped_malformed_path += 1
            self._add_sample("malformed_path", str(rel))
            return None

        subject = parts[0]
        if parts[1] != "Past_Papers":
            self.summary.skipped_malformed_path += 1
            self._add_sample("malformed_path", str(rel))
            return None

        year_str, session, paper = parts[2], parts[3], parts[4]
        filename = parts[-1]

        if not year_str.isdigit():
            self.summary.skipped_invalid_meta += 1
            self._add_sample("invalid_meta", f"invalid year: {rel}")
            return None

        year = int(year_str)
        if year < self.start_year or year > self.end_year:
            self.summary.skipped_year_out_of_range += 1
            return None

        if session not in SESSION_SET or paper not in PAPER_SET:
            self.summary.skipped_invalid_meta += 1
            self._add_sample("invalid_meta", f"invalid session/paper: {rel}")
            return None

        if len(parts) == 7:
            variant = parts[5]
        else:
            # Infer variant for legacy layouts lacking variant folder.
            inferred = self._extract_variant_from_filename(filename)
            variant = inferred or "Variant_1"

        if variant not in VARIANT_SET:
            self.summary.skipped_invalid_meta += 1
            self._add_sample("invalid_meta", f"invalid variant: {rel}")
            return None

        kind = "qp" if name_upper.endswith("_QP.PDF") else "ms"

        if self.strict_filename_check:
            match = FILENAME_RE.match(filename)
            if not match:
                self.summary.skipped_filename_mismatch += 1
                self._add_sample("filename_mismatch", f"unparseable filename: {rel}")
                return None

            f_year = int(match.group("year"))
            f_session = match.group("session")
            f_paper = match.group("paper")
            f_variant = match.group("variant") or "Variant_1"
            f_variant = f"Variant_{f_variant.split('_')[-1]}"
            f_kind = match.group("kind").lower()

            if (
                f_year != year
                or f_session != session
                or f_paper != paper
                or f_variant != variant
                or f_kind != kind
            ):
                self.summary.skipped_filename_mismatch += 1
                self._add_sample(
                    "filename_mismatch",
                    f"{rel} -> path({year},{session},{paper},{variant},{kind}) "
                    f"vs file({f_year},{f_session},{f_paper},{f_variant},{f_kind})",
                )
                return None

        self.summary.candidates += 1
        return Candidate(
            source=file_path,
            subject=subject,
            year=year,
            session=session,
            paper=paper,
            variant=variant,
            kind=kind,
        )

    def _target_path(self, candidate: Candidate) -> Path:
        return (
            self.dest_root
            / candidate.subject
            / str(candidate.year)
            / candidate.session
            / candidate.paper
            / candidate.variant
            / f"{candidate.kind}.pdf"
        )

    def _apply(self, src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)

        if dst.exists() and not self.overwrite:
            self.summary.skipped_existing_dest += 1
            self._add_sample("existing_dest", str(dst))
            return

        try:
            if not self.execute:
                self.summary.moved_or_copied += 1
                return

            if dst.exists() and self.overwrite:
                dst.unlink()

            if self.action == "move":
                shutil.move(str(src), str(dst))
            else:
                shutil.copy2(str(src), str(dst))
            self.summary.moved_or_copied += 1
        except Exception as exc:
            self.summary.failed_ops += 1
            self._add_sample("failed_ops", f"{src} -> {dst}: {exc}")

    def run(self) -> Dict[str, object]:
        for file_path in self.source_root.rglob("*.pdf"):
            candidate = self._parse_candidate(file_path)
            if not candidate:
                continue
            self._apply(candidate.source, self._target_path(candidate))

        return {
            "mode": self.action,
            "execute": self.execute,
            "source_root": str(self.source_root),
            "dest_root": str(self.dest_root),
            "year_range": {"start": self.start_year, "end": self.end_year},
            "strict_filename_check": self.strict_filename_check,
            "overwrite": self.overwrite,
            "summary": self.summary.__dict__,
            "samples": self.samples,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate QP/MS from O_LEVEL_PAST_PAPERS to OA_MAIN_DATASET"
    )
    parser.add_argument("--source-root", default="O_LEVEL_PAST_PAPERS")
    parser.add_argument("--dest-root", default="OA_MAIN_DATASET")
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument(
        "--action",
        choices=["move", "copy"],
        default="move",
        help="File operation to apply (default: move)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform operation. Without this flag, dry-run only.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination qp/ms if already present.",
    )
    parser.add_argument(
        "--no-strict-filename-check",
        action="store_true",
        help="Disable filename-vs-path consistency validation.",
    )
    parser.add_argument(
        "--report-path",
        default="OA_MAIN_DATASET/migration_report.json",
        help="Write JSON report here",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    migrator = Migrator(
        source_root=Path(args.source_root),
        dest_root=Path(args.dest_root),
        start_year=args.start_year,
        end_year=args.end_year,
        action=args.action,
        execute=args.execute,
        overwrite=args.overwrite,
        strict_filename_check=not args.no_strict_filename_check,
    )
    result = migrator.run()

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    s = result["summary"]
    mode = "EXECUTE" if args.execute else "DRY-RUN"
    print(f"[{mode}] action={args.action} source={args.source_root} dest={args.dest_root}")
    print(
        "Summary -> "
        f"scanned={s['scanned_files']}, "
        f"candidates={s['candidates']}, "
        f"applied={s['moved_or_copied']}, "
        f"out_of_range={s['skipped_year_out_of_range']}, "
        f"non_qp_ms={s['skipped_non_qp_ms']}, "
        f"malformed={s['skipped_malformed_path']}, "
        f"invalid_meta={s['skipped_invalid_meta']}, "
        f"filename_mismatch={s['skipped_filename_mismatch']}, "
        f"existing_dest={s['skipped_existing_dest']}, "
        f"failed={s['failed_ops']}"
    )
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
