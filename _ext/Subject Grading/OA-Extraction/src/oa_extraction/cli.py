from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import extract_qa
from .types import OAExtractionError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract one handwritten question-answer pair from an image or PDF.")
    parser.add_argument("path", help="Path to a PNG, JPG, JPEG, or PDF file.")
    parser.add_argument("--json-out", help="Optional output file for the result JSON.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        result = extract_qa(args.path)
    except OAExtractionError as exc:
        payload = {
            "error": {
                "code": exc.code,
                "message": str(exc),
                "details": exc.details,
            }
        }
        print(json.dumps(payload, indent=2), file=sys.stderr)
        return 1

    dumped = result.model_dump(mode="json")
    json_text = json.dumps(dumped, indent=2 if args.pretty else None)

    if args.json_out:
        output_path = Path(args.json_out)
        output_path.write_text(json_text + ("\n" if args.pretty else ""), encoding="utf-8")

    print(json_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
