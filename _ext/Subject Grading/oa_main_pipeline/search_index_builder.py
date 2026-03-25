"""Manual prewarm entrypoint for local search indexes."""

from __future__ import annotations

import argparse
import json

from .config import PipelineConfig
from .dataset_repository import DatasetRepository
from .fallback_repository import FallbackDatasetRepository
from .o_level_main_repository import MainJsonRepository
from .search_index import SearchIndexManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build local search indexes for O/A evaluator")
    parser.add_argument(
        "--source",
        choices=["all", "o_level_json", "oa_main_dataset", "o_level_main_json"],
        default="all",
        help="Index source to build",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Clear in-memory cache before ensuring indexes are built",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = PipelineConfig()
    repository = DatasetRepository(config)
    fallback_repository = FallbackDatasetRepository(config)
    main_repository = MainJsonRepository(config)
    manager = SearchIndexManager(
        repository=repository,
        fallback_repository=fallback_repository,
        main_repository=main_repository,
        config=config,
    )
    if args.force:
        manager.reload()

    sources = (
        ["o_level_json", "oa_main_dataset", "o_level_main_json"]
        if args.source == "all"
        else [args.source]
    )
    payload = {
        source: manager.ensure_built(source)  # type: ignore[arg-type]
        for source in sources
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
