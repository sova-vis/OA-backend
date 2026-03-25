"""Configuration constants for Phase 1 evaluator pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_text(name: str, default: str) -> str:
    return (os.getenv(name) or default).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env_text(name, str(default)))
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env_text(name, str(default)))
    except Exception:
        return float(default)


def _env_bool(name: str, default: bool) -> bool:
    value = _env_text(name, "true" if default else "false").lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return bool(default)


@dataclass(frozen=True)
class PipelineConfig:
    dataset_root: Path = Path("OA_MAIN_DATASET")
    fallback_root: Path = Path("O_LEVEL_JSON")
    main_json_root: Path = field(
        default_factory=lambda: Path(_env_text("OA_MAIN_JSON_ROOT", "O_LEVEL_MAIN_JSON"))
    )
    enable_fallback: bool = True
    source_priority: str = field(
        default_factory=lambda: _env_text("OA_SOURCE_PRIORITY", "o_level_main_first")
    )
    search_method: str = field(
        default_factory=lambda: _env_text("OA_SEARCH_METHOD", "embedding_local")
    )
    search_cache_dir: Path = field(
        default_factory=lambda: Path(_env_text("OA_SEARCH_CACHE_DIR", "SEARCH_INDEX_CACHE"))
    )
    embed_model: str = field(
        default_factory=lambda: _env_text("OA_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    embed_backend: str = field(
        default_factory=lambda: _env_text("OA_EMBED_BACKEND", "sentence_transformers").strip().lower()
    )
    search_top_k: int = field(
        default_factory=lambda: max(1, _env_int("OA_SEARCH_TOP_K", 50))
    )
    search_rerank_k: int = field(
        default_factory=lambda: max(1, _env_int("OA_SEARCH_RERANK_K", 10))
    )
    search_accepted_threshold: float = field(
        default_factory=lambda: _env_float("OA_SEARCH_ACCEPTED_THRESHOLD", 0.78)
    )
    search_review_threshold: float = field(
        default_factory=lambda: _env_float("OA_SEARCH_REVIEW_THRESHOLD", 0.62)
    )
    search_auto_rebuild: bool = field(
        default_factory=lambda: _env_bool("OA_SEARCH_AUTO_REBUILD", True)
    )
    pair_summary_filename: str = "pair_extraction_summary.json"
    qp_extracted_filename: str = "qp_extracted.json"
    ms_extracted_filename: str = "ms_extracted.json"

    token_overlap_weight: float = 0.55
    sequence_similarity_weight: float = 0.30
    number_bonus_weight: float = 0.15

    accepted_threshold: float = 0.82
    review_threshold: float = 0.65

    fully_correct_threshold: float = 0.85
    partially_correct_threshold: float = 0.45

    top_alternatives: int = 4  # best + next 3

    use_grok_grading: bool = True
    grok_api_key: str = field(default_factory=lambda: _env_text("Grok_API", ""))
    grok_model: str = field(default_factory=lambda: _env_text("OA_GROK_GRADING_MODEL", "grok-4-1-fast-reasoning"))
    grok_timeout_seconds: int = 20
    grok_max_retries: int = 1

    debug_runs_dir: Path = field(
        default_factory=lambda: Path(_env_text("OA_DEBUG_RUNS_DIR", "DEBUG_RUNS"))
    )

    debug_store_image: bool = field(
        default_factory=lambda: _env_bool("OA_DEBUG_STORE_IMAGE", False)
    )

    vision_auto_accept_confidence: float = field(
        default_factory=lambda: _env_float("OA_VISION_AUTO_ACCEPT_CONFIDENCE", 0.92)
    )
    vision_auto_accept_requires_no_warnings: bool = field(
        default_factory=lambda: _env_bool("OA_VISION_AUTO_ACCEPT_REQUIRES_NO_WARNINGS", True)
    )
    match_auto_accept_confidence: float = field(
        default_factory=lambda: _env_float("OA_MATCH_AUTO_ACCEPT_CONFIDENCE", 0.85)
    )
    match_auto_accept_margin: float = field(
        default_factory=lambda: _env_float("OA_MATCH_AUTO_ACCEPT_MARGIN", 0.05)
    )
