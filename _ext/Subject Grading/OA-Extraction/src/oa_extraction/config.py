from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv_if_present(start_dir: Path | None = None) -> None:
    candidate = (start_dir or Path.cwd()) / ".env"
    if not candidate.exists():
        return

    for raw_line in candidate.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


def _get_env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


def _get_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def _get_env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    api_key: str | None
    base_url: str
    model: str
    timeout_seconds: float
    max_retries: int
    ocr_confidence_threshold: float
    split_confidence_threshold: float
    classification_confidence_threshold: float
    azure_endpoint: str | None
    azure_api_key: str | None
    azure_api_version: str
    enable_azure_fallback: bool
    grok_fallback_ocr_threshold: float
    grok_fallback_split_threshold: float
    enable_image_variants: bool
    enable_targeted_repair: bool
    engine_disagreement_threshold: float
    repair_confidence_threshold: float
    selection_score_margin: float
    azure_poll_interval_seconds: float = 1.0
    image_detail: str = "high"
    max_image_size_bytes: int = 20 * 1024 * 1024

    @classmethod
    def from_env(cls, start_dir: Path | None = None) -> "Settings":
        _load_dotenv_if_present(start_dir)
        return cls(
            api_key=os.getenv("Grok_API") or os.getenv("XAI_API_KEY"),
            base_url=os.getenv("OA_GROK_BASE_URL", "https://api.x.ai/v1").rstrip("/"),
            model=os.getenv("OA_GROK_MODEL", "grok-4.20-reasoning"),
            timeout_seconds=_get_env_float("OA_GROK_OCR_SPLIT_TIMEOUT_SECONDS", 180.0),
            max_retries=_get_env_int("OA_GROK_OCR_SPLIT_MAX_RETRIES", 2),
            ocr_confidence_threshold=_get_env_float("OA_OCR_CONFIDENCE_THRESHOLD", 0.85),
            split_confidence_threshold=_get_env_float("OA_SPLIT_CONFIDENCE_THRESHOLD", 0.90),
            classification_confidence_threshold=_get_env_float(
                "OA_CLASSIFICATION_CONFIDENCE_THRESHOLD", 0.80
            ),
            azure_endpoint=(
                os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT") or os.getenv("AZURE_ENDPOINT")
            ),
            azure_api_key=(
                os.getenv("AZURE_DOCUMENT_INTELLIGENCE_API_KEY") or os.getenv("AZURE_KEY")
            ),
            azure_api_version=os.getenv(
                "OA_AZURE_DOCUMENT_INTELLIGENCE_API_VERSION",
                "2024-11-30",
            ),
            enable_azure_fallback=_get_env_bool("OA_ENABLE_AZURE_FALLBACK", True),
            grok_fallback_ocr_threshold=_get_env_float("OA_GROK_FALLBACK_OCR_THRESHOLD", 0.90),
            grok_fallback_split_threshold=_get_env_float("OA_GROK_FALLBACK_SPLIT_THRESHOLD", 0.92),
            enable_image_variants=_get_env_bool("OA_ENABLE_IMAGE_VARIANTS", True),
            enable_targeted_repair=_get_env_bool("OA_ENABLE_TARGETED_REPAIR", True),
            engine_disagreement_threshold=_get_env_float("OA_ENGINE_DISAGREEMENT_THRESHOLD", 0.08),
            repair_confidence_threshold=_get_env_float("OA_REPAIR_CONFIDENCE_THRESHOLD", 0.85),
            selection_score_margin=_get_env_float("OA_SELECTION_SCORE_MARGIN", 0.05),
        )
