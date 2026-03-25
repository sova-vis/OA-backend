#!/usr/bin/env python3
"""OCR ingestion pipeline for OA_MAIN_DATASET.

Processes QP/MS pairs from OA_MAIN_DATASET/index.json and writes extraction
artifacts into each variant folder:
- qp_extracted.json
- ms_extracted.json
- pair_extraction_summary.json

Also writes central operational reports:
- OA_MAIN_DATASET/ocr_run_report.json
- OA_MAIN_DATASET/ocr_review_queue.json

Optional debug artifacts are written outside OA_MAIN_DATASET:
- OCR_DEBUG_OUTPUT/<run_timestamp>/<subject>/<year>/<session>/<paper>/<variant>/*.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import traceback
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import requests

try:  # pragma: no cover - optional import for local tests
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional import for local tests

    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:
        return False


try:  # pragma: no cover - optional import for local tests
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover - optional import for local tests
    fitz = None  # type: ignore

try:  # pragma: no cover - optional import for local tests
    from azure.ai.formrecognizer import DocumentAnalysisClient
    from azure.core.credentials import AzureKeyCredential
except ImportError:  # pragma: no cover - optional import for local tests
    DocumentAnalysisClient = Any  # type: ignore
    AzureKeyCredential = None  # type: ignore


GROK_CHAT_URL = "https://api.x.ai/v1/chat/completions"
DEFAULT_GROK_MODEL = "grok-4-1-fast-reasoning"
DEFAULT_DEBUG_DIR = "OCR_DEBUG_OUTPUT"

QUESTION_WITH_SUB_RE = re.compile(
    r"^\s*(?P<q>\d{1,3})\s*\((?P<sub>[A-Za-zivxIVX]{1,6})\)\s*(?P<rest>.*)$"
)
QUESTION_ONLY_RE = re.compile(r"^\s*(?P<q>\d{1,3})\s*[\).:-]?\s*(?P<rest>.*)$")
TOP_LEVEL_SECTION_Q_RE = re.compile(
    r"^\s*(?P<section>[A-Da-d])\s*(?P<q>\d{1,3})\s*[\).:-]?\s*(?P<rest>.*)$"
)
SUB_ONLY_RE = re.compile(r"^\s*\((?P<sub>[A-Za-zivxIVX]{1,6})\)\s*(?P<rest>.*)$")
MARK_RE = re.compile(r"\[(\d{1,3})\]|(\d{1,3})\s*marks?", re.IGNORECASE)
MCQ_OPTION_PATTERN = re.compile(r"\bA\b.*\bB\b.*\bC\b.*\bD\b", re.IGNORECASE)

MCQ_PAGE_EXCLUDE_HINTS = (
    "read these instructions first",
    "periodic table of elements",
)
MCQ_LINE_NOISE_PATTERNS = [
    re.compile(r"\bturn\s+over\b", re.IGNORECASE),
    re.compile(r"\bcambridge\s+international\s+examinations\b", re.IGNORECASE),
    re.compile(r"\bcambridge\s+international\b", re.IGNORECASE),
    re.compile(r"\bcuc?les\b", re.IGNORECASE),
    re.compile(r"\bpage\s+\d+\s+of\s+\d+\b", re.IGNORECASE),
    re.compile(r"\b(?:syllabus|mark scheme|published|copyright acknowledgements)\b", re.IGNORECASE),
    re.compile(r"\b\d{4}/\d{2}\b"),
    re.compile(r"\b\d{1,2}/\d{2}/[A-Z]/[A-Z]/\d{2}\b", re.IGNORECASE),
]
MS_NOISE_PATTERNS = [
    re.compile(r"\bcambridge\b", re.IGNORECASE),
    re.compile(r"\bsyllabus\b", re.IGNORECASE),
    re.compile(r"\bpaper\b", re.IGNORECASE),
    re.compile(r"\bmark\s+scheme\b", re.IGNORECASE),
    re.compile(r"\bpublished\b", re.IGNORECASE),
    re.compile(r"\bturn\s+over\b", re.IGNORECASE),
    re.compile(r"\bpage\s+\d+\s+of\s+\d+\b", re.IGNORECASE),
]
STRUCTURED_PAGE_EXCLUDE_HINTS = (
    "read these instructions first",
    "periodic table of the elements",
    "data sheet",
    "blank page",
)
STRUCTURED_LINE_NOISE_PATTERNS = [
    re.compile(r"\bturn\s+over\b", re.IGNORECASE),
    re.compile(r"\bcambridge\b", re.IGNORECASE),
    re.compile(r"\bcuc?les\b", re.IGNORECASE),
    re.compile(r"\bpage\s+\d+\s+of\s+\d+\b", re.IGNORECASE),
    re.compile(r"\b(?:syllabus|published|candidate number|centre number)\b", re.IGNORECASE),
    re.compile(r"\b\d{4}/\d{2}\b"),
    re.compile(r"\b\d{1,2}/\d{2}/[A-Z]/[A-Z]/\d{2}\b", re.IGNORECASE),
]
STRUCTURED_MS_LINE_NOISE_PATTERNS = [
    re.compile(r"\bquestion\s+answer\s+marks\b", re.IGNORECASE),
    re.compile(r"\bgeneral points\b", re.IGNORECASE),
    re.compile(r"\bquestion conclusions\b", re.IGNORECASE),
    re.compile(r"\boctober/november\b", re.IGNORECASE),
    re.compile(r"\bmay/june\b", re.IGNORECASE),
    re.compile(r"\bpage\s+\d+\s+of\s+\d+\b", re.IGNORECASE),
]


@dataclass(frozen=True)
class PairMeta:
    subject: str
    year: int
    session: str
    paper: str
    variant: str
    qp_path: Path
    ms_path: Path

    @property
    def pair_id(self) -> str:
        return "|".join(
            [self.subject, str(self.year), self.session, self.paper, self.variant]
        )

    @property
    def variant_dir(self) -> Path:
        return self.qp_path.parent


@dataclass
class PipelineSettings:
    grok_api_key: Optional[str]
    grok_model: str
    grok_max_retries: int
    grok_timeout_normalize: int
    grok_timeout_repair: int
    azure_endpoint: str
    azure_key: str
    ocr_per_page_timeout: float
    ocr_overall_timeout: float
    ocr_max_retries: int
    ocr_retry_base_delay: float
    ocr_retry_max_delay: float
    ocr_concurrent_pages: int
    review_conf_threshold: float
    review_match_threshold: float
    use_grok_normalization: bool
    parser_profile: str
    expected_max_question: int
    debug_enabled: bool
    debug_dir: Path
    debug_level: str
    debug_run_id: str
    progress_log_enabled: bool
    progress_log_path: Path


class _StepTimer:
    def __init__(
        self,
        logger: "RunProgressLogger",
        step: str,
        pair_id: Optional[str],
        message: Optional[str],
    ) -> None:
        self.logger = logger
        self.step = step
        self.pair_id = pair_id
        self.message = message or ""
        self._start = 0.0

    def __enter__(self) -> "_StepTimer":
        self._start = time.perf_counter()
        self.logger.log("STEP_START", self.pair_id, self.step, self.message or "start")
        return self

    def __exit__(self, exc_type: Any, exc: Any, _tb: Any) -> bool:
        elapsed = max(0.0, time.perf_counter() - self._start)
        self.logger._record_step_duration(self.step, elapsed)
        if exc is None:
            self.logger.log(
                "STEP_END",
                self.pair_id,
                self.step,
                self.message or "done",
                duration=elapsed,
            )
            return False
        self.logger.error(
            self.pair_id,
            self.step,
            f"{self.message or 'failed'}: {exc}",
            duration=elapsed,
        )
        return False


class RunProgressLogger:
    def __init__(
        self,
        *,
        enabled: bool,
        log_path: Path,
        run_id: str,
    ) -> None:
        self.enabled = bool(enabled)
        self.log_path = Path(log_path)
        self.run_id = run_id
        self._start = time.perf_counter()
        self._fh: Optional[Any] = None
        self.step_stats: Dict[str, Dict[str, float]] = {}
        self.warning_count = 0
        self.error_count = 0
        self.warning_messages: List[str] = []
        self.error_messages: List[str] = []
        self.pair_status: Counter[str] = Counter()
        self.pair_timings: Dict[str, float] = {}
        if self.enabled:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.log_path.open("w", encoding="utf-8")

    @property
    def active(self) -> bool:
        return self.enabled and self._fh is not None

    def close(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None

    def _write(self, line: str) -> None:
        if not self.active:
            return
        assert self._fh is not None
        self._fh.write(line + "\n")
        self._fh.flush()

    def _redact(self, text: str) -> str:
        value = str(text or "")
        value = re.sub(r"(xai-[A-Za-z0-9_-]{8,})", "xai-***REDACTED***", value)
        value = re.sub(r"(api[_-]?key\s*[:=]\s*)([^,\s]+)", r"\1***REDACTED***", value, flags=re.IGNORECASE)
        return value

    def _fmt(self, level: str, pair_id: Optional[str], step: str, message: str) -> str:
        pair_part = pair_id or "-"
        safe = self._redact(message)
        return f"[{_utc_now()}] [{level}] [{pair_part}] [{step}] {safe}"

    def log(
        self,
        level: str,
        pair_id: Optional[str],
        step: str,
        message: str,
        *,
        duration: Optional[float] = None,
    ) -> None:
        msg = message
        if duration is not None:
            msg = f"{message} (duration={duration:.3f}s)"
        self._write(self._fmt(level, pair_id, step, msg))

    def info(self, pair_id: Optional[str], step: str, message: str) -> None:
        self.log("INFO", pair_id, step, message)

    def warn(
        self,
        pair_id: Optional[str],
        step: str,
        message: str,
        *,
        duration: Optional[float] = None,
    ) -> None:
        self.warning_count += 1
        self.warning_messages.append(f"{step}:{self._redact(message)}")
        self.log("STEP_WARN", pair_id, step, message, duration=duration)

    def error(
        self,
        pair_id: Optional[str],
        step: str,
        message: str,
        *,
        duration: Optional[float] = None,
    ) -> None:
        self.error_count += 1
        self.error_messages.append(f"{step}:{self._redact(message)}")
        self.log("STEP_ERROR", pair_id, step, message, duration=duration)

    def step(self, step: str, pair_id: Optional[str] = None, message: Optional[str] = None) -> _StepTimer:
        return _StepTimer(self, step, pair_id, message)

    def _record_step_duration(self, step: str, duration: float) -> None:
        stats = self.step_stats.setdefault(step, {"count": 0.0, "total": 0.0, "max": 0.0})
        stats["count"] += 1.0
        stats["total"] += float(duration)
        stats["max"] = max(stats["max"], float(duration))

    def record_pair_status(self, status: str) -> None:
        self.pair_status[str(status or "unknown")] += 1

    def record_pair_timing(self, pair_id: str, duration: float) -> None:
        self.pair_timings[pair_id] = float(duration)

    def write_header(self, payload: Dict[str, Any]) -> None:
        if not self.active:
            return
        self._write("=== OCR PIPELINE PROGRESS START ===")
        self._write(f"run_id={self.run_id}")
        self._write(f"started_at={_utc_now()}")
        self._write(f"config={self._redact(json.dumps(payload, ensure_ascii=False))}")
        self._write("-----------------------------------")

    def write_final_report(self, run_summary: Dict[str, Any]) -> None:
        if not self.active:
            return
        total_elapsed = max(0.0, time.perf_counter() - self._start)
        self._write("")
        self._write("=== OCR PIPELINE FINAL REPORT ===")
        self._write(f"finished_at={_utc_now()}")
        self._write(f"total_elapsed_seconds={total_elapsed:.3f}")
        self._write(f"summary={self._redact(json.dumps(run_summary, ensure_ascii=False))}")
        self._write(f"warnings={self.warning_count}")
        self._write(f"errors={self.error_count}")

        if self.pair_status:
            self._write(f"pair_status_counts={dict(self.pair_status)}")
        if self.pair_timings:
            top_pairs = sorted(self.pair_timings.items(), key=lambda x: x[1], reverse=True)[:10]
            self._write(f"slowest_pairs={top_pairs}")

        if self.step_stats:
            self._write("step_timings:")
            ordered = sorted(self.step_stats.items(), key=lambda x: x[1]["total"], reverse=True)
            for step, stats in ordered:
                count = max(1.0, stats["count"])
                avg = stats["total"] / count
                self._write(
                    f"  - {step}: count={int(stats['count'])} total={stats['total']:.3f}s "
                    f"avg={avg:.3f}s max={stats['max']:.3f}s"
                )

        if self.warning_messages:
            self._write("warning_samples:")
            for entry in self.warning_messages[:30]:
                self._write(f"  - {entry}")
        if self.error_messages:
            self._write("error_samples:")
            for entry in self.error_messages[:30]:
                self._write(f"  - {entry}")
        self._write("=== OCR PIPELINE PROGRESS END ===")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_slug_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _truncate_text(text: str, limit: int) -> str:
    value = text or ""
    if len(value) <= limit:
        return value
    return f"{value[:limit]} ... [truncated {len(value) - limit} chars]"


def _clean_json_from_llm(text: str) -> str:
    data = (text or "").strip()
    if data.startswith("```"):
        data = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", data)
        data = re.sub(r"\n?```$", "", data)
    if "{" in data and "}" in data:
        start = data.find("{")
        end = data.rfind("}")
        if end > start:
            data = data[start : end + 1]
    return data.strip()


def _trim_to_balanced_object(text: str) -> str:
    data = (text or "").strip()
    if not data:
        return data
    start = data.find("{")
    end = data.rfind("}")
    if start >= 0 and end > start:
        return data[start : end + 1].strip()
    return data


def _remove_trailing_json_commas(text: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", text or "")


def _append_missing_json_closers(text: str) -> str:
    data = text or ""
    stack: List[str] = []
    in_string = False
    escape = False
    for ch in data:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            stack.append("}")
            continue
        if ch == "[":
            stack.append("]")
            continue
        if ch in {"}", "]"} and stack and stack[-1] == ch:
            stack.pop()
    if not stack:
        return data
    return f"{data}{''.join(reversed(stack))}"


def _attempt_json_load_with_repair(
    cleaned: str,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    attempts: List[Dict[str, Any]] = []
    last_error: Optional[str] = None

    def _try(strategy: str, candidate: str) -> Tuple[Optional[Dict[str, Any]], Optional[Exception]]:
        nonlocal last_error
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                attempts.append({"strategy": strategy, "success": True})
                return parsed, None
            attempts.append({"strategy": strategy, "success": False, "error": "top_level_not_object"})
            last_error = "top_level_not_object"
            return None, ValueError("top_level_not_object")
        except json.JSONDecodeError as exc:
            attempts.append(
                {
                    "strategy": strategy,
                    "success": False,
                    "error": str(exc),
                    "error_class": "JSONDecodeError",
                    "json_error": {
                        "line": exc.lineno,
                        "col": exc.colno,
                        "pos": exc.pos,
                        "msg": exc.msg,
                    },
                }
            )
            last_error = str(exc)
            return None, exc
        except Exception as exc:  # noqa: BLE001
            attempts.append(
                {
                    "strategy": strategy,
                    "success": False,
                    "error": str(exc),
                    "error_class": exc.__class__.__name__,
                }
            )
            last_error = str(exc)
            return None, exc

    parsed, err = _try("direct", cleaned)
    if parsed is not None:
        return parsed, {"strategy": "direct", "attempts": attempts}

    # Some models append non-JSON tails; try strict decoder first.
    decoder = json.JSONDecoder()
    try:
        raw_obj, end = decoder.raw_decode(cleaned)
        trailing = cleaned[end:].strip()
        if isinstance(raw_obj, dict) and not trailing:
            attempts.append({"strategy": "raw_decode", "success": True})
            return raw_obj, {"strategy": "raw_decode", "attempts": attempts}
        attempts.append(
            {
                "strategy": "raw_decode",
                "success": False,
                "error": "trailing_content" if trailing else "top_level_not_object",
            }
        )
    except Exception as exc:  # noqa: BLE001
        attempts.append(
            {
                "strategy": "raw_decode",
                "success": False,
                "error": str(exc),
                "error_class": exc.__class__.__name__,
            }
        )

    candidates: List[Tuple[str, str]] = []
    trimmed = _trim_to_balanced_object(cleaned)
    if trimmed and trimmed != cleaned:
        candidates.append(("trim_object", trimmed))
    no_commas = _remove_trailing_json_commas(trimmed or cleaned)
    if no_commas and no_commas not in {cleaned, trimmed}:
        candidates.append(("remove_trailing_commas", no_commas))
    closed = _append_missing_json_closers(no_commas or trimmed or cleaned)
    if closed and closed not in {cleaned, trimmed, no_commas}:
        candidates.append(("append_missing_closers", closed))
    no_commas_closed = _append_missing_json_closers(_remove_trailing_json_commas(trimmed or cleaned))
    if no_commas_closed and no_commas_closed not in {cleaned, trimmed, no_commas, closed}:
        candidates.append(("remove_commas_and_close", no_commas_closed))

    for strategy, candidate in candidates:
        parsed, err = _try(strategy, candidate)
        if parsed is not None:
            return parsed, {"strategy": strategy, "attempts": attempts}

    json_error: Dict[str, Any] = {}
    for entry in reversed(attempts):
        if entry.get("error_class") == "JSONDecodeError":
            json_error = dict(entry.get("json_error") or {})
            break
    return None, {"strategy": "failed", "attempts": attempts, "json_error": json_error, "error": last_error}


def _load_env_file_fallback(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def _normalize_pair_path(path_text: str) -> Path:
    normalized = str(path_text or "").replace("\\", "/")
    return Path(normalized)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OCR extraction pipeline for OA_MAIN_DATASET"
    )
    parser.add_argument("--index-path", default="OA_MAIN_DATASET/index.json")
    parser.add_argument("--run-report-path", default="OA_MAIN_DATASET/ocr_run_report.json")
    parser.add_argument("--review-queue-path", default="OA_MAIN_DATASET/ocr_review_queue.json")

    parser.add_argument("--subject", action="append", default=[])
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--session", default=None)
    parser.add_argument("--paper", default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--resume", dest="resume", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)

    parser.add_argument("--force", action="store_true", help="Reprocess accepted pairs too")
    parser.add_argument(
        "--no-grok-normalization",
        action="store_true",
        help="Disable Grok fallback normalization for ambiguous OCR structure",
    )
    parser.add_argument(
        "--no-raw-ocr",
        action="store_true",
        help="Do not write raw OCR pages/text into per-folder JSON outputs",
    )

    parser.add_argument(
        "--profile",
        choices=["auto", "mcq", "structured"],
        default="auto",
        help="Parser profile selection. auto uses Paper_1=>mcq, others=>structured",
    )
    parser.add_argument(
        "--expected-max-question",
        type=int,
        default=40,
        help="Expected max question number for MCQ papers",
    )
    parser.add_argument(
        "--ocr-grok-model",
        default=None,
        help="Override OCR normalization model (default from OCR_GROK_MODEL)",
    )
    parser.add_argument(
        "--grok-max-retries",
        type=int,
        default=None,
        help="Override Grok max retries for OCR normalization/repair",
    )
    parser.add_argument(
        "--grok-timeout-normalize",
        type=int,
        default=None,
        help="Override Grok timeout seconds for normalization calls",
    )
    parser.add_argument(
        "--grok-timeout-repair",
        type=int,
        default=None,
        help="Override Grok timeout seconds for pair-repair calls",
    )

    parser.add_argument("--debug", dest="debug", action="store_true")
    parser.add_argument("--no-debug", dest="debug", action="store_false")
    parser.set_defaults(debug=None)
    parser.add_argument(
        "--debug-dir",
        default=None,
        help="Debug root folder outside OA_MAIN_DATASET (default OCR_DEBUG_OUTPUT)",
    )
    parser.add_argument(
        "--debug-level",
        choices=["basic", "full"],
        default=None,
        help="basic=compact debug payloads, full=raw diagnostic payloads",
    )
    parser.add_argument("--progress-log", dest="progress_log", action="store_true")
    parser.add_argument("--no-progress-log", dest="progress_log", action="store_false")
    parser.set_defaults(progress_log=None)
    parser.add_argument(
        "--progress-log-path",
        default=None,
        help="Optional path for realtime progress log text file",
    )

    parser.add_argument("--dry-run", action="store_true", help="List selected pairs only")

    return parser.parse_args()


def load_settings(args: argparse.Namespace) -> PipelineSettings:
    _load_env_file_fallback(Path(".env"))

    grok_api_key = (os.getenv("Grok_API") or "").strip() or None
    grok_model = (
        args.ocr_grok_model
        or os.getenv("OCR_GROK_MODEL")
        or DEFAULT_GROK_MODEL
    ).strip()

    azure_endpoint = (os.getenv("AZURE_ENDPOINT") or "").strip()
    azure_key = (os.getenv("AZURE_KEY") or "").strip()
    grok_max_retries = max(
        1,
        _safe_int(
            args.grok_max_retries
            if args.grok_max_retries is not None
            else os.getenv("OCR_GROK_MAX_RETRIES"),
            2,
        ),
    )
    grok_timeout_normalize = max(
        15,
        _safe_int(
            args.grok_timeout_normalize
            if args.grok_timeout_normalize is not None
            else os.getenv("OCR_GROK_TIMEOUT_NORMALIZE"),
            90,
        ),
    )
    grok_timeout_repair = max(
        15,
        _safe_int(
            args.grok_timeout_repair
            if args.grok_timeout_repair is not None
            else os.getenv("OCR_GROK_TIMEOUT_REPAIR"),
            90,
        ),
    )

    if not azure_endpoint or not azure_key:
        raise EnvironmentError(
            "Missing OCR credentials. Set AZURE_ENDPOINT and AZURE_KEY."
        )

    env_debug_enabled = _safe_bool(os.getenv("OCR_DEBUG"), False)
    debug_enabled = env_debug_enabled if args.debug is None else bool(args.debug)
    debug_dir = Path(
        args.debug_dir or (os.getenv("OCR_DEBUG_DIR") or DEFAULT_DEBUG_DIR)
    )
    debug_level = (args.debug_level or (os.getenv("OCR_DEBUG_LEVEL") or "basic")).strip().lower()
    if debug_level not in {"basic", "full"}:
        debug_level = "basic"

    env_progress_enabled = _safe_bool(os.getenv("OCR_PROGRESS_LOG"), True)
    progress_enabled = env_progress_enabled if args.progress_log is None else bool(args.progress_log)

    # Keep run_id consistent across debug/progress outputs.
    run_id = _utc_slug_now()
    progress_override = args.progress_log_path or (os.getenv("OCR_PROGRESS_LOG_PATH") or "").strip()
    if not progress_override:
        progress_path = debug_dir / run_id / "pipeline_progress.txt"
    else:
        candidate = Path(progress_override)
        # Support both file and directory-style overrides.
        if progress_override.endswith(("/", "\\")) or candidate.suffix == "":
            progress_path = candidate / run_id / "pipeline_progress.txt"
        else:
            progress_path = candidate

    return PipelineSettings(
        grok_api_key=grok_api_key,
        grok_model=grok_model,
        grok_max_retries=grok_max_retries,
        grok_timeout_normalize=grok_timeout_normalize,
        grok_timeout_repair=grok_timeout_repair,
        azure_endpoint=azure_endpoint,
        azure_key=azure_key,
        ocr_per_page_timeout=_safe_float(os.getenv("OCR_PER_PAGE_TIMEOUT"), 120.0),
        ocr_overall_timeout=_safe_float(os.getenv("OCR_OVERALL_TIMEOUT"), 900.0),
        ocr_max_retries=max(1, _safe_int(os.getenv("OCR_MAX_RETRIES"), 3)),
        ocr_retry_base_delay=_safe_float(os.getenv("OCR_RETRY_BASE_DELAY"), 1.0),
        ocr_retry_max_delay=_safe_float(os.getenv("OCR_RETRY_MAX_DELAY"), 30.0),
        ocr_concurrent_pages=max(1, _safe_int(os.getenv("OCR_CONCURRENT_PAGES"), 2)),
        review_conf_threshold=_safe_float(os.getenv("OCR_REVIEW_CONFIDENCE_THRESHOLD"), 0.75),
        review_match_threshold=_safe_float(os.getenv("OCR_REVIEW_MATCH_THRESHOLD"), 0.80),
        use_grok_normalization=not args.no_grok_normalization,
        parser_profile=args.profile,
        expected_max_question=max(1, int(args.expected_max_question)),
        debug_enabled=debug_enabled,
        debug_dir=debug_dir,
        debug_level=debug_level,
        debug_run_id=run_id,
        progress_log_enabled=progress_enabled,
        progress_log_path=progress_path,
    )


def build_azure_client(settings: PipelineSettings) -> DocumentAnalysisClient:
    if AzureKeyCredential is None:
        raise RuntimeError(
            "Azure Form Recognizer SDK is required: pip install azure-ai-formrecognizer"
        )
    return DocumentAnalysisClient(
        endpoint=settings.azure_endpoint,
        credential=AzureKeyCredential(settings.azure_key),
    )


def read_index(index_path: Path) -> List[PairMeta]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    valid_pairs = payload.get("valid_pairs") or []
    out: List[PairMeta] = []
    for item in valid_pairs:
        out.append(
            PairMeta(
                subject=str(item["subject"]),
                year=int(item["year"]),
                session=str(item["session"]),
                paper=str(item["paper"]),
                variant=str(item["variant"]),
                qp_path=_normalize_pair_path(str(item["qp_path"])),
                ms_path=_normalize_pair_path(str(item["ms_path"])),
            )
        )
    return out


def apply_filters(
    pairs: Sequence[PairMeta],
    *,
    subjects: Sequence[str],
    year: Optional[int],
    session: Optional[str],
    paper: Optional[str],
    variant: Optional[str],
    limit: Optional[int],
) -> List[PairMeta]:
    subject_set = {s.strip() for s in subjects if s and s.strip()}
    out: List[PairMeta] = []
    for p in pairs:
        if subject_set and p.subject not in subject_set:
            continue
        if year is not None and p.year != year:
            continue
        if session and p.session != session:
            continue
        if paper and p.paper != paper:
            continue
        if variant and p.variant != variant:
            continue
        out.append(p)
        if limit is not None and len(out) >= limit:
            break
    return out


def load_run_report(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {
            "generated_at_utc": _utc_now(),
            "summary": {},
            "pairs": {},
        }
    return json.loads(path.read_text(encoding="utf-8"))


def _maybe_step(
    progress_logger: Optional[RunProgressLogger],
    step: str,
    *,
    pair_id: Optional[str] = None,
    message: Optional[str] = None,
) -> Any:
    if progress_logger and progress_logger.active:
        return progress_logger.step(step, pair_id=pair_id, message=message)
    return nullcontext()


def select_for_resume(
    pairs: Sequence[PairMeta],
    run_report: Dict[str, Any],
    *,
    resume: bool,
    force: bool,
) -> List[PairMeta]:
    if force:
        return list(pairs)
    if not resume:
        return list(pairs)

    existing = run_report.get("pairs") or {}
    out: List[PairMeta] = []
    for pair in pairs:
        status = (existing.get(pair.pair_id) or {}).get("status")
        if status == "accepted":
            continue
        out.append(pair)
    return out


def _polygon_to_pairs(poly: Any) -> List[Tuple[float, float]]:
    if not poly:
        return []
    pts: List[Tuple[float, float]] = []
    for p in poly:
        if hasattr(p, "x") and hasattr(p, "y"):
            pts.append((float(p.x), float(p.y)))
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            pts.append((float(p[0]), float(p[1])))
        elif isinstance(p, dict) and "x" in p and "y" in p:
            pts.append((float(p["x"]), float(p["y"])))
    return pts


def _single_page_pdf_bytes(src_doc: fitz.Document, page_index: int) -> bytes:
    out = fitz.open()
    out.insert_pdf(src_doc, from_page=page_index, to_page=page_index)
    data = out.tobytes()
    out.close()
    return data


def _backoff_delay(attempt: int, settings: PipelineSettings) -> float:
    if attempt <= 1:
        return 0.0
    delay = settings.ocr_retry_base_delay * (2 ** (attempt - 2))
    return min(delay, settings.ocr_retry_max_delay)


def _ocr_single_page(
    client: DocumentAnalysisClient,
    page_bytes: bytes,
    page_number: int,
    settings: PipelineSettings,
) -> Tuple[int, Dict[str, Any], Optional[str]]:
    last_error: Optional[str] = None
    for attempt in range(1, settings.ocr_max_retries + 1):
        delay = _backoff_delay(attempt, settings)
        if delay > 0:
            time.sleep(delay)
        try:
            poller = client.begin_analyze_document("prebuilt-read", page_bytes)
            result = poller.result(timeout=settings.ocr_per_page_timeout)

            if not result.pages:
                page_payload = {
                    "page_number": page_number,
                    "width": None,
                    "height": None,
                    "unit": None,
                    "lines": [],
                    "words": [],
                    "ocr_page_text": "",
                    "avg_confidence": 0.0,
                }
                return page_number, page_payload, None

            page = result.pages[0]
            words: List[Dict[str, Any]] = []
            lines: List[Dict[str, Any]] = []
            confidences: List[float] = []

            for w in (page.words or []):
                conf = _safe_float(getattr(w, "confidence", 0.0), 0.0)
                confidences.append(conf)
                words.append(
                    {
                        "text": str(getattr(w, "content", "") or "").strip(),
                        "confidence": conf,
                        "bbox": _polygon_to_pairs(getattr(w, "polygon", None)),
                    }
                )

            for ln in (page.lines or []):
                lines.append(
                    {
                        "text": str(getattr(ln, "content", "") or "").strip(),
                        "bbox": _polygon_to_pairs(getattr(ln, "polygon", None)),
                    }
                )

            page_text = "\n".join(l["text"] for l in lines if l.get("text")).strip()
            page_payload = {
                "page_number": page_number,
                "width": _safe_float(getattr(page, "width", 0.0), 0.0),
                "height": _safe_float(getattr(page, "height", 0.0), 0.0),
                "unit": str(getattr(page, "unit", "") or ""),
                "lines": lines,
                "words": words,
                "ocr_page_text": page_text,
                "avg_confidence": mean(confidences) if confidences else 0.0,
            }
            return page_number, page_payload, None
        except Exception as exc:  # noqa: BLE001
            last_error = f"attempt={attempt}: {exc}"

    return (
        page_number,
        {
            "page_number": page_number,
            "width": None,
            "height": None,
            "unit": None,
            "lines": [],
            "words": [],
            "ocr_page_text": "",
            "avg_confidence": 0.0,
        },
        last_error,
    )


def run_ocr_on_pdf(
    client: DocumentAnalysisClient,
    pdf_path: Path,
    settings: PipelineSettings,
) -> Dict[str, Any]:
    if fitz is None:
        raise RuntimeError("PyMuPDF is required: pip install pymupdf")

    start = time.perf_counter()
    with fitz.open(str(pdf_path)) as src_doc:
        page_count = src_doc.page_count
        page_bytes = [_single_page_pdf_bytes(src_doc, i) for i in range(page_count)]

    pages_output: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    workers = max(1, min(settings.ocr_concurrent_pages, len(page_bytes) or 1))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_ocr_single_page, client, page_bytes[i], i + 1, settings): i + 1
            for i in range(len(page_bytes))
        }
        for fut in as_completed(futures):
            page_no = futures[fut]
            pno, payload, err = fut.result()
            pages_output.append(payload)
            if err:
                errors.append({"page_number": pno or page_no, "error": err})

    pages_output.sort(key=lambda p: int(p.get("page_number") or 0))
    full_text = "\n\n".join(
        (p.get("ocr_page_text") or "").strip()
        for p in pages_output
        if (p.get("ocr_page_text") or "").strip()
    ).strip()

    conf_values = [
        _safe_float(p.get("avg_confidence"), 0.0)
        for p in pages_output
        if p.get("avg_confidence") is not None
    ]

    return {
        "source_pdf": str(pdf_path),
        "processed_at_utc": _utc_now(),
        "pages": pages_output,
        "full_text": full_text,
        "metadata": {
            "page_count": len(pages_output),
            "ocr_duration_seconds": round(time.perf_counter() - start, 3),
            "avg_confidence": round(mean(conf_values), 4) if conf_values else 0.0,
            "errors": errors,
            "error_count": len(errors),
        },
    }


def _extract_marks(text: str) -> Optional[int]:
    hit = MARK_RE.search(text or "")
    if not hit:
        return None
    for group in hit.groups():
        if group and group.isdigit():
            return int(group)
    return None


def parse_anchor(
    line_text: str,
    current_question_number: Optional[str] = None,
    *,
    allow_blank_rest: bool = False,
) -> Optional[Tuple[str, Optional[str], str]]:
    text = (line_text or "").strip()
    if not text:
        return None

    m_sub = QUESTION_WITH_SUB_RE.match(text)
    if m_sub:
        q_num = str(int(m_sub.group("q")))
        sub = f"({m_sub.group('sub').strip().lower()})"
        rest = _normalize_ws(m_sub.group("rest") or "")
        if rest or allow_blank_rest:
            return q_num, sub, rest

    m_q = QUESTION_ONLY_RE.match(text)
    if m_q:
        q_num = str(int(m_q.group("q")))
        rest = _normalize_ws(m_q.group("rest") or "")
        if rest or allow_blank_rest:
            return q_num, None, rest

    m_top = TOP_LEVEL_SECTION_Q_RE.match(text)
    if m_top:
        q_num = str(int(m_top.group("q")))
        rest = _normalize_ws(m_top.group("rest") or "")
        if rest or allow_blank_rest:
            return q_num, None, rest

    if current_question_number:
        m_only_sub = SUB_ONLY_RE.match(text)
        if m_only_sub:
            sub = f"({m_only_sub.group('sub').strip().lower()})"
            rest = _normalize_ws(m_only_sub.group("rest") or "")
            if rest or allow_blank_rest:
                return current_question_number, sub, rest

    return None


def _line_matches_noise(text: str, patterns: Sequence[re.Pattern[str]]) -> bool:
    value = text or ""
    return any(p.search(value) for p in patterns)


def _is_structured_page_excluded(
    page_text: str,
    *,
    kind: str,
    lines: Sequence[Dict[str, Any]],
) -> Tuple[bool, Optional[str]]:
    lower = (page_text or "").lower()
    if not lower:
        return False, None

    for hint in STRUCTURED_PAGE_EXCLUDE_HINTS:
        if hint in lower:
            if hint == "blank page":
                return True, "blank_page"
            if hint == "data sheet":
                return True, "data_sheet_page"
            if hint == "periodic table of the elements":
                return True, "periodic_table_page"
            return True, "instructions_page"

    # Footer/header-dominated pages are low-value noise for structured parsing.
    if kind == "qp" and lines:
        line_count = len(lines)
        noise_count = 0
        alpha_count = 0
        for line in lines:
            text = _normalize_ws(str((line or {}).get("text") or ""))
            if not text:
                continue
            if _line_matches_noise(text, STRUCTURED_LINE_NOISE_PATTERNS):
                noise_count += 1
            if re.search(r"[A-Za-z]", text):
                alpha_count += 1
        if line_count >= 8 and noise_count / max(line_count, 1) >= 0.85 and alpha_count <= 2:
            return True, "footer_heavy_page"

    return False, None


def _is_structured_ms_noise_line(text: str) -> bool:
    value = text or ""
    if _line_matches_noise(value, STRUCTURED_LINE_NOISE_PATTERNS):
        return True
    return _line_matches_noise(value, STRUCTURED_MS_LINE_NOISE_PATTERNS)


def _is_mcq_page_excluded(page_text: str) -> Tuple[bool, Optional[str]]:
    lower = (page_text or "").lower()
    if not lower:
        return False, None
    if "periodic table of elements" in lower:
        return True, "periodic_table_page"
    if "read these instructions first" in lower:
        return True, "instructions_page"
    if "answer sheet" in lower and "there are forty questions" in lower:
        return True, "instructions_page"
    for hint in MCQ_PAGE_EXCLUDE_HINTS:
        if hint in lower:
            return True, "excluded_hint"
    return False, None


def _looks_like_mcq_question_text(rest: str) -> bool:
    text = _normalize_ws(rest)
    if not text:
        return False
    lower = text.lower()
    if _line_matches_noise(text, MCQ_LINE_NOISE_PATTERNS):
        return False
    words = text.split()
    if len(words) < 3:
        return False
    if not re.search(r"[A-Za-z]", text):
        return False
    if MCQ_OPTION_PATTERN.search(text):
        return True
    if "?" in text:
        return True
    if any(marker in lower for marker in ["which", "what", "how", "statement", "explain", "describe"]):
        return True
    if len(words) >= 5:
        return True
    if len(words) >= 3 and re.match(r"^[A-Z]", text):
        return True
    return False


def _finalize_item(
    current: Dict[str, Any],
    *,
    item_text_field: str,
    force_marks: Optional[int] = None,
) -> Dict[str, Any]:
    text_blob = _normalize_ws(" ".join(current.pop("_parts", [])))
    current[item_text_field] = text_blob
    if force_marks is not None:
        current["marks"] = force_marks
    else:
        current["marks"] = _extract_marks(text_blob)
    return current


def _dedupe_items_by_question(
    items: Sequence[Dict[str, Any]],
    *,
    item_text_field: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in items:
        key = str(item.get("question_number") or "")
        grouped[key].append(item)

    deduped: List[Dict[str, Any]] = []
    duplicate_info: List[Dict[str, Any]] = []
    for key, group in grouped.items():
        if len(group) == 1:
            deduped.append(group[0])
            continue
        best = max(group, key=lambda it: len(_normalize_ws(str(it.get(item_text_field) or ""))))
        deduped.append(best)
        duplicate_info.append(
            {
                "question_number": key,
                "duplicates": len(group),
                "kept_length": len(_normalize_ws(str(best.get(item_text_field) or ""))),
            }
        )

    def _sort_key(item: Dict[str, Any]) -> Tuple[int, str]:
        q = str(item.get("question_number") or "")
        return (_safe_int(q, 9999), q)

    deduped.sort(key=_sort_key)
    return deduped, duplicate_info


def extract_qp_items_mcq(
    ocr_payload: Dict[str, Any],
    *,
    expected_max_question: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    items: List[Dict[str, Any]] = []
    candidates: List[Dict[str, Any]] = []
    page_filters: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    last_q: Optional[int] = None

    for page in ocr_payload.get("pages") or []:
        page_no = _safe_int(page.get("page_number"), 0)
        page_text = str(page.get("ocr_page_text") or "")
        excluded, reason = _is_mcq_page_excluded(page_text)
        page_filters.append(
            {
                "page_number": page_no,
                "excluded": excluded,
                "reason": reason,
                "line_count": len(page.get("lines") or []),
            }
        )

        for line_idx, line in enumerate(page.get("lines") or [], start=1):
            text = _normalize_ws(str(line.get("text") or ""))
            if not text:
                continue

            anchor = QUESTION_ONLY_RE.match(text)
            if anchor:
                q_num = _safe_int(anchor.group("q"), 0)
                rest = _normalize_ws(anchor.group("rest") or "")
                accepted = True
                reject_reason: Optional[str] = None
                is_page_number_header = (
                    line_idx <= 2 and text.isdigit() and _safe_int(text, -1) == page_no
                )

                if excluded:
                    accepted = False
                    reject_reason = f"page_excluded:{reason}"
                elif is_page_number_header:
                    accepted = False
                    reject_reason = "page_number_header"
                elif q_num < 1 or q_num > expected_max_question:
                    accepted = False
                    reject_reason = "out_of_expected_range"
                elif _line_matches_noise(text, MCQ_LINE_NOISE_PATTERNS) or _line_matches_noise(
                    rest, MCQ_LINE_NOISE_PATTERNS
                ):
                    accepted = False
                    reject_reason = "header_footer_noise"
                elif rest and not _looks_like_mcq_question_text(rest):
                    accepted = False
                    reject_reason = "not_question_like"
                elif last_q is not None and q_num <= last_q:
                    accepted = False
                    reject_reason = "non_monotonic_numbering"
                elif last_q is not None and (q_num - last_q) > 8:
                    accepted = False
                    reject_reason = "suspicious_number_jump"

                candidates.append(
                    {
                        "kind": "qp_mcq_anchor",
                        "page_number": page_no,
                        "line_number": line_idx,
                        "line_text": text,
                        "candidate_question_number": q_num,
                        "accepted": accepted,
                        "reason": "accepted" if accepted else reject_reason,
                    }
                )

                if accepted:
                    if current is not None:
                        items.append(_finalize_item(current, item_text_field="question_text", force_marks=1))
                    current = {
                        "question_number": str(q_num),
                        "sub_question": None,
                        "page_number": page_no,
                        "marks": 1,
                        "question_text": "",
                        "_parts": [rest] if rest else [],
                    }
                    last_q = q_num
                    continue

            if current is not None and not excluded and not _line_matches_noise(text, MCQ_LINE_NOISE_PATTERNS):
                current["_parts"].append(text)

    if current is not None:
        items.append(_finalize_item(current, item_text_field="question_text", force_marks=1))

    deduped, duplicates = _dedupe_items_by_question(items, item_text_field="question_text")
    cleaned = [it for it in deduped if _normalize_ws(str(it.get("question_text") or ""))]

    parser_meta = {
        "profile": "mcq",
        "expected_max_question": expected_max_question,
        "raw_items": len(items),
        "final_items": len(cleaned),
        "conflicts": [],
        "duplicates_removed": duplicates,
        "page_filters": page_filters,
        "candidate_summary": {
            "total": len(candidates),
            "accepted": sum(1 for c in candidates if c.get("accepted")),
            "rejected": sum(1 for c in candidates if not c.get("accepted")),
        },
    }
    return cleaned, parser_meta, candidates


def _structured_anchor_acceptance(
    *,
    q_num: str,
    rest: str,
    line_text: str,
    allow_blank_rest: bool,
) -> Tuple[bool, str]:
    q_int = _safe_int(q_num, -1)
    if q_int <= 0 or q_int > 200:
        return False, "out_of_structured_range"
    if _line_matches_noise(line_text, STRUCTURED_LINE_NOISE_PATTERNS):
        return False, "header_footer_noise"
    if not rest and not allow_blank_rest:
        return False, "blank_rest"
    return True, "accepted"


def extract_items_structured(
    ocr_payload: Dict[str, Any],
    *,
    item_text_field: str,
    allow_blank_rest: bool,
    kind: str = "qp",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    items: List[Dict[str, Any]] = []
    candidates: List[Dict[str, Any]] = []
    page_filters: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    current_q: Optional[str] = None

    for page in ocr_payload.get("pages") or []:
        page_no = _safe_int(page.get("page_number"), 0)
        page_text = str(page.get("ocr_page_text") or "")
        page_lines = page.get("lines") or []
        excluded, reason = _is_structured_page_excluded(
            page_text,
            kind=kind,
            lines=page_lines,
        )
        page_filters.append(
            {
                "page_number": page_no,
                "excluded": excluded,
                "reason": reason,
                "line_count": len(page_lines),
            }
        )
        if excluded:
            continue

        for line_idx, line in enumerate(page.get("lines") or [], start=1):
            text = _normalize_ws(str(line.get("text") or ""))
            if not text:
                continue

            anchor = parse_anchor(
                text,
                current_question_number=current_q,
                allow_blank_rest=allow_blank_rest,
            )
            if anchor:
                q_num, sub_q, rest = anchor
                accept, reason = _structured_anchor_acceptance(
                    q_num=q_num,
                    rest=rest,
                    line_text=text,
                    allow_blank_rest=allow_blank_rest,
                )
                candidates.append(
                    {
                        "kind": f"{item_text_field}_structured_anchor",
                        "page_number": page_no,
                        "line_number": line_idx,
                        "line_text": text,
                        "candidate_question_number": q_num,
                        "candidate_sub_question": sub_q,
                        "accepted": accept,
                        "reason": reason,
                    }
                )
                if not accept:
                    if current is not None:
                        current["_parts"].append(text)
                    continue

                current_q = q_num
                if current is not None:
                    items.append(_finalize_item(current, item_text_field=item_text_field))
                current = {
                    "question_number": q_num,
                    "sub_question": sub_q,
                    "page_number": page_no,
                    "marks": None,
                    item_text_field: "",
                    "_parts": [rest] if rest else [],
                }
            elif current is not None and not _line_matches_noise(text, STRUCTURED_LINE_NOISE_PATTERNS):
                current["_parts"].append(text)

    if current is not None:
        items.append(_finalize_item(current, item_text_field=item_text_field))

    cleaned = [
        it for it in items if _normalize_ws(str(it.get(item_text_field) or ""))
    ]
    parser_meta = {
        "profile": "structured",
        "raw_items": len(items),
        "final_items": len(cleaned),
        "page_filters": page_filters,
        "conflicts": [],
        "candidate_summary": {
            "total": len(candidates),
            "accepted": sum(1 for c in candidates if c.get("accepted")),
            "rejected": sum(1 for c in candidates if not c.get("accepted")),
        },
    }
    return cleaned, parser_meta, candidates


def _is_ms_noise_line(text: str) -> bool:
    return _line_matches_noise(text, MS_NOISE_PATTERNS)


def _structured_ms_anchor_acceptance(
    *,
    q_num: str,
    sub_q: Optional[str],
    rest: str,
    line_text: str,
) -> Tuple[bool, str]:
    accept, reason = _structured_anchor_acceptance(
        q_num=q_num,
        rest=rest,
        line_text=line_text,
        allow_blank_rest=False,
    )
    if not accept:
        return accept, reason
    lower_rest = (rest or "").strip().lower()
    if not lower_rest:
        return False, "blank_rest"
    if lower_rest in {"question", "answer", "marks", "general points", "conclusions"}:
        return False, "header_column_row"
    if re.fullmatch(r"\d{1,3}", lower_rest):
        return False, "marks_only_row"
    if sub_q is None and not re.search(r"[A-Za-z]", lower_rest):
        return False, "no_alpha_content"
    return True, "accepted"


def extract_ms_items_structured(
    ocr_payload: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    items: List[Dict[str, Any]] = []
    candidates: List[Dict[str, Any]] = []
    page_filters: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    current_q: Optional[str] = None

    for page in ocr_payload.get("pages") or []:
        page_no = _safe_int(page.get("page_number"), 0)
        page_text = str(page.get("ocr_page_text") or "")
        page_lines = page.get("lines") or []
        excluded, reason = _is_structured_page_excluded(
            page_text,
            kind="ms",
            lines=page_lines,
        )
        page_filters.append(
            {
                "page_number": page_no,
                "excluded": excluded,
                "reason": reason,
                "line_count": len(page_lines),
            }
        )
        if excluded:
            continue

        for line_idx, line in enumerate(page_lines, start=1):
            text = _normalize_ws(str(line.get("text") or ""))
            if not text:
                continue
            if _is_structured_ms_noise_line(text):
                candidates.append(
                    {
                        "kind": "marking_scheme_structured_noise",
                        "page_number": page_no,
                        "line_number": line_idx,
                        "line_text": text,
                        "accepted": False,
                        "reason": "noise_line",
                    }
                )
                continue

            anchor = parse_anchor(
                text,
                current_question_number=current_q,
                allow_blank_rest=True,
            )
            if anchor:
                q_num, sub_q, rest = anchor
                accept, reason = _structured_ms_anchor_acceptance(
                    q_num=q_num,
                    sub_q=sub_q,
                    rest=rest,
                    line_text=text,
                )
                candidates.append(
                    {
                        "kind": "marking_scheme_structured_anchor",
                        "page_number": page_no,
                        "line_number": line_idx,
                        "line_text": text,
                        "candidate_question_number": q_num,
                        "candidate_sub_question": sub_q,
                        "accepted": accept,
                        "reason": reason,
                    }
                )
                if not accept:
                    if current is not None and not re.fullmatch(r"\d{1,3}", text):
                        current["_parts"].append(text)
                    continue

                current_q = q_num
                if current is not None:
                    items.append(_finalize_item(current, item_text_field="marking_scheme"))
                current = {
                    "question_number": q_num,
                    "sub_question": sub_q,
                    "page_number": page_no,
                    "marks": _extract_marks(rest),
                    "marking_scheme": "",
                    "_parts": [rest] if rest else [],
                }
                continue

            if current is None:
                continue
            if re.fullmatch(r"\d{1,3}", text):
                continue
            if _is_structured_ms_noise_line(text):
                continue
            current["_parts"].append(text)

    if current is not None:
        items.append(_finalize_item(current, item_text_field="marking_scheme"))

    cleaned = [
        it
        for it in items
        if _normalize_ws(str(it.get("marking_scheme") or ""))
        and not re.fullmatch(r"\d{1,3}", _normalize_ws(str(it.get("marking_scheme") or "")))
    ]
    parser_meta = {
        "profile": "structured",
        "raw_items": len(items),
        "final_items": len(cleaned),
        "page_filters": page_filters,
        "conflicts": [],
        "candidate_summary": {
            "total": len(candidates),
            "accepted": sum(1 for c in candidates if c.get("accepted")),
            "rejected": sum(1 for c in candidates if not c.get("accepted")),
        },
    }
    return cleaned, parser_meta, candidates


def extract_ms_items_mcq(
    ocr_payload: Dict[str, Any],
    *,
    expected_max_question: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    table_started = False
    answers: Dict[int, Dict[str, Any]] = {}
    pending_questions: Deque[int] = deque()
    seen_question_heading = False
    seen_answer_or_marks_heading = False
    token_events: List[Dict[str, Any]] = []
    conflicts: List[Dict[str, Any]] = []
    duplicates: List[Dict[str, Any]] = []
    discounted_questions: set[int] = set()

    def assign_answer(q_num: int, ans: str, page_no: int, source_line: str, source: str) -> None:
        if q_num in answers:
            existing = answers[q_num]["answer"]
            if existing != ans:
                conflicts.append(
                    {
                        "question_number": q_num,
                        "existing": existing,
                        "incoming": ans,
                        "page_number": page_no,
                        "source": source,
                    }
                )
            else:
                duplicates.append(
                    {
                        "question_number": q_num,
                        "answer": ans,
                        "page_number": page_no,
                        "source": source,
                    }
                )
            return
        answers[q_num] = {
            "answer": ans,
            "page_number": page_no,
            "source_line": source_line,
            "source": source,
        }

    for page in ocr_payload.get("pages") or []:
        page_no = _safe_int(page.get("page_number"), 0)
        for line_idx, line in enumerate(page.get("lines") or [], start=1):
            text = _normalize_ws(str(line.get("text") or ""))
            if not text:
                continue
            lower = text.lower()
            stripped = lower.strip()

            if not table_started:
                if "question" in lower and ("answer" in lower or "marks" in lower or "key" in lower):
                    table_started = True
                    token_events.append(
                        {
                            "event": "table_started",
                            "page_number": page_no,
                            "line_number": line_idx,
                            "line_text": text,
                            "reason": "header_row",
                        }
                    )
                if stripped == "question":
                    seen_question_heading = True
                    token_events.append(
                        {
                            "event": "table_hint_question_heading",
                            "page_number": page_no,
                            "line_number": line_idx,
                            "line_text": text,
                        }
                    )
                    continue
                if stripped in {"answer", "marks", "key"}:
                    seen_answer_or_marks_heading = True
                    token_events.append(
                        {
                            "event": "table_hint_answer_or_marks_heading",
                            "page_number": page_no,
                            "line_number": line_idx,
                            "line_text": text,
                        }
                    )
                    continue
                if re.fullmatch(r"\d{1,3}\s+[A-Da-d]\b", text):
                    table_started = True
                    token_events.append(
                        {
                            "event": "table_started",
                            "page_number": page_no,
                            "line_number": line_idx,
                            "line_text": text,
                            "reason": "inline_number_answer",
                        }
                    )
                elif seen_question_heading and (
                    re.fullmatch(r"\d{1,3}", text)
                    or re.fullmatch(r"\d{1,3}\s*[\).:-]\s*.*", text)
                ):
                    table_started = True
                    token_events.append(
                        {
                            "event": "table_started",
                            "page_number": page_no,
                            "line_number": line_idx,
                            "line_text": text,
                            "reason": "question_heading_followed_by_number",
                        }
                    )
                elif seen_question_heading and seen_answer_or_marks_heading:
                    table_started = True
                    token_events.append(
                        {
                            "event": "table_started",
                            "page_number": page_no,
                            "line_number": line_idx,
                            "line_text": text,
                            "reason": "split_headers",
                        }
                    )

                if not table_started:
                    continue

            if _is_ms_noise_line(lower):
                token_events.append(
                    {
                        "event": "noise_line_skipped",
                        "page_number": page_no,
                        "line_number": line_idx,
                        "line_text": text,
                    }
                )
                continue

            discounted_match = re.search(r"(\d{1,3})\s+question\s+discounted", lower)
            if discounted_match:
                q_num = int(discounted_match.group(1))
                if 1 <= q_num <= expected_max_question:
                    discounted_questions.add(q_num)
                    while q_num in pending_questions:
                        pending_questions.remove(q_num)
                    token_events.append(
                        {
                            "event": "question_discounted",
                            "question_number": q_num,
                            "page_number": page_no,
                            "line_number": line_idx,
                            "line_text": text,
                        }
                    )
                    continue
            if "question discounted" in lower and pending_questions:
                q_num = pending_questions.pop()
                discounted_questions.add(q_num)
                token_events.append(
                    {
                        "event": "question_discounted",
                        "question_number": q_num,
                        "page_number": page_no,
                        "line_number": line_idx,
                        "line_text": text,
                    }
                )
                continue

            tokens = re.findall(r"[A-Za-z]+|\d+", text)
            if not tokens:
                continue

            for token in tokens:
                if token.isdigit():
                    q_num = int(token)
                    if q_num < 1 or q_num > expected_max_question:
                        token_events.append(
                            {
                                "event": "number_ignored_out_of_range",
                                "token": token,
                                "page_number": page_no,
                                "line_number": line_idx,
                            }
                        )
                        continue

                    pending_questions.append(q_num)
                    token_events.append(
                        {
                            "event": "pending_question",
                            "question_number": q_num,
                            "page_number": page_no,
                            "line_number": line_idx,
                            "pending_queue_size": len(pending_questions),
                        }
                    )
                    continue

                upper = token.upper()
                if upper not in {"A", "B", "C", "D"}:
                    continue

                if pending_questions:
                    pending_question: Optional[int] = None
                    while pending_questions and pending_question is None:
                        candidate = pending_questions.popleft()
                        if candidate in discounted_questions:
                            token_events.append(
                                {
                                    "event": "answer_ignored_discounted_question",
                                    "question_number": candidate,
                                    "answer": upper,
                                    "page_number": page_no,
                                    "line_number": line_idx,
                                }
                            )
                            continue
                        pending_question = candidate
                    if pending_question is None:
                        token_events.append(
                            {
                                "event": "orphan_answer",
                                "answer": upper,
                                "page_number": page_no,
                                "line_number": line_idx,
                            }
                        )
                        continue
                    assign_answer(
                        pending_question,
                        upper,
                        page_no,
                        text,
                        source="state_machine_pending_question",
                    )
                    token_events.append(
                        {
                            "event": "answer_assigned_to_pending",
                            "question_number": pending_question,
                            "answer": upper,
                            "page_number": page_no,
                            "line_number": line_idx,
                        }
                    )
                else:
                    token_events.append(
                        {
                            "event": "orphan_answer",
                            "answer": upper,
                            "page_number": page_no,
                            "line_number": line_idx,
                        }
                    )

    entries: List[Dict[str, Any]] = []
    for q_num in sorted(answers.keys()):
        value = answers[q_num]
        entries.append(
            {
                "question_number": str(q_num),
                "sub_question": None,
                "page_number": value["page_number"],
                "marks": 1,
                "marking_scheme": value["answer"],
            }
        )

    parser_meta = {
        "profile": "mcq",
        "expected_max_question": expected_max_question,
        "table_started": table_started,
        "final_items": len(entries),
        "pending_unanswered": list(pending_questions),
        "orphan_answers": [],
        "conflicts": conflicts,
        "duplicates": duplicates,
        "discounted_questions": sorted(discounted_questions),
    }
    return entries, parser_meta, token_events


def _detect_ambiguity(
    items: Sequence[Dict[str, Any]],
    full_text: str,
    *,
    profile: str,
    expected_max_question: int,
) -> bool:
    if not items:
        return True

    if profile == "mcq":
        unique_q = {
            str(it.get("question_number") or "")
            for it in items
            if str(it.get("question_number") or "").isdigit()
        }
        # MCQ should usually reconstruct nearly the full sequence. Trigger
        # normalization early if reconstruction is materially incomplete.
        if len(unique_q) < max(5, int(expected_max_question * 0.9)):
            return True
        return False

    if len(items) == 1 and len((full_text or "").split()) > 250:
        return True

    keys = [
        (
            str(it.get("question_number") or ""),
            str(it.get("sub_question") or ""),
        )
        for it in items
    ]
    dup_rate = 1.0 - (len(set(keys)) / max(len(keys), 1))
    return dup_rate > 0.35


def _mcq_side_metrics(items: Sequence[Dict[str, Any]], expected_max_question: int) -> Dict[str, Any]:
    valid, invalid = _numeric_ids(items)
    in_range = [q for q in valid if 1 <= q <= expected_max_question]
    out_of_range = [q for q in valid if q < 1 or q > expected_max_question]
    counts = Counter(in_range)
    unique = sorted(set(in_range))
    duplicate_count = sum((v - 1) for v in counts.values() if v > 1)
    expected = set(range(1, expected_max_question + 1))
    missing = sorted(expected - set(unique))
    score = (
        len(unique) * 10
        - len(missing) * 10
        - len(invalid) * 8
        - len(out_of_range) * 8
        - duplicate_count * 6
    )
    return {
        "unique_count": len(unique),
        "missing_count": len(missing),
        "missing_questions": missing,
        "invalid_count": len(invalid),
        "out_of_range_count": len(out_of_range),
        "duplicate_count": duplicate_count,
        "score": score,
    }


class GrokCallError(RuntimeError):
    def __init__(self, message: str, debug: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.debug = debug or {}


def _call_grok_json(
    grok_api_key: str,
    model: str,
    system_prompt: str,
    user_payload: Dict[str, Any],
    timeout: int = 120,
    max_retries: int = 3,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    headers = {
        "Authorization": f"Bearer {grok_api_key}",
        "Content-Type": "application/json",
    }

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        "temperature": 0.0,
        "max_tokens": 4000,
    }

    last_err: Optional[str] = None
    last_debug: Dict[str, Any] = {
        "model": model,
        "timeout_seconds": timeout,
        "max_retries": max_retries,
        "prompt_chars": len(json.dumps(user_payload, ensure_ascii=False)),
        "attempts": [],
    }
    attempts: List[Dict[str, Any]] = []
    for idx in range(1, max_retries + 1):
        try:
            started = time.perf_counter()
            resp = requests.post(
                GROK_CHAT_URL,
                headers=headers,
                json=body,
                timeout=timeout,
            )
            elapsed = round(time.perf_counter() - started, 3)
            attempts.append({"attempt": idx, "status_code": resp.status_code, "elapsed_seconds": elapsed})
            last_debug["attempts"] = attempts
            if resp.status_code >= 300:
                last_err = f"Grok API {resp.status_code}: {resp.text[:400]}"
                attempts[-1]["error_class"] = "http_error"
                attempts[-1]["response_chars"] = len(resp.text or "")
                continue
            payload = resp.json()
            content = payload["choices"][0]["message"]["content"]
            had_fence = str(content or "").strip().startswith("```")
            cleaned = _clean_json_from_llm(content)
            attempts[-1]["response_chars"] = len(str(content or ""))
            attempts[-1]["cleaned_chars"] = len(cleaned or "")
            attempts[-1]["had_fence_wrapper"] = bool(had_fence)
            parsed, parse_meta = _attempt_json_load_with_repair(cleaned)
            if parsed is None:
                parse_attempts = parse_meta.get("attempts") or []
                decode_entry: Optional[Dict[str, Any]] = None
                for item in reversed(parse_attempts):
                    if item.get("error_class") == "JSONDecodeError":
                        decode_entry = item
                        break
                if decode_entry:
                    json_error = decode_entry.get("json_error") or {}
                    line = _safe_int(json_error.get("line"), 0)
                    col = _safe_int(json_error.get("col"), 0)
                    pos = _safe_int(json_error.get("pos"), 0)
                    msg = str(json_error.get("msg") or decode_entry.get("error") or "JSON decode failed")
                    last_err = f"json_decode_error: line={line} col={col} pos={pos} msg={msg}"
                else:
                    last_err = f"json_decode_error: {str(parse_meta.get('error') or 'unknown')}"
                attempts[-1]["error_class"] = "json_decode_error"
                if decode_entry and decode_entry.get("json_error"):
                    attempts[-1]["json_error"] = decode_entry.get("json_error")
                attempts[-1]["parse_attempts"] = parse_attempts
                attempts[-1]["cleaned_preview"] = _truncate_text(cleaned, 600)
                continue
            strategy = str(parse_meta.get("strategy") or "direct")
            attempts[-1]["parse_strategy"] = strategy
            attempts[-1]["json_repaired"] = strategy not in {"direct", "raw_decode"}
            return parsed, {
                "success": True,
                "attempts": attempts,
                "model": model,
                "prompt_chars": len(json.dumps(user_payload, ensure_ascii=False)),
            }
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
            attempts.append(
                {
                    "attempt": idx,
                    "error": str(exc),
                    "error_class": exc.__class__.__name__,
                }
            )
            last_debug["attempts"] = attempts

    raise GrokCallError(
        f"Grok normalization failed: {last_err}",
        debug=last_debug,
    )


def maybe_normalize_with_grok(
    *,
    kind: str,
    profile: str,
    items: List[Dict[str, Any]],
    ocr_payload: Dict[str, Any],
    settings: PipelineSettings,
    progress_logger: Optional[RunProgressLogger] = None,
    pair_id: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, Any]]:
    issues: List[str] = []
    grok_debug: Dict[str, Any] = {
        "enabled": bool(settings.use_grok_normalization),
        "attempted": False,
        "skipped_reason": None,
        "model": settings.grok_model,
        "result": None,
    }

    if not settings.use_grok_normalization:
        grok_debug["skipped_reason"] = "disabled"
        if progress_logger and progress_logger.active:
            progress_logger.info(pair_id, f"grok_normalize_{kind}", "skipped: disabled")
        return items, issues, grok_debug

    if not settings.grok_api_key:
        if _detect_ambiguity(
            items,
            ocr_payload.get("full_text") or "",
            profile=profile,
            expected_max_question=settings.expected_max_question,
        ):
            issues.append("grok_skipped_missing_api_key")
        grok_debug["skipped_reason"] = "missing_api_key"
        if progress_logger and progress_logger.active:
            progress_logger.warn(pair_id, f"grok_normalize_{kind}", "skipped: missing_api_key")
        return items, issues, grok_debug

    ambiguous = _detect_ambiguity(
        items,
        ocr_payload.get("full_text") or "",
        profile=profile,
        expected_max_question=settings.expected_max_question,
    )
    if not ambiguous:
        grok_debug["skipped_reason"] = "not_ambiguous"
        if progress_logger and progress_logger.active:
            progress_logger.info(pair_id, f"grok_normalize_{kind}", "skipped: not_ambiguous")
        return items, issues, grok_debug

    field_name = "question_text" if kind == "qp" else "marking_scheme"
    item_instructions = (
        "For MCQ, question_number must be between 1 and expected_max_question and "
        "marking_scheme must be a single letter A/B/C/D (for ms)."
        if profile == "mcq"
        else "Preserve question_number and sub_question hierarchy."
    )

    system_prompt = (
        "You are a strict OCR post-processor. Return valid JSON only. "
        "Extract numbered exam items with fields: question_number, sub_question, "
        f"{field_name}, marks, page_number. "
        f"{item_instructions} Do not invent content."
    )
    user_payload = {
        "kind": kind,
        "profile": profile,
        "expected_max_question": settings.expected_max_question,
        "ocr_full_text": (ocr_payload.get("full_text") or "")[:70000],
        "existing_items": items,
        "output_schema": {
            "items": [
                {
                    "question_number": "1",
                    "sub_question": "(a)",
                    field_name: "text",
                    "marks": 1,
                    "page_number": 1,
                }
            ]
        },
    }

    grok_debug["attempted"] = True
    if progress_logger and progress_logger.active:
        progress_logger.info(
            pair_id,
            f"grok_normalize_{kind}",
            f"attempting model={settings.grok_model} input_items={len(items)}",
        )
    try:
        parsed, call_debug = _call_grok_json(
            grok_api_key=settings.grok_api_key,
            model=settings.grok_model,
            system_prompt=system_prompt,
            user_payload=user_payload,
            timeout=settings.grok_timeout_normalize,
            max_retries=settings.grok_max_retries,
        )
        grok_items = parsed.get("items") or []
        normalized: List[Dict[str, Any]] = []

        for it in grok_items:
            q_num = str(it.get("question_number") or "").strip()
            if not q_num or not q_num.isdigit():
                continue

            q_int = int(q_num)
            if profile == "mcq" and not (1 <= q_int <= settings.expected_max_question):
                continue

            sub = it.get("sub_question")
            if sub:
                sub = str(sub).strip().lower()
                if not sub.startswith("("):
                    sub = f"({sub})"

            text_value = _normalize_ws(str(it.get(field_name) or ""))
            if not text_value:
                continue

            if profile == "mcq" and kind == "ms":
                answer = text_value.upper().strip()
                if answer not in {"A", "B", "C", "D"}:
                    continue
                text_value = answer

            normalized.append(
                {
                    "question_number": str(q_int),
                    "sub_question": sub,
                    "page_number": _safe_int(it.get("page_number"), 1),
                    "marks": _safe_int(it.get("marks"), 0) or _extract_marks(text_value),
                    field_name: text_value,
                }
            )

        grok_debug["result"] = {
            "call": call_debug,
            "input_items": len(items),
            "output_items": len(normalized),
        }
        if progress_logger and progress_logger.active:
            progress_logger.info(
                pair_id,
                f"grok_normalize_{kind}",
                f"completed output_items={len(normalized)} response_attempts={len(call_debug.get('attempts') or [])}",
            )

        if normalized:
            if profile == "mcq":
                before_metrics = _mcq_side_metrics(items, settings.expected_max_question)
                after_metrics = _mcq_side_metrics(normalized, settings.expected_max_question)
                grok_debug["result"]["before_metrics"] = before_metrics
                grok_debug["result"]["after_metrics"] = after_metrics

                no_regression = (
                    after_metrics["missing_count"] <= before_metrics["missing_count"]
                    and after_metrics["unique_count"] >= before_metrics["unique_count"]
                    and after_metrics["invalid_count"] <= before_metrics["invalid_count"]
                    and after_metrics["out_of_range_count"] <= before_metrics["out_of_range_count"]
                )
                improved = after_metrics["score"] > before_metrics["score"]

                if not no_regression:
                    issues.append("grok_normalization_rejected_regression")
                    if progress_logger and progress_logger.active:
                        progress_logger.warn(
                            pair_id,
                            f"grok_normalize_{kind}",
                            (
                                "rejected_regression "
                                f"before={before_metrics['unique_count']}/{settings.expected_max_question} "
                                f"after={after_metrics['unique_count']}/{settings.expected_max_question}"
                            ),
                        )
                    return items, issues, grok_debug
                if not improved:
                    issues.append("grok_normalization_no_improvement")
                    if progress_logger and progress_logger.active:
                        progress_logger.info(
                            pair_id,
                            f"grok_normalize_{kind}",
                            "no improvement after normalization; baseline kept",
                        )
                    return items, issues, grok_debug

            issues.append("grok_normalization_applied")
            return normalized, issues, grok_debug

        issues.append("grok_normalization_empty_result")
        return items, issues, grok_debug
    except Exception as exc:  # noqa: BLE001
        issues.append(f"grok_normalization_failed:{exc}")
        error_meta: Dict[str, Any] = {"error": str(exc)}
        if isinstance(exc, GrokCallError):
            error_meta["call"] = exc.debug
        grok_debug["result"] = error_meta
        if progress_logger and progress_logger.active:
            call_meta = error_meta.get("call") or {}
            attempts = call_meta.get("attempts") or []
            if attempts:
                last = attempts[-1]
                progress_logger.error(
                    pair_id,
                    f"grok_normalize_{kind}",
                    f"failed model={settings.grok_model} class={last.get('error_class') or 'unknown'} "
                    f"response_chars={last.get('response_chars')} cleaned_chars={last.get('cleaned_chars')} "
                    f"json_error={last.get('json_error')} msg={str(exc)}",
                )
            else:
                progress_logger.error(
                    pair_id,
                    f"grok_normalize_{kind}",
                    f"failed model={settings.grok_model} msg={str(exc)}",
                )
        return items, issues, grok_debug


def _repair_trigger_reasons(profile: str, quality_metrics: Dict[str, Any]) -> List[str]:
    reasons: List[str] = []
    counts = quality_metrics.get("counts") or {}
    if profile == "mcq":
        if _safe_int(counts.get("missing_qp_questions"), 0) > 0:
            reasons.append("missing_qp_questions")
        if _safe_int(counts.get("missing_ms_questions"), 0) > 0:
            reasons.append("missing_ms_questions")
        if _safe_int(counts.get("parser_conflicts"), 0) > 0:
            reasons.append("parser_conflicts")
        if _safe_int(counts.get("duplicate_qp_ids"), 0) > 0:
            reasons.append("duplicate_qp_ids")
        if _safe_int(counts.get("duplicate_ms_ids"), 0) > 0:
            reasons.append("duplicate_ms_ids")
        if _safe_int(counts.get("pairing_issues"), 0) > 0:
            reasons.append("pairing_issues")
        if _safe_int(quality_metrics.get("unmatched_qp_count"), 0) > 0:
            reasons.append("unmatched_qp")
        if _safe_int(quality_metrics.get("unmatched_ms_count"), 0) > 0:
            reasons.append("unmatched_ms")
    else:
        if _safe_int(counts.get("invalid_qp_ids"), 0) > 0:
            reasons.append("invalid_qp_ids")
        if _safe_int(counts.get("invalid_ms_ids"), 0) > 0:
            reasons.append("invalid_ms_ids")
        if _safe_int(counts.get("duplicate_qp_ids"), 0) > 0:
            reasons.append("duplicate_qp_ids")
        if _safe_int(counts.get("duplicate_ms_ids"), 0) > 0:
            reasons.append("duplicate_ms_ids")
        if _safe_int(counts.get("parser_conflicts"), 0) > 0:
            reasons.append("parser_conflicts")
        if _safe_int(counts.get("pairing_issues"), 0) > 0:
            reasons.append("pairing_issues")
        if _safe_int(quality_metrics.get("unmatched_qp_count"), 0) > 0:
            reasons.append("unmatched_qp")
        if _safe_int(quality_metrics.get("unmatched_ms_count"), 0) > 0:
            reasons.append("unmatched_ms")
        if _safe_float(quality_metrics.get("coverage_balanced"), 1.0) < 0.8:
            reasons.append("low_coverage_balanced")
    return sorted(set(reasons))


def _extract_mcq_answer_letter(text: str) -> Optional[str]:
    value = _normalize_ws(text).upper()
    if not value:
        return None
    direct = value.strip()
    if direct in {"A", "B", "C", "D"}:
        return direct
    hits = re.findall(r"\b([ABCD])\b", value)
    if not hits:
        return None
    return hits[0]


def _collect_target_question_ids(
    quality_metrics: Dict[str, Any],
    *,
    limit: int = 24,
) -> List[int]:
    collected: set[int] = set()

    for key in ("missing_qp_questions", "missing_ms_questions"):
        for value in quality_metrics.get(key) or []:
            q = _safe_int(value, 0)
            if q > 0:
                collected.add(q)

    for key in ("unmatched_qp", "unmatched_ms"):
        for item in quality_metrics.get(key) or []:
            q_raw = str((item or {}).get("question_number") or "").strip()
            if q_raw.isdigit():
                collected.add(int(q_raw))

    for conflict in quality_metrics.get("parse_conflicts") or []:
        q = _safe_int((conflict or {}).get("question_number"), 0)
        if q > 0:
            collected.add(q)

    for issue in quality_metrics.get("pairing_issues") or []:
        match = re.search(r":(\d+)$", str(issue or ""))
        if match:
            collected.add(int(match.group(1)))

    return sorted(collected)[: max(1, int(limit))]


def _subset_items_by_question_ids(
    items: Sequence[Dict[str, Any]],
    target_ids: Sequence[int],
    *,
    text_field: str,
    max_text_chars: int = 900,
) -> List[Dict[str, Any]]:
    targets = {str(int(q)) for q in target_ids if int(q) > 0}
    out: List[Dict[str, Any]] = []
    for item in items:
        q_raw = str(item.get("question_number") or "").strip()
        if q_raw.isdigit():
            q_raw = str(int(q_raw))
        if q_raw not in targets:
            continue
        payload = dict(item)
        payload["sub_question"] = _norm_key(payload.get("question_number"), payload.get("sub_question"))[1] or None
        if text_field in payload:
            payload[text_field] = _truncate_text(_normalize_ws(str(payload.get(text_field) or "")), max_text_chars)
        out.append(payload)
    return out


def _extract_target_text_windows(
    full_text: str,
    target_ids: Sequence[int],
    *,
    window_chars: int = 1400,
    max_windows: int = 20,
) -> List[Dict[str, Any]]:
    source = str(full_text or "")
    if not source:
        return []
    windows: List[Dict[str, Any]] = []
    max_windows = max(1, int(max_windows))
    for q in list(target_ids)[:max_windows]:
        q_num = int(q)
        patterns = [
            re.compile(rf"(?m)^\s*{q_num}\b"),
            re.compile(rf"\b{q_num}\b"),
        ]
        match = None
        for pattern in patterns:
            match = pattern.search(source)
            if match is not None:
                break
        if match is None:
            continue
        half = max(200, window_chars // 2)
        start = max(0, match.start() - half)
        end = min(len(source), match.start() + half)
        snippet = _normalize_ws(source[start:end])
        if not snippet:
            continue
        windows.append(
            {
                "question_number": q_num,
                "start": start,
                "end": end,
                "text": _truncate_text(snippet, window_chars),
            }
        )
    return windows


def _sanitize_mcq_qp_items(
    items: Sequence[Dict[str, Any]],
    *,
    expected_max_question: int,
) -> List[Dict[str, Any]]:
    by_q: Dict[int, Dict[str, Any]] = {}
    for it in items:
        q_raw = str(it.get("question_number") or "").strip()
        if not q_raw.isdigit():
            continue
        q_num = int(q_raw)
        if q_num < 1 or q_num > expected_max_question:
            continue
        text = _normalize_ws(str(it.get("question_text") or ""))
        if not text:
            continue
        # Keep short stems if they still look like an MCQ line.
        if len(text.split()) < 4 and not MCQ_OPTION_PATTERN.search(text):
            continue
        page_no = max(1, _safe_int(it.get("page_number"), 1))
        payload = {
            "question_number": str(q_num),
            "sub_question": None,
            "page_number": page_no,
            "marks": 1,
            "question_text": text,
        }
        existing = by_q.get(q_num)
        if existing is None or len(payload["question_text"]) > len(existing.get("question_text", "")):
            by_q[q_num] = payload

    return [by_q[q] for q in sorted(by_q.keys())]


def _sanitize_mcq_ms_items(
    items: Sequence[Dict[str, Any]],
    *,
    expected_max_question: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_q: Dict[int, Dict[str, Any]] = {}
    conflicts: List[Dict[str, Any]] = []
    duplicates: List[Dict[str, Any]] = []
    for it in items:
        q_raw = str(it.get("question_number") or "").strip()
        if not q_raw.isdigit():
            continue
        q_num = int(q_raw)
        if q_num < 1 or q_num > expected_max_question:
            continue
        answer = _extract_mcq_answer_letter(str(it.get("marking_scheme") or ""))
        if answer is None:
            continue
        page_no = max(1, _safe_int(it.get("page_number"), 1))
        payload = {
            "question_number": str(q_num),
            "sub_question": None,
            "page_number": page_no,
            "marks": 1,
            "marking_scheme": answer,
        }
        existing = by_q.get(q_num)
        if existing is not None:
            if existing["marking_scheme"] != answer:
                conflicts.append(
                    {
                        "question_number": q_num,
                        "existing": existing["marking_scheme"],
                        "incoming": answer,
                        "source": "grok_repair",
                    }
                )
            else:
                duplicates.append(
                    {
                        "question_number": q_num,
                        "answer": answer,
                        "source": "grok_repair",
                    }
                )
            continue
        by_q[q_num] = payload

    return [by_q[q] for q in sorted(by_q.keys())], conflicts, duplicates


def _merge_mcq_items(
    base_items: Sequence[Dict[str, Any]],
    repaired_items: Sequence[Dict[str, Any]],
    *,
    text_field: str,
) -> List[Dict[str, Any]]:
    merged: Dict[int, Dict[str, Any]] = {}
    for item in base_items:
        q = str(item.get("question_number") or "").strip()
        if not q.isdigit():
            continue
        merged[int(q)] = dict(item)
    for item in repaired_items:
        q = str(item.get("question_number") or "").strip()
        if not q.isdigit():
            continue
        merged[int(q)] = dict(item)
    out = [merged[k] for k in sorted(merged.keys())]
    # Keep consistent shape in MCQ mode.
    for item in out:
        item["sub_question"] = None
        item["marks"] = 1
        if text_field == "question_text":
            item["question_text"] = _normalize_ws(str(item.get("question_text") or ""))
        else:
            ans = _extract_mcq_answer_letter(str(item.get("marking_scheme") or ""))
            item["marking_scheme"] = ans or ""
    return [it for it in out if _normalize_ws(str(it.get(text_field) or ""))]


def _score_mcq_candidate(pairing: Dict[str, Any], quality_metrics: Dict[str, Any]) -> int:
    counts = quality_metrics.get("counts") or {}
    score = 0
    score += int(pairing.get("matched_count") or 0) * 10
    score -= _safe_int(counts.get("missing_qp_questions"), 0) * 8
    score -= _safe_int(counts.get("missing_ms_questions"), 0) * 8
    score -= _safe_int(quality_metrics.get("unmatched_qp_count"), 0) * 4
    score -= _safe_int(quality_metrics.get("unmatched_ms_count"), 0) * 4
    score -= _safe_int(counts.get("parser_conflicts"), 0) * 6
    score -= _safe_int(counts.get("pairing_issues"), 0) * 6
    score -= _safe_int(counts.get("invalid_qp_ids"), 0) * 10
    score -= _safe_int(counts.get("invalid_ms_ids"), 0) * 10
    return score


def _evaluate_mcq_candidate(
    *,
    qp_items: Sequence[Dict[str, Any]],
    ms_items: Sequence[Dict[str, Any]],
    expected_max_question: int,
    ms_conflicts: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    pairing = pair_qp_ms(qp_items, ms_items, profile="mcq")
    quality = compute_quality_metrics(
        profile="mcq",
        qp_items=qp_items,
        ms_items=ms_items,
        pairing=pairing,
        expected_max_question=expected_max_question,
        qp_parser_meta={"conflicts": []},
        ms_parser_meta={"conflicts": list(ms_conflicts)},
    )
    score = _score_mcq_candidate(pairing, quality)
    return {
        "pairing": pairing,
        "quality_metrics": quality,
        "score": score,
        "ms_conflicts": list(ms_conflicts),
    }


def _sanitize_structured_qp_items(
    items: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    cleaned: List[Dict[str, Any]] = []
    for item in items:
        q, sub = _norm_key(item.get("question_number"), item.get("sub_question"))
        if not q.isdigit():
            continue
        q_int = int(q)
        if q_int < 1 or q_int > 200:
            continue
        text = _normalize_ws(str(item.get("question_text") or ""))
        if not text:
            continue
        if _line_matches_noise(text, STRUCTURED_LINE_NOISE_PATTERNS):
            continue
        cleaned.append(
            {
                "question_number": str(q_int),
                "sub_question": sub or None,
                "page_number": max(1, _safe_int(item.get("page_number"), 1)),
                "marks": _safe_int(item.get("marks"), 0) or _extract_marks(text),
                "question_text": text,
            }
        )
    deduped, conflicts = _dedupe_by_composite_key(cleaned, text_field="question_text")
    return deduped, conflicts


def _sanitize_structured_ms_items(
    items: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    cleaned: List[Dict[str, Any]] = []
    for item in items:
        q, sub = _norm_key(item.get("question_number"), item.get("sub_question"))
        if not q.isdigit():
            continue
        q_int = int(q)
        if q_int < 1 or q_int > 200:
            continue
        text = _normalize_ws(str(item.get("marking_scheme") or ""))
        if not text:
            continue
        if _is_structured_ms_noise_line(text):
            continue
        cleaned.append(
            {
                "question_number": str(q_int),
                "sub_question": sub or None,
                "page_number": max(1, _safe_int(item.get("page_number"), 1)),
                "marks": _safe_int(item.get("marks"), 0) or _extract_marks(text),
                "marking_scheme": text,
            }
        )
    deduped, conflicts = _dedupe_by_composite_key(cleaned, text_field="marking_scheme")
    return deduped, conflicts


def _merge_structured_items(
    base_items: Sequence[Dict[str, Any]],
    repaired_items: Sequence[Dict[str, Any]],
    *,
    text_field: str,
) -> List[Dict[str, Any]]:
    merged: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for item in base_items:
        key = _norm_key(item.get("question_number"), item.get("sub_question"))
        if not key[0]:
            continue
        merged[key] = dict(item)
    for item in repaired_items:
        key = _norm_key(item.get("question_number"), item.get("sub_question"))
        if not key[0]:
            continue
        merged[key] = dict(item)
    out = [dict(v) for v in merged.values()]
    for item in out:
        q, sub = _norm_key(item.get("question_number"), item.get("sub_question"))
        item["question_number"] = q
        item["sub_question"] = sub or None
        item["page_number"] = max(1, _safe_int(item.get("page_number"), 1))
        text_value = _normalize_ws(str(item.get(text_field) or ""))
        item[text_field] = text_value
        if not item.get("marks"):
            item["marks"] = _extract_marks(text_value)
    out = [item for item in out if item.get("question_number") and _normalize_ws(str(item.get(text_field) or ""))]
    out.sort(key=lambda it: (_safe_int(str(it.get("question_number") or ""), 9999), str(it.get("question_number") or ""), str(it.get("sub_question") or "")))
    return out


def _score_structured_candidate(pairing: Dict[str, Any], quality_metrics: Dict[str, Any]) -> int:
    counts = quality_metrics.get("counts") or {}
    coverage_balanced = _safe_float(quality_metrics.get("coverage_balanced"), 0.0)
    score = 0
    score += _safe_int(pairing.get("matched_count"), 0) * 10
    score += int(round(coverage_balanced * 100))
    score -= _safe_int(counts.get("invalid_qp_ids"), 0) * 10
    score -= _safe_int(counts.get("invalid_ms_ids"), 0) * 10
    score -= _safe_int(counts.get("duplicate_qp_ids"), 0) * 7
    score -= _safe_int(counts.get("duplicate_ms_ids"), 0) * 7
    score -= _safe_int(counts.get("pairing_issues"), 0) * 8
    score -= _safe_int(counts.get("parser_conflicts"), 0) * 8
    score -= _safe_int(quality_metrics.get("unmatched_qp_count"), 0) * 4
    score -= _safe_int(quality_metrics.get("unmatched_ms_count"), 0) * 4
    return score


def _evaluate_structured_candidate(
    *,
    qp_items: Sequence[Dict[str, Any]],
    ms_items: Sequence[Dict[str, Any]],
    qp_conflicts: Sequence[Dict[str, Any]],
    ms_conflicts: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    pairing = pair_qp_ms(qp_items, ms_items, profile="structured")
    quality = compute_quality_metrics(
        profile="structured",
        qp_items=qp_items,
        ms_items=ms_items,
        pairing=pairing,
        expected_max_question=40,
        qp_parser_meta={"conflicts": list(qp_conflicts)},
        ms_parser_meta={"conflicts": list(ms_conflicts)},
    )
    score = _score_structured_candidate(pairing, quality)
    return {
        "pairing": pairing,
        "quality_metrics": quality,
        "score": score,
        "qp_conflicts": list(qp_conflicts),
        "ms_conflicts": list(ms_conflicts),
    }


def _maybe_repair_structured_pair_with_grok(
    *,
    pair: PairMeta,
    qp_items: List[Dict[str, Any]],
    ms_items: List[Dict[str, Any]],
    qp_ocr_payload: Dict[str, Any],
    ms_ocr_payload: Dict[str, Any],
    baseline_pairing: Dict[str, Any],
    baseline_quality_metrics: Dict[str, Any],
    settings: PipelineSettings,
    progress_logger: Optional[RunProgressLogger],
    repair_debug: Dict[str, Any],
) -> Dict[str, Any]:
    issues: List[str] = []
    baseline_conflicts = baseline_quality_metrics.get("parser_conflicts") or {}
    baseline_qp_conflicts = list(baseline_conflicts.get("qp") or [])
    baseline_ms_conflicts = list(baseline_conflicts.get("ms") or [])
    trigger_reasons = _repair_trigger_reasons("structured", baseline_quality_metrics)
    repair_debug["trigger_reasons"] = trigger_reasons
    if not trigger_reasons:
        repair_debug["skipped_reason"] = "no_structural_trigger"
        if progress_logger and progress_logger.active:
            progress_logger.info(pair.pair_id, "grok_pair_repair", "skipped: no_structural_trigger")
        return {
            "qp_items": qp_items,
            "ms_items": ms_items,
            "pairing": baseline_pairing,
            "quality_metrics": baseline_quality_metrics,
            "issues": issues,
            "debug": repair_debug,
            "qp_conflicts": baseline_qp_conflicts,
            "ms_conflicts": baseline_ms_conflicts,
        }

    if not settings.use_grok_normalization:
        repair_debug["skipped_reason"] = "disabled"
        if progress_logger and progress_logger.active:
            progress_logger.info(pair.pair_id, "grok_pair_repair", "skipped: disabled")
        return {
            "qp_items": qp_items,
            "ms_items": ms_items,
            "pairing": baseline_pairing,
            "quality_metrics": baseline_quality_metrics,
            "issues": issues,
            "debug": repair_debug,
            "qp_conflicts": baseline_qp_conflicts,
            "ms_conflicts": baseline_ms_conflicts,
        }

    if not settings.grok_api_key:
        issues.append("grok_pair_repair_skipped_missing_api_key")
        repair_debug["skipped_reason"] = "missing_api_key"
        if progress_logger and progress_logger.active:
            progress_logger.warn(pair.pair_id, "grok_pair_repair", "skipped: missing_api_key")
        return {
            "qp_items": qp_items,
            "ms_items": ms_items,
            "pairing": baseline_pairing,
            "quality_metrics": baseline_quality_metrics,
            "issues": issues,
            "debug": repair_debug,
            "qp_conflicts": baseline_qp_conflicts,
            "ms_conflicts": baseline_ms_conflicts,
        }

    repair_debug["attempted"] = True
    baseline_score = _score_structured_candidate(baseline_pairing, baseline_quality_metrics)
    repair_debug["baseline_score"] = baseline_score
    if progress_logger and progress_logger.active:
        progress_logger.info(
            pair.pair_id,
            "grok_pair_repair",
            f"attempting model={settings.grok_model} baseline_score={baseline_score} triggers={trigger_reasons}",
        )

    target_ids = set(_collect_target_question_ids(baseline_quality_metrics, limit=30))
    for item in baseline_pairing.get("unmatched_qp") or []:
        q_raw = str((item or {}).get("question_number") or "").strip()
        if q_raw.isdigit():
            target_ids.add(int(q_raw))
    for item in baseline_pairing.get("unmatched_ms") or []:
        q_raw = str((item or {}).get("question_number") or "").strip()
        if q_raw.isdigit():
            target_ids.add(int(q_raw))
    target_question_numbers = sorted(target_ids)
    repair_debug["target_question_numbers"] = target_question_numbers
    if not target_question_numbers:
        repair_debug["skipped_reason"] = "no_target_questions"
        if progress_logger and progress_logger.active:
            progress_logger.info(pair.pair_id, "grok_pair_repair", "skipped: no_target_questions")
        return {
            "qp_items": qp_items,
            "ms_items": ms_items,
            "pairing": baseline_pairing,
            "quality_metrics": baseline_quality_metrics,
            "issues": issues,
            "debug": repair_debug,
            "qp_conflicts": baseline_qp_conflicts,
            "ms_conflicts": baseline_ms_conflicts,
        }

    qp_context_windows = _extract_target_text_windows(
        str(qp_ocr_payload.get("full_text") or ""),
        target_question_numbers,
        window_chars=1800,
        max_windows=24,
    )
    ms_context_windows = _extract_target_text_windows(
        str(ms_ocr_payload.get("full_text") or ""),
        target_question_numbers,
        window_chars=1200,
        max_windows=24,
    )
    repair_debug["target_windows"] = {
        "qp": len(qp_context_windows),
        "ms": len(ms_context_windows),
    }

    system_prompt = (
        "You repair OCR extraction for Cambridge O Level structured papers. "
        "Return JSON only with keys qp_fixes and ms_fixes. "
        "Each item fields: question_number, sub_question, page_number, marks, and either question_text or marking_scheme. "
        "Preserve numbering hierarchy and do not invent questions."
    )
    user_payload = {
        "pair_id": pair.pair_id,
        "profile": "structured",
        "trigger_reasons": trigger_reasons,
        "target_question_numbers": target_question_numbers,
        "current_quality_metrics": baseline_quality_metrics,
        "current_qp_target_items": _subset_items_by_question_ids(
            qp_items,
            target_question_numbers,
            text_field="question_text",
            max_text_chars=900,
        ),
        "current_ms_target_items": _subset_items_by_question_ids(
            ms_items,
            target_question_numbers,
            text_field="marking_scheme",
            max_text_chars=900,
        ),
        "qp_context_windows": qp_context_windows,
        "ms_context_windows": ms_context_windows,
    }
    repair_debug["payload_chars"] = len(json.dumps(user_payload, ensure_ascii=False))

    try:
        parsed, call_debug = _call_grok_json(
            grok_api_key=settings.grok_api_key,
            model=settings.grok_model,
            system_prompt=system_prompt,
            user_payload=user_payload,
            timeout=settings.grok_timeout_repair,
            max_retries=settings.grok_max_retries,
        )
        repair_debug["call"] = call_debug
    except Exception as exc:  # noqa: BLE001
        issues.append(f"grok_pair_repair_failed:{exc}")
        error_meta: Dict[str, Any] = {"error": str(exc)}
        if isinstance(exc, GrokCallError):
            error_meta["call"] = exc.debug
        repair_debug["result"] = error_meta
        if progress_logger and progress_logger.active:
            progress_logger.error(
                pair.pair_id,
                "grok_pair_repair",
                f"failed model={settings.grok_model} msg={str(exc)}",
            )
        return {
            "qp_items": qp_items,
            "ms_items": ms_items,
            "pairing": baseline_pairing,
            "quality_metrics": baseline_quality_metrics,
            "issues": issues,
            "debug": repair_debug,
            "qp_conflicts": baseline_qp_conflicts,
            "ms_conflicts": baseline_ms_conflicts,
        }

    raw_qp = (
        parsed.get("qp_fixes")
        or parsed.get("qp_items")
        or parsed.get("questions")
        or []
    )
    raw_ms = (
        parsed.get("ms_fixes")
        or parsed.get("ms_items")
        or parsed.get("marking_entries")
        or []
    )

    repaired_qp, repaired_qp_conflicts = _sanitize_structured_qp_items(raw_qp)
    repaired_ms, repaired_ms_conflicts = _sanitize_structured_ms_items(raw_ms)
    merged_qp = _merge_structured_items(qp_items, repaired_qp, text_field="question_text")
    merged_ms = _merge_structured_items(ms_items, repaired_ms, text_field="marking_scheme")

    baseline_candidate = {
        "name": "baseline",
        "qp_items": qp_items,
        "ms_items": ms_items,
        "pairing": baseline_pairing,
        "quality_metrics": baseline_quality_metrics,
        "score": baseline_score,
        "qp_conflicts": baseline_qp_conflicts,
        "ms_conflicts": baseline_ms_conflicts,
    }
    candidate_records: List[Dict[str, Any]] = [
        {
            "name": "baseline",
            "score": baseline_score,
            "matched": baseline_pairing.get("matched_count", 0),
            "coverage_balanced": baseline_quality_metrics.get("coverage_balanced"),
        }
    ]
    candidates: List[Dict[str, Any]] = [baseline_candidate]

    if repaired_qp or repaired_ms:
        repaired_qp_eval = repaired_qp if repaired_qp else list(qp_items)
        repaired_ms_eval = repaired_ms if repaired_ms else list(ms_items)
        repaired_eval = _evaluate_structured_candidate(
            qp_items=repaired_qp_eval,
            ms_items=repaired_ms_eval,
            qp_conflicts=repaired_qp_conflicts,
            ms_conflicts=repaired_ms_conflicts,
        )
        candidates.append(
            {
                "name": "repaired_only",
                "qp_items": repaired_qp_eval,
                "ms_items": repaired_ms_eval,
                "pairing": repaired_eval["pairing"],
                "quality_metrics": repaired_eval["quality_metrics"],
                "score": repaired_eval["score"],
                "qp_conflicts": repaired_eval["qp_conflicts"],
                "ms_conflicts": repaired_eval["ms_conflicts"],
            }
        )
        candidate_records.append(
            {
                "name": "repaired_only",
                "score": repaired_eval["score"],
                "matched": repaired_eval["pairing"].get("matched_count", 0),
                "coverage_balanced": repaired_eval["quality_metrics"].get("coverage_balanced"),
            }
        )

    merged_eval = _evaluate_structured_candidate(
        qp_items=merged_qp,
        ms_items=merged_ms,
        qp_conflicts=repaired_qp_conflicts,
        ms_conflicts=repaired_ms_conflicts,
    )
    candidates.append(
        {
            "name": "merged",
            "qp_items": merged_qp,
            "ms_items": merged_ms,
            "pairing": merged_eval["pairing"],
            "quality_metrics": merged_eval["quality_metrics"],
            "score": merged_eval["score"],
            "qp_conflicts": merged_eval["qp_conflicts"],
            "ms_conflicts": merged_eval["ms_conflicts"],
        }
    )
    candidate_records.append(
        {
            "name": "merged",
            "score": merged_eval["score"],
            "matched": merged_eval["pairing"].get("matched_count", 0),
            "coverage_balanced": merged_eval["quality_metrics"].get("coverage_balanced"),
        }
    )
    repair_debug["candidates"] = candidate_records
    repair_debug["repaired_counts"] = {
        "raw_qp_items": len(raw_qp),
        "raw_ms_items": len(raw_ms),
        "sanitized_qp_items": len(repaired_qp),
        "sanitized_ms_items": len(repaired_ms),
        "repaired_qp_conflicts": len(repaired_qp_conflicts),
        "repaired_ms_conflicts": len(repaired_ms_conflicts),
    }

    best = max(candidates, key=lambda c: int(c.get("score") or 0))
    repair_debug["selected_candidate"] = best["name"]
    if best["name"] != "baseline" and int(best["score"]) >= baseline_score + 2:
        issues.append("grok_pair_repair_applied")
        repair_debug["applied"] = True
        if progress_logger and progress_logger.active:
            progress_logger.info(
                pair.pair_id,
                "grok_pair_repair",
                f"applied candidate={best['name']} score={best['score']} baseline={baseline_score}",
            )
        return {
            "qp_items": list(best["qp_items"]),
            "ms_items": list(best["ms_items"]),
            "pairing": dict(best["pairing"]),
            "quality_metrics": dict(best["quality_metrics"]),
            "issues": issues,
            "debug": repair_debug,
            "qp_conflicts": list(best.get("qp_conflicts") or []),
            "ms_conflicts": list(best.get("ms_conflicts") or []),
        }

    issues.append("grok_pair_repair_no_improvement")
    if progress_logger and progress_logger.active:
        progress_logger.warn(
            pair.pair_id,
            "grok_pair_repair",
            f"no improvement candidate={best['name']} score={best['score']} baseline={baseline_score}",
        )
    return {
        "qp_items": qp_items,
        "ms_items": ms_items,
        "pairing": baseline_pairing,
        "quality_metrics": baseline_quality_metrics,
        "issues": issues,
        "debug": repair_debug,
        "qp_conflicts": baseline_qp_conflicts,
        "ms_conflicts": baseline_ms_conflicts,
    }


def maybe_repair_pair_with_grok(
    *,
    pair: PairMeta,
    profile: str,
    qp_items: List[Dict[str, Any]],
    ms_items: List[Dict[str, Any]],
    qp_ocr_payload: Dict[str, Any],
    ms_ocr_payload: Dict[str, Any],
    baseline_pairing: Dict[str, Any],
    baseline_quality_metrics: Dict[str, Any],
    settings: PipelineSettings,
    progress_logger: Optional[RunProgressLogger] = None,
) -> Dict[str, Any]:
    issues: List[str] = []
    repair_debug: Dict[str, Any] = {
        "enabled": bool(settings.use_grok_normalization),
        "attempted": False,
        "applied": False,
        "profile": profile,
        "model": settings.grok_model,
        "trigger_reasons": [],
        "skipped_reason": None,
        "candidates": [],
        "selected_candidate": "baseline",
    }

    if profile != "mcq":
        return _maybe_repair_structured_pair_with_grok(
            pair=pair,
            qp_items=qp_items,
            ms_items=ms_items,
            qp_ocr_payload=qp_ocr_payload,
            ms_ocr_payload=ms_ocr_payload,
            baseline_pairing=baseline_pairing,
            baseline_quality_metrics=baseline_quality_metrics,
            settings=settings,
            progress_logger=progress_logger,
            repair_debug=repair_debug,
        )

    trigger_reasons = _repair_trigger_reasons(profile, baseline_quality_metrics)
    repair_debug["trigger_reasons"] = trigger_reasons
    if not trigger_reasons:
        repair_debug["skipped_reason"] = "no_structural_trigger"
        if progress_logger and progress_logger.active:
            progress_logger.info(pair.pair_id, "grok_pair_repair", "skipped: no_structural_trigger")
        return {
            "qp_items": qp_items,
            "ms_items": ms_items,
            "pairing": baseline_pairing,
            "quality_metrics": baseline_quality_metrics,
            "issues": issues,
            "debug": repair_debug,
            "ms_conflicts": baseline_quality_metrics.get("parse_conflicts", []),
        }

    if not settings.use_grok_normalization:
        repair_debug["skipped_reason"] = "disabled"
        if progress_logger and progress_logger.active:
            progress_logger.info(pair.pair_id, "grok_pair_repair", "skipped: disabled")
        return {
            "qp_items": qp_items,
            "ms_items": ms_items,
            "pairing": baseline_pairing,
            "quality_metrics": baseline_quality_metrics,
            "issues": issues,
            "debug": repair_debug,
            "ms_conflicts": baseline_quality_metrics.get("parse_conflicts", []),
        }

    if not settings.grok_api_key:
        issues.append("grok_pair_repair_skipped_missing_api_key")
        repair_debug["skipped_reason"] = "missing_api_key"
        if progress_logger and progress_logger.active:
            progress_logger.warn(pair.pair_id, "grok_pair_repair", "skipped: missing_api_key")
        return {
            "qp_items": qp_items,
            "ms_items": ms_items,
            "pairing": baseline_pairing,
            "quality_metrics": baseline_quality_metrics,
            "issues": issues,
            "debug": repair_debug,
            "ms_conflicts": baseline_quality_metrics.get("parse_conflicts", []),
        }

    repair_debug["attempted"] = True
    baseline_score = _score_mcq_candidate(baseline_pairing, baseline_quality_metrics)
    repair_debug["baseline_score"] = baseline_score
    if progress_logger and progress_logger.active:
        progress_logger.info(
            pair.pair_id,
            "grok_pair_repair",
            f"attempting model={settings.grok_model} baseline_score={baseline_score} triggers={trigger_reasons}",
        )

    target_ids = set(_collect_target_question_ids(baseline_quality_metrics, limit=24))
    for item in baseline_pairing.get("unmatched_qp") or []:
        q_raw = str((item or {}).get("question_number") or "").strip()
        if q_raw.isdigit():
            target_ids.add(int(q_raw))
    for item in baseline_pairing.get("unmatched_ms") or []:
        q_raw = str((item or {}).get("question_number") or "").strip()
        if q_raw.isdigit():
            target_ids.add(int(q_raw))
    target_question_numbers = sorted(target_ids)
    repair_debug["target_question_numbers"] = target_question_numbers
    if not target_question_numbers:
        repair_debug["skipped_reason"] = "no_target_questions"
        if progress_logger and progress_logger.active:
            progress_logger.info(pair.pair_id, "grok_pair_repair", "skipped: no_target_questions")
        return {
            "qp_items": qp_items,
            "ms_items": ms_items,
            "pairing": baseline_pairing,
            "quality_metrics": baseline_quality_metrics,
            "issues": issues,
            "debug": repair_debug,
            "ms_conflicts": baseline_quality_metrics.get("parse_conflicts", []),
        }

    qp_context_windows = _extract_target_text_windows(
        str(qp_ocr_payload.get("full_text") or ""),
        target_question_numbers,
        window_chars=1400,
        max_windows=20,
    )
    ms_context_windows = _extract_target_text_windows(
        str(ms_ocr_payload.get("full_text") or ""),
        target_question_numbers,
        window_chars=800,
        max_windows=20,
    )
    repair_debug["target_windows"] = {
        "qp": len(qp_context_windows),
        "ms": len(ms_context_windows),
    }

    system_prompt = (
        "You repair OCR extraction for a Cambridge O Level MCQ paper. "
        "Return JSON object only (no markdown), using keys qp_fixes and ms_fixes. "
        "Each qp_fixes item: question_number (1..expected_max_question), page_number, question_text, marks=1. "
        "Each ms_fixes item: question_number (1..expected_max_question), page_number, marking_scheme one of A/B/C/D, marks=1. "
        "Only return entries for target_question_numbers. "
        "Prefer minimal targeted fixes; do not rewrite the full paper and do not invent content."
    )
    user_payload = {
        "pair_id": pair.pair_id,
        "profile": profile,
        "expected_max_question": settings.expected_max_question,
        "trigger_reasons": trigger_reasons,
        "target_question_numbers": target_question_numbers,
        "current_quality_metrics": baseline_quality_metrics,
        "current_qp_target_items": _subset_items_by_question_ids(
            qp_items,
            target_question_numbers,
            text_field="question_text",
            max_text_chars=900,
        ),
        "current_ms_target_items": _subset_items_by_question_ids(
            ms_items,
            target_question_numbers,
            text_field="marking_scheme",
            max_text_chars=20,
        ),
        "qp_context_windows": qp_context_windows,
        "ms_context_windows": ms_context_windows,
        "output_schema": {
            "qp_fixes": [
                {
                    "question_number": "1",
                    "page_number": 1,
                    "question_text": "MCQ question stem with options A/B/C/D",
                    "marks": 1,
                }
            ],
            "ms_fixes": [
                {
                    "question_number": "1",
                    "page_number": 2,
                    "marking_scheme": "A",
                    "marks": 1,
                }
            ],
        },
    }
    repair_debug["payload_chars"] = len(json.dumps(user_payload, ensure_ascii=False))

    try:
        parsed, call_debug = _call_grok_json(
            grok_api_key=settings.grok_api_key,
            model=settings.grok_model,
            system_prompt=system_prompt,
            user_payload=user_payload,
            timeout=settings.grok_timeout_repair,
            max_retries=settings.grok_max_retries,
        )
        repair_debug["call"] = call_debug
    except Exception as exc:  # noqa: BLE001
        issues.append(f"grok_pair_repair_failed:{exc}")
        error_meta: Dict[str, Any] = {"error": str(exc)}
        if isinstance(exc, GrokCallError):
            error_meta["call"] = exc.debug
        repair_debug["result"] = error_meta
        if progress_logger and progress_logger.active:
            call_meta = error_meta.get("call") or {}
            attempts = call_meta.get("attempts") or []
            if attempts:
                last = attempts[-1]
                progress_logger.error(
                    pair.pair_id,
                    "grok_pair_repair",
                    f"failed model={settings.grok_model} class={last.get('error_class') or 'unknown'} "
                    f"response_chars={last.get('response_chars')} cleaned_chars={last.get('cleaned_chars')} "
                    f"json_error={last.get('json_error')} msg={str(exc)}",
                )
            else:
                progress_logger.error(
                    pair.pair_id,
                    "grok_pair_repair",
                    f"failed model={settings.grok_model} msg={str(exc)}",
                )
        return {
            "qp_items": qp_items,
            "ms_items": ms_items,
            "pairing": baseline_pairing,
            "quality_metrics": baseline_quality_metrics,
            "issues": issues,
            "debug": repair_debug,
            "ms_conflicts": baseline_quality_metrics.get("parse_conflicts", []),
        }

    raw_qp = (
        parsed.get("qp_fixes")
        or parsed.get("qp_items")
        or parsed.get("questions")
        or []
    )
    raw_ms = (
        parsed.get("ms_fixes")
        or parsed.get("ms_items")
        or parsed.get("marking_entries")
        or []
    )

    repaired_qp = _sanitize_mcq_qp_items(
        raw_qp,
        expected_max_question=settings.expected_max_question,
    )
    repaired_ms, repaired_ms_conflicts, repaired_ms_duplicates = _sanitize_mcq_ms_items(
        raw_ms,
        expected_max_question=settings.expected_max_question,
    )

    merged_qp = _merge_mcq_items(qp_items, repaired_qp, text_field="question_text")
    merged_ms = _merge_mcq_items(ms_items, repaired_ms, text_field="marking_scheme")

    candidate_records: List[Dict[str, Any]] = []
    baseline_candidate = {
        "name": "baseline",
        "qp_items": qp_items,
        "ms_items": ms_items,
        "pairing": baseline_pairing,
        "quality_metrics": baseline_quality_metrics,
        "score": baseline_score,
        "ms_conflicts": baseline_quality_metrics.get("parse_conflicts", []),
    }
    candidate_records.append(
        {
            "name": "baseline",
            "score": baseline_score,
            "qp_items": len(qp_items),
            "ms_items": len(ms_items),
            "matched": baseline_pairing.get("matched_count", 0),
            "missing_qp_questions": len(baseline_quality_metrics.get("missing_qp_questions") or []),
            "missing_ms_questions": len(baseline_quality_metrics.get("missing_ms_questions") or []),
        }
    )

    candidates: List[Dict[str, Any]] = [baseline_candidate]
    if repaired_qp or repaired_ms:
        repaired_qp_eval = repaired_qp if repaired_qp else list(qp_items)
        repaired_ms_eval = repaired_ms if repaired_ms else list(ms_items)
        repaired_eval = _evaluate_mcq_candidate(
            qp_items=repaired_qp_eval,
            ms_items=repaired_ms_eval,
            expected_max_question=settings.expected_max_question,
            ms_conflicts=repaired_ms_conflicts,
        )
        candidates.append(
            {
                "name": "repaired_only",
                "qp_items": repaired_qp_eval,
                "ms_items": repaired_ms_eval,
                "pairing": repaired_eval["pairing"],
                "quality_metrics": repaired_eval["quality_metrics"],
                "score": repaired_eval["score"],
                "ms_conflicts": repaired_eval["ms_conflicts"],
            }
        )
        candidate_records.append(
            {
                "name": "repaired_only",
                "score": repaired_eval["score"],
                "qp_items": len(repaired_qp_eval),
                "ms_items": len(repaired_ms_eval),
                "matched": repaired_eval["pairing"].get("matched_count", 0),
                "missing_qp_questions": len(repaired_eval["quality_metrics"].get("missing_qp_questions") or []),
                "missing_ms_questions": len(repaired_eval["quality_metrics"].get("missing_ms_questions") or []),
            }
        )

    merged_eval = _evaluate_mcq_candidate(
        qp_items=merged_qp,
        ms_items=merged_ms,
        expected_max_question=settings.expected_max_question,
        ms_conflicts=repaired_ms_conflicts,
    )
    candidates.append(
        {
            "name": "merged",
            "qp_items": merged_qp,
            "ms_items": merged_ms,
            "pairing": merged_eval["pairing"],
            "quality_metrics": merged_eval["quality_metrics"],
            "score": merged_eval["score"],
            "ms_conflicts": merged_eval["ms_conflicts"],
        }
    )
    candidate_records.append(
        {
            "name": "merged",
            "score": merged_eval["score"],
            "qp_items": len(merged_qp),
            "ms_items": len(merged_ms),
            "matched": merged_eval["pairing"].get("matched_count", 0),
            "missing_qp_questions": len(merged_eval["quality_metrics"].get("missing_qp_questions") or []),
            "missing_ms_questions": len(merged_eval["quality_metrics"].get("missing_ms_questions") or []),
        }
    )

    best = max(candidates, key=lambda c: int(c.get("score") or 0))
    repair_debug["candidates"] = candidate_records
    repair_debug["selected_candidate"] = best["name"]
    repair_debug["repaired_counts"] = {
        "raw_qp_items": len(raw_qp),
        "raw_ms_items": len(raw_ms),
        "sanitized_qp_items": len(repaired_qp),
        "sanitized_ms_items": len(repaired_ms),
        "repaired_ms_conflicts": len(repaired_ms_conflicts),
        "repaired_ms_duplicates": len(repaired_ms_duplicates),
    }

    # Apply only when there is material improvement.
    if best["name"] != "baseline" and int(best["score"]) >= baseline_score + 2:
        issues.append("grok_pair_repair_applied")
        repair_debug["applied"] = True
        if progress_logger and progress_logger.active:
            progress_logger.info(
                pair.pair_id,
                "grok_pair_repair",
                f"applied candidate={best['name']} score={best['score']} baseline={baseline_score}",
            )
        return {
            "qp_items": list(best["qp_items"]),
            "ms_items": list(best["ms_items"]),
            "pairing": dict(best["pairing"]),
            "quality_metrics": dict(best["quality_metrics"]),
            "issues": issues,
            "debug": repair_debug,
            "ms_conflicts": list(best.get("ms_conflicts") or []),
        }

    issues.append("grok_pair_repair_no_improvement")
    if progress_logger and progress_logger.active:
        progress_logger.warn(
            pair.pair_id,
            "grok_pair_repair",
            f"no improvement candidate={best['name']} score={best['score']} baseline={baseline_score}",
        )
    return {
        "qp_items": qp_items,
        "ms_items": ms_items,
        "pairing": baseline_pairing,
        "quality_metrics": baseline_quality_metrics,
        "issues": issues,
        "debug": repair_debug,
        "ms_conflicts": baseline_quality_metrics.get("parse_conflicts", []),
    }


def _norm_key(q: Any, sub: Any) -> Tuple[str, str]:
    q_str = str(q or "").strip()
    if q_str.isdigit():
        q_str = str(int(q_str))
    sub_str = str(sub or "").strip().lower()
    if sub_str and not sub_str.startswith("("):
        sub_str = f"({sub_str})"
    return q_str, sub_str


def _item_quality_score(item: Dict[str, Any], *, text_field: str) -> int:
    text = _normalize_ws(str(item.get(text_field) or ""))
    lower = text.lower()
    score = min(len(text), 1200)
    if not text:
        score -= 1000
    if re.fullmatch(r"\d{1,3}", text):
        score -= 300
    if "page " in lower and " of " in lower:
        score -= 300
    if "cambridge" in lower:
        score -= 250
    if "question answer marks" in lower or "general points" in lower:
        score -= 250
    return score


def _dedupe_by_composite_key(
    items: Sequence[Dict[str, Any]],
    *,
    text_field: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    groups: Dict[Tuple[str, str], List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
    ordered: List[Tuple[Tuple[str, str], int]] = []
    for idx, item in enumerate(items):
        key = _norm_key(item.get("question_number"), item.get("sub_question"))
        if not key[0]:
            key = (f"__noq_{idx}", "")
        groups[key].append((idx, item))
        if len(groups[key]) == 1:
            ordered.append((key, idx))

    deduped: List[Dict[str, Any]] = []
    conflicts: List[Dict[str, Any]] = []
    for key, _first_idx in ordered:
        group = groups[key]
        if len(group) == 1:
            deduped.append(dict(group[0][1]))
            continue
        scored = sorted(
            group,
            key=lambda pair: (
                _item_quality_score(pair[1], text_field=text_field),
                -pair[0],
            ),
            reverse=True,
        )
        kept_idx, kept_item = scored[0]
        dropped = [idx for idx, _ in group if idx != kept_idx]
        conflicts.append(
            {
                "question_number": None if key[0].startswith("__noq_") else key[0],
                "sub_question": key[1] or None,
                "duplicates": len(group),
                "kept_index": kept_idx,
                "dropped_indexes": dropped,
                "kept_preview": _truncate_text(
                    _normalize_ws(str(kept_item.get(text_field) or "")),
                    220,
                ),
            }
        )
        deduped.append(dict(kept_item))
    return deduped, conflicts


def pair_qp_ms(
    qp_items: Sequence[Dict[str, Any]],
    ms_items: Sequence[Dict[str, Any]],
    *,
    profile: str,
) -> Dict[str, Any]:
    used_ms: set[int] = set()
    matches: List[Dict[str, Any]] = []
    unmatched_qp: List[Dict[str, Any]] = []
    unmatched_ms: List[Dict[str, Any]] = []
    pairing_attempts: List[Dict[str, Any]] = []
    pairing_issues: List[str] = []

    if profile == "mcq":
        ms_by_q: Dict[str, List[int]] = defaultdict(list)
        for idx, ms in enumerate(ms_items):
            key = str(ms.get("question_number") or "")
            if key.isdigit():
                key = str(int(key))
            ms_by_q[key].append(idx)

        for qp in qp_items:
            q_key = str(qp.get("question_number") or "")
            if q_key.isdigit():
                q_key = str(int(q_key))
            candidates = [idx for idx in ms_by_q.get(q_key, []) if idx not in used_ms]
            if not candidates:
                unmatched = dict(qp)
                unmatched["reason"] = "no_ms_match_for_question"
                unmatched_qp.append(unmatched)
                pairing_attempts.append(
                    {
                        "question_number": q_key,
                        "status": "unmatched",
                        "reason": "no_ms_match_for_question",
                    }
                )
                continue

            if len(candidates) > 1:
                pairing_issues.append(f"duplicate_ms_key:{q_key}")

            ms_idx = candidates[0]
            used_ms.add(ms_idx)
            ms = ms_items[ms_idx]
            matches.append(
                {
                    "question_number": q_key,
                    "sub_question": None,
                    "qp_page": qp.get("page_number"),
                    "ms_page": ms.get("page_number"),
                    "qp_text": qp.get("question_text", ""),
                    "marking_scheme": ms.get("marking_scheme", ""),
                    "qp_marks": qp.get("marks"),
                    "ms_marks": ms.get("marks"),
                    "match_method": "question_number_exact",
                }
            )
            pairing_attempts.append(
                {
                    "question_number": q_key,
                    "status": "matched",
                    "method": "question_number_exact",
                    "ms_index": ms_idx,
                }
            )

        unmatched_ms = [dict(ms, reason="no_qp_match_for_question") for i, ms in enumerate(ms_items) if i not in used_ms]
        return {
            "matches": matches,
            "unmatched_qp": unmatched_qp,
            "unmatched_ms": unmatched_ms,
            "matched_count": len(matches),
            "qp_count": len(qp_items),
            "ms_count": len(ms_items),
            "attempts": pairing_attempts,
            "issues": sorted(set(pairing_issues)),
        }

    ms_by_key: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for idx, ms in enumerate(ms_items):
        key = _norm_key(ms.get("question_number"), ms.get("sub_question"))
        ms_by_key[key].append(idx)

    for qp in qp_items:
        key = _norm_key(qp.get("question_number"), qp.get("sub_question"))
        ms_idx: Optional[int] = None
        method = ""

        for cand in ms_by_key.get(key, []):
            if cand not in used_ms:
                ms_idx = cand
                method = "exact_key"
                break

        if ms_idx is None and key[0]:
            candidates = []
            for idx, ms in enumerate(ms_items):
                if idx in used_ms:
                    continue
                ms_key = _norm_key(ms.get("question_number"), ms.get("sub_question"))
                if ms_key[0] == key[0]:
                    candidates.append(idx)
            # Safe fallback only when QP has no sub-question and there is exactly
            # one remaining MS entry for the same top-level question.
            if not key[1] and len(candidates) == 1:
                ms_idx = candidates[0]
                method = "same_question_unique_sub_safe"
            elif len(candidates) > 1:
                pairing_issues.append(f"ambiguous_sub_question_match:{key[0]}")

        if ms_idx is None:
            unmatched = dict(qp)
            unmatched["reason"] = "no_ms_match_for_key"
            unmatched_qp.append(unmatched)
            pairing_attempts.append(
                {
                    "question_number": key[0],
                    "sub_question": key[1] or None,
                    "status": "unmatched",
                    "reason": "no_ms_match_for_key",
                }
            )
            continue

        used_ms.add(ms_idx)
        ms = ms_items[ms_idx]
        matches.append(
            {
                "question_number": qp.get("question_number"),
                "sub_question": qp.get("sub_question"),
                "qp_page": qp.get("page_number"),
                "ms_page": ms.get("page_number"),
                "qp_text": qp.get("question_text", ""),
                "marking_scheme": ms.get("marking_scheme", ""),
                "qp_marks": qp.get("marks"),
                "ms_marks": ms.get("marks"),
                "match_method": method,
            }
        )
        pairing_attempts.append(
            {
                "question_number": key[0],
                "sub_question": key[1] or None,
                "status": "matched",
                "method": method,
                "ms_index": ms_idx,
            }
        )

    unmatched_ms = [dict(ms, reason="no_qp_match_for_key") for i, ms in enumerate(ms_items) if i not in used_ms]
    return {
        "matches": matches,
        "unmatched_qp": unmatched_qp,
        "unmatched_ms": unmatched_ms,
        "matched_count": len(matches),
        "qp_count": len(qp_items),
        "ms_count": len(ms_items),
        "attempts": pairing_attempts,
        "issues": sorted(set(pairing_issues)),
    }


def _numeric_ids(items: Sequence[Dict[str, Any]]) -> Tuple[List[int], List[str]]:
    valid: List[int] = []
    invalid: List[str] = []
    for item in items:
        q = str(item.get("question_number") or "").strip()
        if q.isdigit():
            valid.append(int(q))
        else:
            invalid.append(q)
    return valid, invalid


def compute_quality_metrics(
    *,
    profile: str,
    qp_items: Sequence[Dict[str, Any]],
    ms_items: Sequence[Dict[str, Any]],
    pairing: Dict[str, Any],
    expected_max_question: int,
    qp_parser_meta: Dict[str, Any],
    ms_parser_meta: Dict[str, Any],
) -> Dict[str, Any]:
    qp_valid, qp_invalid = _numeric_ids(qp_items)
    ms_valid, ms_invalid = _numeric_ids(ms_items)

    def _composite_duplicates(items: Sequence[Dict[str, Any]]) -> Tuple[List[Tuple[str, str]], List[int]]:
        key_counts = Counter(
            _norm_key(item.get("question_number"), item.get("sub_question"))
            for item in items
        )
        dup_keys = sorted(
            [k for k, v in key_counts.items() if v > 1 and k[0]],
            key=lambda k: (_safe_int(k[0], 9999), k[0], k[1]),
        )
        dup_ids: List[int] = []
        for q, _sub in dup_keys:
            if q.isdigit():
                dup_ids.append(int(q))
        return dup_keys, sorted(set(dup_ids))

    duplicate_qp_keys, duplicate_qp_ids = _composite_duplicates(qp_items)
    duplicate_ms_keys, duplicate_ms_ids = _composite_duplicates(ms_items)
    matched_count = _safe_int(pairing.get("matched_count"), 0)
    qp_count = len(qp_items)
    ms_count = len(ms_items)
    coverage_qp = matched_count / max(qp_count, 1)
    coverage_ms = matched_count / max(ms_count, 1)
    coverage_balanced = matched_count / max(max(qp_count, ms_count), 1)

    metrics: Dict[str, Any] = {
        "profile": profile,
        "invalid_qp_ids": sorted(set(qp_invalid)),
        "invalid_ms_ids": sorted(set(ms_invalid)),
        "duplicate_qp_ids": duplicate_qp_ids,
        "duplicate_ms_ids": duplicate_ms_ids,
        "duplicate_qp_keys": [
            {"question_number": q, "sub_question": sub or None} for q, sub in duplicate_qp_keys
        ],
        "duplicate_ms_keys": [
            {"question_number": q, "sub_question": sub or None} for q, sub in duplicate_ms_keys
        ],
        "pairing_issues": pairing.get("issues", []),
        "qp_count": qp_count,
        "ms_count": ms_count,
        "matched_count": matched_count,
        "unmatched_qp_count": len(pairing.get("unmatched_qp") or []),
        "unmatched_ms_count": len(pairing.get("unmatched_ms") or []),
        "unmatched_qp": pairing.get("unmatched_qp") or [],
        "unmatched_ms": pairing.get("unmatched_ms") or [],
        "coverage_qp": round(coverage_qp, 4),
        "coverage_ms": round(coverage_ms, 4),
        "coverage_balanced": round(coverage_balanced, 4),
        "parser_conflicts": {
            "qp": qp_parser_meta.get("conflicts", []),
            "ms": ms_parser_meta.get("conflicts", []),
        },
    }

    if profile == "mcq":
        discounted = sorted(
            {
                _safe_int(value, 0)
                for value in (ms_parser_meta.get("discounted_questions") or [])
                if 1 <= _safe_int(value, 0) <= expected_max_question
            }
        )
        expected = set(range(1, expected_max_question + 1)) - set(discounted)
        qp_set = set(qp_valid)
        ms_set = set(ms_valid)
        metrics["expected_question_range"] = {
            "start": 1,
            "end": expected_max_question,
        }
        if discounted:
            metrics["discounted_questions"] = discounted
        metrics["missing_qp_questions"] = sorted(expected - qp_set)
        metrics["missing_ms_questions"] = sorted(expected - ms_set)
        metrics["extra_qp_questions"] = sorted(qp_set - expected)
        metrics["extra_ms_questions"] = sorted(ms_set - expected)
        metrics["parse_conflicts"] = ms_parser_meta.get("conflicts", [])
    else:
        metrics["parse_conflicts"] = (
            (qp_parser_meta.get("conflicts") or []) + (ms_parser_meta.get("conflicts") or [])
        )

    metrics["counts"] = {
        "invalid_qp_ids": len(metrics.get("invalid_qp_ids") or []),
        "invalid_ms_ids": len(metrics.get("invalid_ms_ids") or []),
        "duplicate_qp_ids": len(duplicate_qp_ids),
        "duplicate_ms_ids": len(duplicate_ms_ids),
        "pairing_issues": len(metrics.get("pairing_issues") or []),
        "parser_conflicts": len(metrics.get("parse_conflicts") or []),
        "missing_qp_questions": len(metrics.get("missing_qp_questions") or []),
        "missing_ms_questions": len(metrics.get("missing_ms_questions") or []),
    }
    return metrics


def determine_status(
    *,
    qp_confidence: float,
    ms_confidence: float,
    matched_count: int,
    qp_count: int,
    ms_count: int,
    issues: Sequence[str],
    quality_metrics: Dict[str, Any],
    settings: PipelineSettings,
) -> Tuple[str, float, List[str], Dict[str, Any]]:
    reasons: List[str] = []
    if qp_count == 0:
        reasons.append("no_qp_items")
    if ms_count == 0:
        reasons.append("no_ms_items")
    if matched_count == 0:
        reasons.append("no_matches")

    if reasons:
        validation = {
            "has_qp_items": qp_count > 0,
            "has_ms_items": ms_count > 0,
            "has_matches": matched_count > 0,
            "confidence_ok": False,
            "coverage_ok": False,
            "structural_checks_ok": False,
        }
        return "failed", 0.0, sorted(set(reasons)), validation

    coverage_qp = matched_count / max(qp_count, 1)
    coverage_ms = matched_count / max(ms_count, 1)
    coverage_balanced = matched_count / max(max(qp_count, ms_count), 1)
    if quality_metrics:
        coverage_qp = _safe_float(quality_metrics.get("coverage_qp"), coverage_qp)
        coverage_ms = _safe_float(quality_metrics.get("coverage_ms"), coverage_ms)
        coverage_balanced = _safe_float(quality_metrics.get("coverage_balanced"), coverage_balanced)

    if qp_confidence < settings.review_conf_threshold:
        reasons.append("low_qp_confidence")
    if ms_confidence < settings.review_conf_threshold:
        reasons.append("low_ms_confidence")
    if coverage_balanced < settings.review_match_threshold:
        reasons.append("low_match_coverage")

    if (quality_metrics.get("counts") or {}).get("invalid_qp_ids", 0) > 0:
        reasons.append("invalid_qp_ids_detected")
    if (quality_metrics.get("counts") or {}).get("invalid_ms_ids", 0) > 0:
        reasons.append("invalid_ms_ids_detected")
    if (quality_metrics.get("counts") or {}).get("duplicate_qp_ids", 0) > 0:
        reasons.append("duplicate_qp_ids_detected")
    if (quality_metrics.get("counts") or {}).get("duplicate_ms_ids", 0) > 0:
        reasons.append("duplicate_ms_ids_detected")
    if (quality_metrics.get("counts") or {}).get("parser_conflicts", 0) > 0:
        reasons.append("parse_conflicts_detected")
    if (quality_metrics.get("counts") or {}).get("pairing_issues", 0) > 0:
        reasons.append("pairing_issues_detected")

    if quality_metrics.get("profile") == "mcq":
        if (quality_metrics.get("counts") or {}).get("missing_ms_questions", 0) > 0:
            reasons.append("missing_expected_ms_questions")
        if (quality_metrics.get("counts") or {}).get("missing_qp_questions", 0) > 0:
            reasons.append("missing_expected_qp_questions")

    if any(i.startswith("grok_normalization") for i in issues):
        reasons.append("used_grok_normalization")

    structural_ok = not any(
        r
        for r in reasons
        if r
        in {
            "invalid_qp_ids_detected",
            "invalid_ms_ids_detected",
            "duplicate_qp_ids_detected",
            "duplicate_ms_ids_detected",
            "parse_conflicts_detected",
            "pairing_issues_detected",
            "missing_expected_ms_questions",
            "missing_expected_qp_questions",
        }
    )
    validation = {
        "has_qp_items": qp_count > 0,
        "has_ms_items": ms_count > 0,
        "has_matches": matched_count > 0,
        "confidence_ok": qp_confidence >= settings.review_conf_threshold
        and ms_confidence >= settings.review_conf_threshold,
        "coverage_ok": coverage_balanced >= settings.review_match_threshold,
        "coverage_qp": round(coverage_qp, 4),
        "coverage_ms": round(coverage_ms, 4),
        "coverage_balanced": round(coverage_balanced, 4),
        "structural_checks_ok": structural_ok,
    }

    if reasons:
        return "review_required", coverage_balanced, sorted(set(reasons)), validation

    return "accepted", coverage_balanced, [], validation


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _compact_candidates(
    candidates: Sequence[Dict[str, Any]],
    *,
    level: str,
    max_items: int = 500,
) -> Dict[str, Any]:
    if level == "full":
        return {
            "total": len(candidates),
            "items": list(candidates),
        }
    return {
        "total": len(candidates),
        "accepted": sum(1 for c in candidates if c.get("accepted") is True),
        "rejected": sum(1 for c in candidates if c.get("accepted") is False),
        "items_preview": list(candidates[:max_items]),
    }


def _confidence_distribution(pages: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    bins = {
        "0.0-0.5": 0,
        "0.5-0.7": 0,
        "0.7-0.85": 0,
        "0.85-0.95": 0,
        "0.95-1.0": 0,
    }
    for page in pages:
        conf = _safe_float(page.get("avg_confidence"), 0.0)
        if conf < 0.5:
            bins["0.0-0.5"] += 1
        elif conf < 0.7:
            bins["0.5-0.7"] += 1
        elif conf < 0.85:
            bins["0.7-0.85"] += 1
        elif conf < 0.95:
            bins["0.85-0.95"] += 1
        else:
            bins["0.95-1.0"] += 1
    return bins


def _build_ocr_debug_payload(
    *,
    kind: str,
    ocr_payload: Dict[str, Any],
    page_filters: Optional[Sequence[Dict[str, Any]]],
    level: str,
) -> Dict[str, Any]:
    pages = ocr_payload.get("pages") or []
    base = {
        "kind": kind,
        "source_pdf": ocr_payload.get("source_pdf"),
        "processed_at_utc": ocr_payload.get("processed_at_utc"),
        "metadata": ocr_payload.get("metadata") or {},
        "confidence_distribution": _confidence_distribution(pages),
        "page_filters": list(page_filters or []),
    }

    if level == "full":
        base["raw_ocr"] = {
            "full_text": ocr_payload.get("full_text", ""),
            "pages": pages,
        }
        return base

    base["full_text_preview"] = _truncate_text(str(ocr_payload.get("full_text") or ""), 8000)
    base["page_summaries"] = [
        {
            "page_number": _safe_int(page.get("page_number"), 0),
            "line_count": len(page.get("lines") or []),
            "word_count": len(page.get("words") or []),
            "avg_confidence": _safe_float(page.get("avg_confidence"), 0.0),
        }
        for page in pages
    ]
    return base


def _debug_pair_dir(settings: PipelineSettings, pair: PairMeta) -> Path:
    return (
        settings.debug_dir
        / settings.debug_run_id
        / pair.subject
        / str(pair.year)
        / pair.session
        / pair.paper
        / pair.variant
    )


def write_pair_debug_files(
    *,
    settings: PipelineSettings,
    pair: PairMeta,
    qp_ocr_payload: Dict[str, Any],
    ms_ocr_payload: Dict[str, Any],
    qp_page_filters: Sequence[Dict[str, Any]],
    anchor_candidates_qp: Sequence[Dict[str, Any]],
    anchor_candidates_ms: Sequence[Dict[str, Any]],
    normalization_debug: Dict[str, Any],
    pairing_debug: Dict[str, Any],
    validation_debug: Dict[str, Any],
    grok_debug: Dict[str, Any],
) -> Optional[Path]:
    if not settings.debug_enabled:
        return None

    pair_debug_dir = _debug_pair_dir(settings, pair)
    pair_debug_dir.mkdir(parents=True, exist_ok=True)

    files_payload: Dict[str, Dict[str, Any]] = {
        "qp_ocr_debug.json": _build_ocr_debug_payload(
            kind="qp",
            ocr_payload=qp_ocr_payload,
            page_filters=qp_page_filters,
            level=settings.debug_level,
        ),
        "ms_ocr_debug.json": _build_ocr_debug_payload(
            kind="ms",
            ocr_payload=ms_ocr_payload,
            page_filters=None,
            level=settings.debug_level,
        ),
        "anchor_detection_debug.json": {
            "qp": _compact_candidates(anchor_candidates_qp, level=settings.debug_level),
            "ms": _compact_candidates(anchor_candidates_ms, level=settings.debug_level),
        },
        "normalization_debug.json": normalization_debug,
        "matching_debug.json": pairing_debug,
        "validation_debug.json": validation_debug,
        "grok_debug.json": grok_debug,
    }

    written_files: List[str] = []
    for filename, payload in files_payload.items():
        path = pair_debug_dir / filename
        write_json(path, payload)
        written_files.append(filename)

    manifest = {
        "pair_id": pair.pair_id,
        "generated_at_utc": _utc_now(),
        "debug_level": settings.debug_level,
        "run_id": settings.debug_run_id,
        "files": written_files,
    }
    write_json(pair_debug_dir / "pair_debug_manifest.json", manifest)
    return pair_debug_dir


def resolve_parser_profile(settings: PipelineSettings, pair: PairMeta) -> str:
    if settings.parser_profile != "auto":
        return settings.parser_profile
    return "mcq" if str(pair.paper).strip().lower() == "paper_1" else "structured"


def process_pair(
    pair: PairMeta,
    settings: PipelineSettings,
    client: DocumentAnalysisClient,
    *,
    write_raw_ocr: bool,
    progress_logger: Optional[RunProgressLogger] = None,
) -> Dict[str, Any]:
    start = time.perf_counter()
    issues: List[str] = []
    pair_id = pair.pair_id

    parser_profile = resolve_parser_profile(settings, pair)

    with _maybe_step(progress_logger, "ocr_qp", pair_id=pair_id, message="Azure OCR on question paper"):
        qp_ocr = run_ocr_on_pdf(client, pair.qp_path, settings)
    with _maybe_step(progress_logger, "ocr_ms", pair_id=pair_id, message="Azure OCR on marking scheme"):
        ms_ocr = run_ocr_on_pdf(client, pair.ms_path, settings)

    if parser_profile == "mcq":
        with _maybe_step(progress_logger, "parse_qp", pair_id=pair_id, message="Parse MCQ QP"):
            qp_items, qp_parser_meta, qp_anchor_candidates = extract_qp_items_mcq(
                qp_ocr,
                expected_max_question=settings.expected_max_question,
            )
        with _maybe_step(progress_logger, "parse_ms", pair_id=pair_id, message="Parse MCQ MS"):
            ms_items, ms_parser_meta, ms_anchor_candidates = extract_ms_items_mcq(
                ms_ocr,
                expected_max_question=settings.expected_max_question,
            )
    else:
        with _maybe_step(progress_logger, "parse_qp", pair_id=pair_id, message="Parse structured QP"):
            qp_items, qp_parser_meta, qp_anchor_candidates = extract_items_structured(
                qp_ocr,
                item_text_field="question_text",
                allow_blank_rest=False,
                kind="qp",
            )
        with _maybe_step(progress_logger, "parse_ms", pair_id=pair_id, message="Parse structured MS"):
            ms_items, ms_parser_meta, ms_anchor_candidates = extract_ms_items_structured(
                ms_ocr,
            )

    qp_before_norm = list(qp_items)
    ms_before_norm = list(ms_items)

    with _maybe_step(progress_logger, "grok_normalize_qp", pair_id=pair_id, message="Grok normalize QP"):
        qp_items, qp_grok_issues, qp_grok_debug = maybe_normalize_with_grok(
            kind="qp",
            profile=parser_profile,
            items=qp_items,
            ocr_payload=qp_ocr,
            settings=settings,
            progress_logger=progress_logger,
            pair_id=pair_id,
        )
    with _maybe_step(progress_logger, "grok_normalize_ms", pair_id=pair_id, message="Grok normalize MS"):
        ms_items, ms_grok_issues, ms_grok_debug = maybe_normalize_with_grok(
            kind="ms",
            profile=parser_profile,
            items=ms_items,
            ocr_payload=ms_ocr,
            settings=settings,
            progress_logger=progress_logger,
            pair_id=pair_id,
        )
    issues.extend(qp_grok_issues)
    issues.extend(ms_grok_issues)

    # Deterministic dedupe by composite key before pairing.
    qp_items, qp_dedupe_conflicts = _dedupe_by_composite_key(
        qp_items,
        text_field="question_text",
    )
    ms_items, ms_dedupe_conflicts = _dedupe_by_composite_key(
        ms_items,
        text_field="marking_scheme",
    )
    if qp_dedupe_conflicts:
        qp_parser_meta.setdefault("conflicts", [])
        qp_parser_meta["conflicts"].extend(qp_dedupe_conflicts)
    if ms_dedupe_conflicts:
        ms_parser_meta.setdefault("conflicts", [])
        ms_parser_meta["conflicts"].extend(ms_dedupe_conflicts)

    qp_conf = _safe_float((qp_ocr.get("metadata") or {}).get("avg_confidence"), 0.0)
    ms_conf = _safe_float((ms_ocr.get("metadata") or {}).get("avg_confidence"), 0.0)

    with _maybe_step(progress_logger, "pairing_baseline", pair_id=pair_id, message="Baseline pairing + quality metrics"):
        pairing = pair_qp_ms(qp_items, ms_items, profile=parser_profile)
        issues.extend(pairing.get("issues") or [])
        quality_metrics = compute_quality_metrics(
            profile=parser_profile,
            qp_items=qp_items,
            ms_items=ms_items,
            pairing=pairing,
            expected_max_question=settings.expected_max_question,
            qp_parser_meta=qp_parser_meta,
            ms_parser_meta=ms_parser_meta,
        )

    with _maybe_step(progress_logger, "grok_pair_repair", pair_id=pair_id, message="Grok pair repair"):
        repair_result = maybe_repair_pair_with_grok(
            pair=pair,
            profile=parser_profile,
            qp_items=qp_items,
            ms_items=ms_items,
            qp_ocr_payload=qp_ocr,
            ms_ocr_payload=ms_ocr,
            baseline_pairing=pairing,
            baseline_quality_metrics=quality_metrics,
            settings=settings,
            progress_logger=progress_logger,
        )
    issues.extend(repair_result.get("issues") or [])
    qp_items = repair_result["qp_items"]
    ms_items = repair_result["ms_items"]
    pairing = repair_result["pairing"]
    quality_metrics = repair_result["quality_metrics"]
    issues.extend(pairing.get("issues") or [])
    pair_repair_debug = repair_result.get("debug") or {}
    if pair_repair_debug.get("applied"):
        qp_parser_meta["repair"] = {
            "applied": True,
            "method": pair_repair_debug.get("selected_candidate"),
            "trigger_reasons": pair_repair_debug.get("trigger_reasons", []),
        }
        ms_parser_meta["repair"] = {
            "applied": True,
            "method": pair_repair_debug.get("selected_candidate"),
            "trigger_reasons": pair_repair_debug.get("trigger_reasons", []),
        }
        # Deterministic conflicts may be resolved by repair; reflect final conflicts.
        if "qp_conflicts" in repair_result:
            qp_parser_meta["conflicts"] = repair_result.get("qp_conflicts", qp_parser_meta.get("conflicts", []))
        ms_parser_meta["conflicts"] = repair_result.get("ms_conflicts", [])
        quality_metrics = compute_quality_metrics(
            profile=parser_profile,
            qp_items=qp_items,
            ms_items=ms_items,
            pairing=pairing,
            expected_max_question=settings.expected_max_question,
            qp_parser_meta=qp_parser_meta,
            ms_parser_meta=ms_parser_meta,
        )

    with _maybe_step(progress_logger, "validation_status", pair_id=pair_id, message="Determine status + validation"):
        status, match_quality, status_reasons, validation = determine_status(
            qp_confidence=qp_conf,
            ms_confidence=ms_conf,
            matched_count=pairing["matched_count"],
            qp_count=pairing["qp_count"],
            ms_count=pairing["ms_count"],
            issues=issues,
            quality_metrics=quality_metrics,
            settings=settings,
        )
        issues.extend(status_reasons)
    if progress_logger and progress_logger.active:
        for reason in status_reasons:
            progress_logger.warn(pair_id, "validation_status", reason)
        quality_counts = quality_metrics.get("counts") or {}
        if _safe_int(quality_counts.get("missing_qp_questions"), 0) > 0:
            progress_logger.warn(
                pair_id,
                "validation_status",
                f"missing_qp_questions={quality_counts.get('missing_qp_questions')}",
            )
        if _safe_int(quality_counts.get("missing_ms_questions"), 0) > 0:
            progress_logger.warn(
                pair_id,
                "validation_status",
                f"missing_ms_questions={quality_counts.get('missing_ms_questions')}",
            )
        if _safe_int(quality_counts.get("parser_conflicts"), 0) > 0:
            progress_logger.warn(
                pair_id,
                "validation_status",
                f"parser_conflicts={quality_counts.get('parser_conflicts')}",
            )

    variant_dir = pair.variant_dir
    qp_out = variant_dir / "qp_extracted.json"
    ms_out = variant_dir / "ms_extracted.json"
    summary_out = variant_dir / "pair_extraction_summary.json"

    qp_payload: Dict[str, Any] = {
        "pair_id": pair.pair_id,
        "source_pdf": str(pair.qp_path),
        "metadata": {
            "subject": pair.subject,
            "year": pair.year,
            "session": pair.session,
            "paper": pair.paper,
            "variant": pair.variant,
            "kind": "qp",
            "generated_at_utc": _utc_now(),
            "parser_profile": parser_profile,
        },
        "ocr_metadata": qp_ocr.get("metadata", {}),
        "questions": qp_items,
    }
    ms_payload: Dict[str, Any] = {
        "pair_id": pair.pair_id,
        "source_pdf": str(pair.ms_path),
        "metadata": {
            "subject": pair.subject,
            "year": pair.year,
            "session": pair.session,
            "paper": pair.paper,
            "variant": pair.variant,
            "kind": "ms",
            "generated_at_utc": _utc_now(),
            "parser_profile": parser_profile,
        },
        "ocr_metadata": ms_ocr.get("metadata", {}),
        "marking_entries": ms_items,
    }

    if write_raw_ocr:
        qp_payload["raw_ocr"] = {
            "full_text": qp_ocr.get("full_text", ""),
            "pages": qp_ocr.get("pages", []),
        }
        ms_payload["raw_ocr"] = {
            "full_text": ms_ocr.get("full_text", ""),
            "pages": ms_ocr.get("pages", []),
        }

    summary_payload = {
        "pair_id": pair.pair_id,
        "metadata": {
            "subject": pair.subject,
            "year": pair.year,
            "session": pair.session,
            "paper": pair.paper,
            "variant": pair.variant,
            "variant_dir": str(variant_dir),
            "generated_at_utc": _utc_now(),
            "parser_profile": parser_profile,
        },
        "status": status,
        "confidence": {
            "qp_avg": round(qp_conf, 4),
            "ms_avg": round(ms_conf, 4),
            "combined_avg": round(mean([qp_conf, ms_conf]), 4),
        },
        "counts": {
            "qp_items": pairing["qp_count"],
            "ms_items": pairing["ms_count"],
            "matched": pairing["matched_count"],
            "unmatched_qp": len(pairing["unmatched_qp"]),
            "unmatched_ms": len(pairing["unmatched_ms"]),
        },
        "match_quality": round(match_quality, 4),
        "issues": sorted(set(issues)),
        "validation": validation,
        "quality_metrics": quality_metrics,
        "parser_diagnostics": {
            "qp": qp_parser_meta,
            "ms": ms_parser_meta,
        },
        "unmatched_qp": pairing["unmatched_qp"],
        "unmatched_ms": pairing["unmatched_ms"],
        "matched_pairs": pairing["matches"],
        "timing": {
            "total_seconds": round(time.perf_counter() - start, 3),
        },
    }

    with _maybe_step(progress_logger, "write_outputs", pair_id=pair_id, message="Write per-variant outputs"):
        write_json(qp_out, qp_payload)
        write_json(ms_out, ms_payload)
        write_json(summary_out, summary_payload)

    normalization_debug = {
        "profile": parser_profile,
        "qp": {
            "before_count": len(qp_before_norm),
            "after_count": len(qp_items),
            "before_preview": qp_before_norm[:80],
            "after_preview": qp_items[:80],
        },
        "ms": {
            "before_count": len(ms_before_norm),
            "after_count": len(ms_items),
            "before_preview": ms_before_norm[:80],
            "after_preview": ms_items[:80],
        },
        "pair_repair": pair_repair_debug,
    }
    if settings.debug_level == "full":
        normalization_debug["qp"]["before_full"] = qp_before_norm
        normalization_debug["ms"]["before_full"] = ms_before_norm

    pairing_debug = {
        "profile": parser_profile,
        "attempts": pairing.get("attempts", []),
        "issues": pairing.get("issues", []),
        "unmatched_qp": pairing.get("unmatched_qp", []),
        "unmatched_ms": pairing.get("unmatched_ms", []),
    }
    if settings.debug_level == "basic":
        pairing_debug["attempts"] = pairing_debug["attempts"][:500]

    validation_debug = {
        "status": status,
        "validation": validation,
        "quality_metrics": quality_metrics,
        "status_reasons": status_reasons,
        "thresholds": {
            "review_conf_threshold": settings.review_conf_threshold,
            "review_match_threshold": settings.review_match_threshold,
            "expected_max_question": settings.expected_max_question,
        },
    }

    grok_debug = {
        "qp": qp_grok_debug,
        "ms": ms_grok_debug,
        "pair_repair": pair_repair_debug,
    }

    pair_debug_dir = write_pair_debug_files(
        settings=settings,
        pair=pair,
        qp_ocr_payload=qp_ocr,
        ms_ocr_payload=ms_ocr,
        qp_page_filters=qp_parser_meta.get("page_filters", []),
        anchor_candidates_qp=qp_anchor_candidates,
        anchor_candidates_ms=ms_anchor_candidates,
        normalization_debug=normalization_debug,
        pairing_debug=pairing_debug,
        validation_debug=validation_debug,
        grok_debug=grok_debug,
    )

    if pair_debug_dir:
        summary_payload["debug"] = {
            "enabled": True,
            "level": settings.debug_level,
            "pair_debug_dir": str(pair_debug_dir),
            "run_id": settings.debug_run_id,
        }
        write_json(summary_out, summary_payload)

    return summary_payload


def build_review_queue(run_report: Dict[str, Any]) -> Dict[str, Any]:
    pairs = run_report.get("pairs") or {}
    queue: List[Dict[str, Any]] = []
    for pair_id, item in pairs.items():
        if item.get("status") not in {"review_required", "failed"}:
            continue
        queue.append(
            {
                "pair_id": pair_id,
                "status": item.get("status"),
                "subject": item.get("subject"),
                "year": item.get("year"),
                "session": item.get("session"),
                "paper": item.get("paper"),
                "variant": item.get("variant"),
                "variant_dir": item.get("variant_dir"),
                "issues": item.get("issues", []),
                "updated_at_utc": item.get("updated_at_utc"),
            }
        )

    queue.sort(
        key=lambda x: (
            x.get("subject") or "",
            int(x.get("year") or 0),
            x.get("session") or "",
            x.get("paper") or "",
            x.get("variant") or "",
        )
    )
    return {
        "generated_at_utc": _utc_now(),
        "count": len(queue),
        "items": queue,
    }


def summarize_statuses(run_report: Dict[str, Any]) -> Dict[str, int]:
    pairs = run_report.get("pairs") or {}
    counters: Dict[str, int] = {
        "accepted": 0,
        "review_required": 0,
        "failed": 0,
    }
    for item in pairs.values():
        status = item.get("status")
        if status in counters:
            counters[status] += 1
    counters["total"] = sum(counters.values())
    return counters


def _pretty_pair(pair: PairMeta) -> str:
    return (
        f"{pair.subject} | {pair.year} | {pair.session} | "
        f"{pair.paper} | {pair.variant}"
    )


def _write_run_debug_selection(
    *,
    settings: PipelineSettings,
    args: argparse.Namespace,
    all_pairs: Sequence[PairMeta],
    filtered: Sequence[PairMeta],
    selected: Sequence[PairMeta],
) -> None:
    if not settings.debug_enabled:
        return
    root = settings.debug_dir / settings.debug_run_id
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_utc": _utc_now(),
        "run_id": settings.debug_run_id,
        "filters": {
            "subject": args.subject,
            "year": args.year,
            "session": args.session,
            "paper": args.paper,
            "variant": args.variant,
            "limit": args.limit,
            "resume": bool(args.resume),
            "force": bool(args.force),
            "profile": settings.parser_profile,
            "expected_max_question": settings.expected_max_question,
            "use_grok_normalization": settings.use_grok_normalization,
            "debug_enabled": settings.debug_enabled,
            "debug_level": settings.debug_level,
        },
        "counts": {
            "all_pairs": len(all_pairs),
            "filtered_pairs": len(filtered),
            "selected_pairs": len(selected),
        },
        "selected_pairs": [
            {
                "pair_id": pair.pair_id,
                "subject": pair.subject,
                "year": pair.year,
                "session": pair.session,
                "paper": pair.paper,
                "variant": pair.variant,
            }
            for pair in selected
        ],
    }
    write_json(root / "run_selection_debug.json", payload)


def _write_run_debug_summary(
    *,
    settings: PipelineSettings,
    run_report: Dict[str, Any],
) -> None:
    if not settings.debug_enabled:
        return
    root = settings.debug_dir / settings.debug_run_id
    root.mkdir(parents=True, exist_ok=True)

    pairs = run_report.get("pairs") or {}
    issue_counter: Counter[str] = Counter()
    for item in pairs.values():
        for issue in item.get("issues") or []:
            issue_counter[issue] += 1

    payload = {
        "generated_at_utc": _utc_now(),
        "run_id": settings.debug_run_id,
        "summary": run_report.get("summary") or {},
        "top_issues": issue_counter.most_common(100),
    }
    write_json(root / "run_summary_debug.json", payload)


def main() -> None:
    args = parse_args()
    settings = load_settings(args)
    progress_logger = RunProgressLogger(
        enabled=settings.progress_log_enabled,
        log_path=settings.progress_log_path,
        run_id=settings.debug_run_id,
    )
    progress_logger.write_header(
        {
            "run_id": settings.debug_run_id,
            "index_path": args.index_path,
            "filters": {
                "subject": args.subject,
                "year": args.year,
                "session": args.session,
                "paper": args.paper,
                "variant": args.variant,
                "limit": args.limit,
            },
            "resume": bool(args.resume),
            "force": bool(args.force),
            "profile": settings.parser_profile,
            "expected_max_question": settings.expected_max_question,
            "grok_model": settings.grok_model,
            "grok_max_retries": settings.grok_max_retries,
            "grok_timeout_normalize": settings.grok_timeout_normalize,
            "grok_timeout_repair": settings.grok_timeout_repair,
            "debug_enabled": settings.debug_enabled,
            "debug_level": settings.debug_level,
            "progress_log_path": str(settings.progress_log_path),
        }
    )

    run_report: Dict[str, Any] = {"summary": {}, "pairs": {}}
    run_summary_for_log: Dict[str, Any] = {}
    all_pairs: List[PairMeta] = []
    filtered: List[PairMeta] = []
    selected: List[PairMeta] = []

    try:
        with _maybe_step(progress_logger, "run_init", message="Initialize run paths"):
            index_path = Path(args.index_path)
            run_report_path = Path(args.run_report_path)
            review_queue_path = Path(args.review_queue_path)

        with _maybe_step(progress_logger, "pair_selection", message="Load index + apply filters + resume selection"):
            all_pairs = read_index(index_path)
            filtered = apply_filters(
                all_pairs,
                subjects=args.subject,
                year=args.year,
                session=args.session,
                paper=args.paper,
                variant=args.variant,
                limit=args.limit,
            )
            run_report = load_run_report(run_report_path)
            selected = select_for_resume(
                filtered,
                run_report,
                resume=args.resume,
                force=args.force,
            )

        print(
            f"Selected pairs: {len(selected)} "
            f"(from {len(filtered)} filtered / {len(all_pairs)} total valid pairs)"
        )
        progress_logger.info(
            None,
            "pair_selection",
            f"selected={len(selected)} filtered={len(filtered)} total={len(all_pairs)}",
        )

        _write_run_debug_selection(
            settings=settings,
            args=args,
            all_pairs=all_pairs,
            filtered=filtered,
            selected=selected,
        )

        if args.dry_run:
            for pair in selected[:50]:
                print(_pretty_pair(pair))
            print("Dry-run complete.")
            _write_run_debug_summary(settings=settings, run_report=run_report)
            run_summary_for_log = {
                "mode": "dry_run",
                "selected_pairs": len(selected),
                "summary": run_report.get("summary") or {},
            }
            return

        if not selected:
            print("No pairs to process.")
            review_queue = build_review_queue(run_report)
            write_json(review_queue_path, review_queue)
            run_report["summary"] = summarize_statuses(run_report)
            run_report["generated_at_utc"] = _utc_now()
            write_json(run_report_path, run_report)
            _write_run_debug_summary(settings=settings, run_report=run_report)
            run_summary_for_log = {
                "mode": "no_selection",
                "summary": run_report.get("summary") or {},
            }
            return

        with _maybe_step(progress_logger, "run_init", message="Initialize OCR client"):
            client = build_azure_client(settings)

        pairs_section = run_report.setdefault("pairs", {})
        run_report["config"] = {
            "index_path": str(index_path),
            "resume": bool(args.resume),
            "force": bool(args.force),
            "use_grok_normalization": bool(settings.use_grok_normalization),
            "write_raw_ocr": not bool(args.no_raw_ocr),
            "parser_profile": settings.parser_profile,
            "expected_max_question": settings.expected_max_question,
            "ocr_grok_model": settings.grok_model,
            "grok_max_retries": settings.grok_max_retries,
            "grok_timeout_normalize": settings.grok_timeout_normalize,
            "grok_timeout_repair": settings.grok_timeout_repair,
            "debug": {
                "enabled": settings.debug_enabled,
                "debug_dir": str(settings.debug_dir),
                "debug_level": settings.debug_level,
                "run_id": settings.debug_run_id,
            },
            "progress_log": {
                "enabled": settings.progress_log_enabled,
                "path": str(settings.progress_log_path),
            },
        }

        for idx, pair in enumerate(selected, start=1):
            pair_start = time.perf_counter()
            progress_logger.log("STEP_START", pair.pair_id, "pair_start", "Begin pair processing")
            print(f"[{idx}/{len(selected)}] {_pretty_pair(pair)}")
            try:
                summary = process_pair(
                    pair,
                    settings,
                    client,
                    write_raw_ocr=not args.no_raw_ocr,
                    progress_logger=progress_logger,
                )
                pairs_section[pair.pair_id] = {
                    "status": summary["status"],
                    "subject": pair.subject,
                    "year": pair.year,
                    "session": pair.session,
                    "paper": pair.paper,
                    "variant": pair.variant,
                    "variant_dir": str(pair.variant_dir),
                    "match_quality": summary["match_quality"],
                    "confidence": summary["confidence"],
                    "issues": summary["issues"],
                    "parser_profile": summary["metadata"].get("parser_profile"),
                    "updated_at_utc": _utc_now(),
                }
                progress_logger.record_pair_status(summary["status"])
                print(
                    "  -> "
                    f"status={summary['status']} matched={summary['counts']['matched']}/"
                    f"{summary['counts']['qp_items']} conf={summary['confidence']['combined_avg']:.3f}"
                )
            except Exception as exc:  # noqa: BLE001
                err = {
                    "status": "failed",
                    "subject": pair.subject,
                    "year": pair.year,
                    "session": pair.session,
                    "paper": pair.paper,
                    "variant": pair.variant,
                    "variant_dir": str(pair.variant_dir),
                    "issues": [f"pipeline_exception:{exc}"],
                    "traceback": traceback.format_exc(),
                    "updated_at_utc": _utc_now(),
                }
                pairs_section[pair.pair_id] = err
                progress_logger.record_pair_status("failed")
                progress_logger.error(pair.pair_id, "pair_end", f"pipeline_exception:{exc}")
                print(f"  -> status=failed reason={exc}")
            finally:
                pair_elapsed = max(0.0, time.perf_counter() - pair_start)
                progress_logger.record_pair_timing(pair.pair_id, pair_elapsed)
                progress_logger.log(
                    "STEP_END",
                    pair.pair_id,
                    "pair_end",
                    "Pair processing complete",
                    duration=pair_elapsed,
                )

            run_report["summary"] = summarize_statuses(run_report)
            run_report["generated_at_utc"] = _utc_now()
            write_json(run_report_path, run_report)

            review_queue = build_review_queue(run_report)
            write_json(review_queue_path, review_queue)

        with _maybe_step(progress_logger, "run_finalize", message="Finalize run reports"):
            _write_run_debug_summary(settings=settings, run_report=run_report)

        print("Run complete.")
        print(f"Run report: {run_report_path}")
        print(f"Review queue: {review_queue_path}")
        print(f"Summary: {run_report.get('summary')}")
        run_summary_for_log = {
            "mode": "full_run",
            "summary": run_report.get("summary") or {},
            "selected_pairs": len(selected),
        }
    except Exception as exc:  # noqa: BLE001
        progress_logger.error(None, "run_finalize", f"unhandled_exception:{exc}")
        run_summary_for_log = {
            "mode": "aborted",
            "error": str(exc),
            "summary": run_report.get("summary") if isinstance(run_report, dict) else {},
        }
        raise
    finally:
        progress_logger.write_final_report(run_summary_for_log)
        progress_logger.close()


if __name__ == "__main__":
    main()
