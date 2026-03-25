"""Persist per-request debug artifacts locally."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Tuple


def _safe_filename(text: str) -> str:
    cleaned = "".join(ch for ch in str(text or "") if ch.isalnum() or ch in {"-", "_", "."})
    cleaned = cleaned.strip("._-")
    return cleaned[:80] or "upload"


def save_debug_run(payload: Dict[str, Any], *, root: Path) -> Tuple[str, Path]:
    """Write a single JSON file and return (run_id, path)."""
    run_id = uuid.uuid4().hex
    ts = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    name_hint = _safe_filename(str(payload.get("filename") or "upload"))
    out_dir = Path(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{ts}_{name_hint}_{run_id}.json"

    body = dict(payload)
    body.setdefault("run_id", run_id)
    body.setdefault("saved_at_unix", int(time.time()))
    body.setdefault("saved_at_iso", time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()))
    body.setdefault("hostname", os.getenv("COMPUTERNAME") or os.getenv("HOSTNAME") or "")

    path.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_id, path

