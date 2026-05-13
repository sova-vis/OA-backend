from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any

import httpx

from .config import Settings
from .types import (
    ConfigurationError,
    LineOCR,
    OCRCandidate,
    OCREngine,
    UncertainSpan,
)


class AzureDocumentIntelligenceClient:
    def __init__(self, settings: Settings, client: httpx.Client | None = None) -> None:
        self.settings = settings
        self._client = client or httpx.Client(timeout=settings.timeout_seconds)

    @property
    def is_available(self) -> bool:
        return bool(self.settings.azure_endpoint and self.settings.azure_api_key)

    def close(self) -> None:
        self._client.close()

    def analyze_path(self, source_path: Path, *, variant_name: str = "original") -> OCRCandidate:
        if not self.is_available:
            raise ConfigurationError(
                "Azure fallback OCR is enabled but Azure Document Intelligence credentials are missing."
            )

        endpoint = self.settings.azure_endpoint.rstrip("/")
        url = (
            f"{endpoint}/documentintelligence/documentModels/prebuilt-read:analyze"
            f"?api-version={self.settings.azure_api_version}"
        )
        headers = {
            "Ocp-Apim-Subscription-Key": str(self.settings.azure_api_key),
            "Content-Type": "application/json",
        }
        payload = {"base64Source": base64.b64encode(source_path.read_bytes()).decode("ascii")}
        initial = self._client.post(url, headers=headers, json=payload)
        initial.raise_for_status()

        operation_location = initial.headers.get("operation-location")
        if not operation_location:
            raise ConfigurationError("Azure Document Intelligence did not return an operation-location header.")

        result_json = self._poll(operation_location, headers)
        return self._to_candidate(result_json, variant_name=variant_name)

    def _poll(self, operation_location: str, headers: dict[str, str]) -> dict[str, Any]:
        while True:
            response = self._client.get(operation_location, headers=headers)
            response.raise_for_status()
            payload = response.json()
            status = str(payload.get("status", "")).lower()
            if status == "succeeded":
                return payload
            if status in {"failed", "partiallyfailed", "canceled"}:
                raise RuntimeError(f"Azure Document Intelligence analyze operation failed with status `{status}`.")
            time.sleep(self.settings.azure_poll_interval_seconds)

    def _to_candidate(self, payload: dict[str, Any], *, variant_name: str) -> OCRCandidate:
        analyze_result = payload.get("analyzeResult", {})
        content = str(analyze_result.get("content", "")).strip()
        pages = analyze_result.get("pages", []) or []

        lines: list[LineOCR] = []
        low_confidence_words: list[UncertainSpan] = []
        confidences: list[float] = []

        for page in pages:
            page_number = int(page.get("pageNumber", 1))
            for index, line in enumerate(page.get("lines", []) or [], start=1):
                text = str(line.get("content", "")).strip()
                if not text:
                    continue
                lines.append(
                    LineOCR(
                        page_number=page_number,
                        line_index=index,
                        text=text,
                        confidence=None,
                    )
                )

            for word in page.get("words", []) or []:
                confidence = word.get("confidence")
                if isinstance(confidence, (int, float)):
                    confidences.append(float(confidence))
                    if float(confidence) < 0.78:
                        low_confidence_words.append(
                            UncertainSpan(
                                page_number=page_number,
                                text=str(word.get("content", "")),
                                reason="Azure low-confidence word",
                                confidence=float(confidence),
                            )
                        )

        if not content and lines:
            page_map: dict[int, list[str]] = {}
            for line in lines:
                page_map.setdefault(line.page_number, []).append(line.text)
            content = "\n\n[[PAGE_BREAK]]\n\n".join(
                "\n".join(page_map[page_number]) for page_number in sorted(page_map)
            )

        ocr_confidence = round(sum(confidences) / len(confidences), 4) if confidences else 0.85
        return OCRCandidate(
            engine=OCREngine.AZURE,
            variant=variant_name,
            full_text=content,
            lines=lines,
            ocr_confidence=ocr_confidence,
            uncertain_spans=low_confidence_words,
        )
