from __future__ import annotations

import json
import time
from typing import Any, Sequence

import httpx

from .arbitration import render_disagreement_report, render_indexed_lines
from .config import Settings
from .prompts import (
    ocr_system_prompt,
    ocr_user_prompt,
    refine_math_answer_system_prompt,
    refine_math_answer_user_prompt,
    repair_system_prompt,
    repair_user_prompt,
    split_classification_system_prompt,
    split_classification_user_prompt,
    split_retry_system_prompt,
    split_retry_user_prompt,
)
from .types import (
    ConfigurationError,
    DocumentPage,
    GrokAPIError,
    LineOCR,
    MathAnswerRefineResult,
    OCRCandidate,
    OCREngine,
    RepairAction,
    RepairDecisionSet,
    SplitRetryResult,
    StructuredExtraction,
    StructuredOCRResult,
    UncertainSpan,
)


class GrokClient:
    def __init__(self, settings: Settings, client: httpx.Client | None = None) -> None:
        if not settings.api_key:
            raise ConfigurationError("Missing Grok API key. Set Grok_API or XAI_API_KEY.")
        self.settings = settings
        self._client = client or httpx.Client(timeout=settings.timeout_seconds)

    def close(self) -> None:
        self._client.close()

    def ocr_pages(self, pages: Sequence[DocumentPage], *, variant_name: str) -> OCRCandidate:
        payload = {
            "model": self.settings.model,
            "store": False,
            "input": [
                {"role": "system", "content": ocr_system_prompt()},
                {
                    "role": "user",
                    "content": [
                        *self._image_inputs(pages),
                        {
                            "type": "input_text",
                            "text": ocr_user_prompt(len(pages), variant_name),
                        },
                    ],
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "structured_ocr_result",
                    "schema": StructuredOCRResult.model_json_schema(),
                    "strict": True,
                }
            },
        }
        response_json = self._post_with_retries(payload)
        output_text = self._extract_output_text(response_json)
        if not output_text.strip():
            raise GrokAPIError("Grok returned an empty structured OCR response.")

        try:
            result = StructuredOCRResult.model_validate_json(output_text)
        except Exception as exc:  # pragma: no cover
            raise GrokAPIError(f"Failed to parse Grok OCR output: {exc}") from exc

        lines: list[LineOCR] = []
        uncertain_spans: list[UncertainSpan] = []
        for page in result.pages:
            page_lines = page.lines or [line for line in page.full_text.splitlines() if line.strip()]
            for index, text in enumerate(page_lines, start=1):
                lines.append(
                    LineOCR(
                        page_number=page.page_number,
                        line_index=index,
                        text=text,
                        confidence=page.ocr_confidence,
                    )
                )
            for span in page.uncertain_spans:
                uncertain_spans.append(
                    UncertainSpan(
                        page_number=page.page_number,
                        text=span,
                        reason="Grok uncertain span",
                    )
                )

        full_text = result.full_text.strip() or self._rebuild_full_text(lines)
        return OCRCandidate(
            engine=OCREngine.GROK,
            variant=variant_name,
            full_text=full_text,
            lines=lines,
            ocr_confidence=result.ocr_confidence,
            uncertain_spans=uncertain_spans,
        )

    def split_and_classify(
        self,
        pages: Sequence[DocumentPage],
        candidate: OCRCandidate,
    ) -> StructuredExtraction:
        payload = {
            "model": self.settings.model,
            "store": False,
            "input": [
                {"role": "system", "content": split_classification_system_prompt()},
                {
                    "role": "user",
                    "content": [
                        *self._image_inputs(pages),
                        {
                            "type": "input_text",
                            "text": split_classification_user_prompt(
                                candidate.full_text,
                                render_indexed_lines(candidate),
                            ),
                        },
                    ],
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "qa_extraction_response",
                    "schema": StructuredExtraction.model_json_schema(),
                    "strict": True,
                }
            },
        }
        response_json = self._post_with_retries(payload)
        output_text = self._extract_output_text(response_json)
        if not output_text.strip():
            raise GrokAPIError("Grok returned an empty structured extraction response.")
        try:
            return StructuredExtraction.model_validate_json(output_text)
        except Exception as exc:  # pragma: no cover
            raise GrokAPIError(f"Failed to parse Grok structured output: {exc}") from exc

    def refine_math_answer(
        self,
        pages: Sequence[DocumentPage],
        *,
        question_raw: str,
        answer_raw: str,
    ) -> MathAnswerRefineResult:
        payload = {
            "model": self.settings.model,
            "store": False,
            "input": [
                {"role": "system", "content": refine_math_answer_system_prompt()},
                {
                    "role": "user",
                    "content": [
                        *self._image_inputs(pages),
                        {
                            "type": "input_text",
                            "text": refine_math_answer_user_prompt(question_raw, answer_raw),
                        },
                    ],
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "math_answer_refine_response",
                    "schema": MathAnswerRefineResult.model_json_schema(),
                    "strict": True,
                }
            },
        }
        response_json = self._post_with_retries(payload)
        output_text = self._extract_output_text(response_json)
        if not output_text.strip():
            raise GrokAPIError("Grok returned an empty math answer refine response.")
        try:
            return MathAnswerRefineResult.model_validate_json(output_text)
        except Exception as exc:  # pragma: no cover
            raise GrokAPIError(f"Failed to parse Grok math answer refine output: {exc}") from exc

    def retry_split(
        self,
        pages: Sequence[DocumentPage],
        candidate: OCRCandidate,
    ) -> SplitRetryResult:
        payload = {
            "model": self.settings.model,
            "store": False,
            "input": [
                {"role": "system", "content": split_retry_system_prompt()},
                {
                    "role": "user",
                    "content": [
                        *self._image_inputs(pages),
                        {
                            "type": "input_text",
                            "text": split_retry_user_prompt(render_indexed_lines(candidate)),
                        },
                    ],
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "split_retry_response",
                    "schema": SplitRetryResult.model_json_schema(),
                    "strict": True,
                }
            },
        }
        response_json = self._post_with_retries(payload)
        output_text = self._extract_output_text(response_json)
        if not output_text.strip():
            raise GrokAPIError("Grok returned an empty split retry response.")
        try:
            return SplitRetryResult.model_validate_json(output_text)
        except Exception as exc:  # pragma: no cover
            raise GrokAPIError(f"Failed to parse Grok split retry output: {exc}") from exc

    def repair_disagreements(
        self,
        pages: Sequence[DocumentPage],
        candidate: OCRCandidate,
        disagreements,
    ) -> list[RepairAction]:
        payload = {
            "model": self.settings.model,
            "store": False,
            "input": [
                {"role": "system", "content": repair_system_prompt()},
                {
                    "role": "user",
                    "content": [
                        *self._image_inputs(pages),
                        {
                            "type": "input_text",
                            "text": repair_user_prompt(
                                candidate.full_text,
                                render_disagreement_report(disagreements),
                            ),
                        },
                    ],
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "repair_response",
                    "schema": RepairDecisionSet.model_json_schema(),
                    "strict": True,
                }
            },
        }
        response_json = self._post_with_retries(payload)
        output_text = self._extract_output_text(response_json)
        if not output_text.strip():
            return []

        try:
            decisions = RepairDecisionSet.model_validate_json(output_text)
        except Exception as exc:  # pragma: no cover
            raise GrokAPIError(f"Failed to parse Grok repair output: {exc}") from exc

        line_lookup = {(line.page_number, line.line_index): line.text for line in candidate.lines}
        actions: list[RepairAction] = []
        for decision in decisions.actions:
            before_text = line_lookup.get((decision.page_number, decision.line_index), "")
            if not before_text:
                continue
            actions.append(
                RepairAction(
                    page_number=decision.page_number,
                    line_index=decision.line_index,
                    before_text=before_text,
                    after_text=decision.repaired_text,
                    source="grok_repair",
                    accepted=decision.confidence >= self.settings.repair_confidence_threshold
                    and decision.repaired_text.strip()
                    and decision.repaired_text.strip() != before_text.strip(),
                    confidence=decision.confidence,
                    rationale=decision.rationale,
                )
            )
        return actions

    def _image_inputs(self, pages: Sequence[DocumentPage]) -> list[dict[str, Any]]:
        return [
            {
                "type": "input_image",
                "image_url": page.to_data_url(),
                "detail": self.settings.image_detail,
            }
            for page in pages
        ]

    def _post_with_retries(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.settings.base_url}/responses"
        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
        }

        attempts = self.settings.max_retries + 1
        last_error: GrokAPIError | None = None
        for attempt in range(1, attempts + 1):
            try:
                response = self._client.post(url, headers=headers, json=payload)
            except httpx.HTTPError as exc:
                last_error = GrokAPIError(f"Network error while calling Grok: {exc}")
            else:
                if response.status_code < 400:
                    return response.json()

                message = self._extract_error_message(response)
                last_error = GrokAPIError(
                    f"Grok request failed with status {response.status_code}: {message}",
                    status_code=response.status_code,
                )
                if response.status_code < 500 and response.status_code not in {408, 429}:
                    raise last_error

            if attempt < attempts:
                time.sleep(min(2 ** (attempt - 1), 4))

        assert last_error is not None
        raise last_error

    @staticmethod
    def _extract_error_message(response: httpx.Response) -> str:
        try:
            body = response.json()
        except json.JSONDecodeError:
            return response.text.strip() or "Unknown error"

        error = body.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error)
        return str(error or body)

    @staticmethod
    def _extract_output_text(response_json: dict[str, Any]) -> str:
        direct = response_json.get("output_text")
        if isinstance(direct, str) and direct.strip():
            return direct

        fragments: list[str] = []
        for item in response_json.get("output", []):
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if not isinstance(content, dict):
                    continue
                if content.get("type") == "output_text":
                    text = content.get("text")
                    if isinstance(text, str):
                        fragments.append(text)
                elif content.get("type") == "text":
                    text = content.get("value")
                    if isinstance(text, str):
                        fragments.append(text)

        if fragments:
            return "\n".join(fragment for fragment in fragments if fragment)

        raise GrokAPIError("Unable to locate text output in Grok response.")

    @staticmethod
    def _rebuild_full_text(lines: list[LineOCR]) -> str:
        pages: dict[int, list[str]] = {}
        for line in lines:
            pages.setdefault(line.page_number, []).append(line.text)
        return "\n\n[[PAGE_BREAK]]\n\n".join(
            "\n".join(pages[page_number]) for page_number in sorted(pages)
        )
