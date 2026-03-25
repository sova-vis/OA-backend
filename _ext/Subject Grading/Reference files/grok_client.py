# grok_client.py
#
# Grok API HTTP client: call chat completions, retry, parse/repair JSON.
# Used by grade_pdf_answer for all 5 Grok calls (section detection, grading,
# refined rubric, page-wise suggestions, mark deduction analysis).
#
# Kept in grade_pdf_answer: payload-building and prompt logic.

from __future__ import annotations

import datetime
import json
import re
import time
from typing import Any, Dict, Optional, Tuple

import requests


# -----------------------------
# CONSTANTS
# -----------------------------
GROK_CHAT_URL = "https://api.x.ai/v1/chat/completions"
GROK_MODEL = "grok-4-1-fast-reasoning"
GROK_REQUEST_TIMEOUT = 150


# -----------------------------
# JSON CLEAN / REPAIR (Grok-specific)
# -----------------------------


def _clean_json_from_llm(text: str) -> str:
    """
    Remove markdown code fences and extract JSON from LLM responses.
    Handles various formats: ```json ... ```, ``` ... ```, or plain JSON.
    """
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()

    json_match = re.search(r"(\{[\s\S]*\})", text)
    if json_match:
        text = json_match.group(1)

    return text.strip()


def _repair_json(text: str, error_pos: Optional[int] = None) -> str:
    """
    Attempt to repair common JSON issues:
    - Trailing commas before closing brackets/braces
    - Control characters
    - Unclosed strings (basic handling)
    """
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    text = "".join(char for char in text if ord(char) >= 32 or char in "\n\t\r")

    if error_pos is not None and error_pos < len(text):
        if error_pos > 0:
            pass  # heuristic kept minimal; trailing comma removal is the main fix

    return text


# -----------------------------
# API CLIENT
# -----------------------------


class GrokAPIError(RuntimeError):
    """
    Raised when call_grok_api exhausts retries.
    token_usage / raw_content from last response when available.
    """

    def __init__(
        self,
        message: str,
        token_usage: Optional[Dict[str, int]] = None,
        raw_content: Optional[str] = None,
    ):
        super().__init__(message)
        self.token_usage = token_usage or {}
        self.raw_content = raw_content


def call_grok_api(
    grok_api_key: str,
    payload: Dict[str, Any],
    *,
    max_retries: int = 3,
    timeout: int = GROK_REQUEST_TIMEOUT,
    retry_backoff: bool = False,
    use_repair: bool = True,
    error_file_prefix: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    POST to Grok chat completions with retries, JSON parse/repair, and
    finish_reason=='length' handling.

    Returns (parsed_json, token_usage). Raises GrokAPIError on final failure.
    Callers that need a fallback should catch GrokAPIError and use
    .token_usage / .raw_content.
    """
    headers = {
        "Authorization": f"Bearer {grok_api_key}",
        "Content-Type": "application/json",
    }

    for attempt in range(max_retries):
        if attempt > 0 and retry_backoff:
            time.sleep(2 ** (attempt - 1))

        try:
            resp = requests.post(
                GROK_CHAT_URL,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
        except (requests.Timeout, requests.ConnectionError) as e:
            if attempt < max_retries - 1:
                print("Network error, retrying...")
                continue
            raise GrokAPIError(f"Grok API network error: {e}") from e

        if resp.status_code >= 300:
            if attempt < max_retries - 1:
                print(f"API error {resp.status_code}, retrying...")
                continue
            raise GrokAPIError(f"Grok API error {resp.status_code}: {resp.text}")

        try:
            data = resp.json()
        except Exception as exc:
            if attempt < max_retries - 1:
                print("Response JSON parse error, retrying...")
                continue
            raise GrokAPIError(f"Grok API response JSON parse error: {exc}") from exc

        usage = data.get("usage", {})
        token_usage = {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        }

        try:
            content = data["choices"][0]["message"]["content"]
        except Exception as exc:
            if attempt < max_retries - 1:
                print("Missing content in response, retrying...")
                continue
            raise GrokAPIError(
                f"Unexpected Grok API response structure: {data}",
                token_usage=token_usage,
            ) from exc

        finish_reason = data.get("choices", [{}])[0].get("finish_reason", "unknown")
        if finish_reason == "length":
            if attempt < max_retries - 1:
                payload["max_tokens"] = payload.get("max_tokens", 4000) + 2000
                print(
                    f"WARNING: Response truncated (finish_reason=length). "
                    f"Increasing max_tokens to {payload['max_tokens']} and retrying..."
                )
                continue
            raise GrokAPIError(
                "Grok response truncated due to max_tokens and no retries left",
                token_usage=token_usage,
                raw_content=content,
            )

        try:
            cleaned = _clean_json_from_llm(content)
            parsed = json.loads(cleaned)
            if attempt > 0:
                print(f"Successfully parsed JSON on attempt {attempt + 1}")
            return parsed, token_usage
        except json.JSONDecodeError as exc:
            if not use_repair:
                if attempt < max_retries - 1:
                    print("JSON parse error, retrying...")
                    continue
                raise GrokAPIError(
                    f"Grok API JSON parse error: {exc.msg} at position {exc.pos}",
                    token_usage=token_usage,
                    raw_content=content,
                ) from exc

            repaired: Optional[str] = None
            try:
                repaired = _repair_json(cleaned, error_pos=exc.pos)
                parsed = json.loads(repaired)
                if attempt > 0:
                    print(f"Successfully parsed JSON after repair on attempt {attempt + 1}")
                return parsed, token_usage
            except (json.JSONDecodeError, Exception) as repair_exc:
                if error_file_prefix:
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    error_file = f"{error_file_prefix}_{ts}_{attempt + 1}.txt"
                    rep_str = repaired if repaired is not None else "(repair raised or N/A)"
                    with open(error_file, "w", encoding="utf-8") as f:
                        f.write(f"=== FULL RESPONSE (length: {len(content)} chars) ===\n")
                        f.write(content)
                        f.write(f"\n\n=== CLEANED (length: {len(cleaned)} chars) ===\n")
                        f.write(cleaned)
                        f.write(f"\n\n=== REPAIRED (length: {len(rep_str)} chars) ===\n")
                        f.write(rep_str)
                        f.write(f"\n\n=== ORIGINAL ERROR ===\n{exc}\n")
                        f.write(f"\n=== REPAIR ERROR ===\n{repair_exc}\n")
                    print(f"DEBUG: Saved malformed response to {error_file}")

                if attempt < max_retries - 1:
                    print("Malformed JSON, retrying...")
                    continue
                raise GrokAPIError(
                    f"Grok API malformed JSON after {max_retries} attempts. "
                    f"Error: {exc.msg} at {exc.pos}. Repair failed: {repair_exc}.",
                    token_usage=token_usage,
                    raw_content=content,
                ) from exc

    raise GrokAPIError(f"Failed to get valid response after {max_retries} attempts")
