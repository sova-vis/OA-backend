from __future__ import annotations

import json
from unittest.mock import patch

from oa_extraction.grok_client import GrokClient
from oa_extraction.types import DocumentPage, MathAnswerRefineResult

from test_pipeline import PNG_BYTES, _settings


def test_refine_math_answer_parses_strict_json_output() -> None:
    pages = [
        DocumentPage(
            page_number=1,
            mime_type="image/png",
            content_bytes=PNG_BYTES,
            source_name="t.png",
            variant_name="original",
        )
    ]
    settings = _settings(enable_azure_fallback=False)
    client = GrokClient(settings)

    payload_out = {
        "refined_answer": r"\frac{1}{2} = 0.5",
        "confidence": 0.9,
        "rationale": "fraction",
    }

    def fake_post(_self, _payload):
        return {"output_text": json.dumps(payload_out)}

    with patch.object(GrokClient, "_post_with_retries", fake_post):
        result = client.refine_math_answer(pages, question_raw="Q?", answer_raw="garbled line")

    assert isinstance(result, MathAnswerRefineResult)
    assert result.refined_answer == r"\frac{1}{2} = 0.5"
    assert result.confidence == 0.9
    client.close()
