from __future__ import annotations

import base64
import json
from pathlib import Path

import fitz

from oa_extraction.config import Settings
from oa_extraction.pipeline import ExtractionPipeline
from oa_extraction.types import (
    LineAssignment,
    LineOCR,
    LineTarget,
    MathAnswerRefineResult,
    OCRCandidate,
    OCREngine,
    RepairAction,
    SplitRetryResult,
    StructuredExtraction,
    SubjectLabel,
)

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR42mP8z/C/HwAFgwJ/l7sRKQAAAABJRU5ErkJggg=="
)


class FakeGrokClient:
    def __init__(
        self,
        *,
        ocr_by_variant: dict[str, OCRCandidate],
        split_results: list[StructuredExtraction],
        repair_actions: list[RepairAction] | None = None,
        split_retry_result: SplitRetryResult | None = None,
        math_refine_result: MathAnswerRefineResult | None = None,
    ) -> None:
        self.ocr_by_variant = ocr_by_variant
        self.split_results = list(split_results)
        self.repair_actions = repair_actions or []
        self.split_retry_result = split_retry_result or SplitRetryResult(assignments=[], split_confidence=0.0)
        self.math_refine_result = math_refine_result
        self.ocr_calls: list[str] = []
        self.split_calls: list[tuple[str, str]] = []
        self.retry_calls = 0
        self.repair_call_count = 0
        self.refine_math_calls = 0

    def ocr_pages(self, pages, *, variant_name: str):
        self.ocr_calls.append(variant_name)
        return self.ocr_by_variant[variant_name]

    def split_and_classify(self, pages, candidate: OCRCandidate):
        self.split_calls.append((str(candidate.engine), candidate.variant))
        if len(self.split_results) == 1:
            return self.split_results[0]
        return self.split_results.pop(0)

    def retry_split(self, pages, candidate: OCRCandidate):
        self.retry_calls += 1
        return self.split_retry_result

    def repair_disagreements(self, pages, candidate: OCRCandidate, disagreements):
        self.repair_call_count += 1
        return self.repair_actions

    def refine_math_answer(self, pages, *, question_raw: str, answer_raw: str):
        self.refine_math_calls += 1
        if self.math_refine_result is None:
            raise AssertionError("Unexpected refine_math_answer call")
        return self.math_refine_result

    def close(self):
        return None


class FakeAzureClient:
    def __init__(self, candidate: OCRCandidate | None = None, *, available: bool = True, error: Exception | None = None) -> None:
        self.candidate = candidate
        self.available = available
        self.error = error
        self.calls: list[str] = []

    @property
    def is_available(self) -> bool:
        return self.available

    def analyze_path(self, source_path: Path, *, variant_name: str = "original") -> OCRCandidate:
        self.calls.append(str(source_path))
        if self.error is not None:
            raise self.error
        assert self.candidate is not None
        return self.candidate

    def close(self):
        return None


def _settings(
    *,
    enable_azure_fallback: bool = True,
    enable_targeted_repair: bool = True,
    enable_math_answer_refine: bool = False,
    math_answer_refine_min_confidence: float = 0.85,
) -> Settings:
    return Settings(
        api_key="test-key",
        base_url="https://api.x.ai/v1",
        model="grok-test",
        timeout_seconds=30,
        max_retries=0,
        ocr_confidence_threshold=0.85,
        split_confidence_threshold=0.90,
        classification_confidence_threshold=0.80,
        azure_endpoint="https://example.cognitiveservices.azure.com",
        azure_api_key="azure-test-key",
        azure_api_version="2024-11-30",
        enable_azure_fallback=enable_azure_fallback,
        grok_fallback_ocr_threshold=0.90,
        grok_fallback_split_threshold=0.92,
        enable_image_variants=True,
        enable_targeted_repair=enable_targeted_repair,
        engine_disagreement_threshold=0.08,
        repair_confidence_threshold=0.85,
        selection_score_margin=0.05,
        enable_math_answer_refine=enable_math_answer_refine,
        math_answer_refine_min_confidence=math_answer_refine_min_confidence,
    )


def _make_candidate(
    *,
    engine: OCREngine,
    variant: str,
    full_text: str,
    ocr_confidence: float,
) -> OCRCandidate:
    lines: list[LineOCR] = []
    pages = [chunk.strip() for chunk in full_text.split("[[PAGE_BREAK]]")]
    for page_number, page_text in enumerate(pages, start=1):
        split_lines = [line.strip() for line in page_text.splitlines() if line.strip()]
        for line_index, line_text in enumerate(split_lines, start=1):
            lines.append(
                LineOCR(
                    page_number=page_number,
                    line_index=line_index,
                    text=line_text,
                    confidence=ocr_confidence,
                )
            )

    return OCRCandidate(
        engine=engine,
        variant=variant,
        full_text=full_text,
        lines=lines,
        ocr_confidence=ocr_confidence,
    )


def _make_structured(
    *,
    question_raw: str,
    answer_raw: str,
    subject: SubjectLabel,
    split_confidence: float,
    classification_confidence: float = 0.95,
    ocr_confidence: float = 0.95,
    whole_text_raw: str | None = None,
) -> StructuredExtraction:
    return StructuredExtraction(
        whole_text_raw=whole_text_raw or "\n".join(part for part in (question_raw, answer_raw) if part),
        question_raw=question_raw,
        answer_raw=answer_raw,
        subject=subject,
        ocr_confidence=ocr_confidence,
        split_confidence=split_confidence,
        classification_confidence=classification_confidence,
    )


def test_pipeline_uses_azure_fallback_when_grok_is_low_confidence(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    image_path.write_bytes(PNG_BYTES)

    grok_original = _make_candidate(
        engine=OCREngine.GROK,
        variant="original",
        full_text="Q. Solve log3 2x + log9 x = 12\nA. x = 6561",
        ocr_confidence=0.71,
    )
    grok_gray = _make_candidate(
        engine=OCREngine.GROK,
        variant="grayscale_autocontrast",
        full_text="Q. Solve log3 2x + log9 x = 12\nA. x = 6561",
        ocr_confidence=0.73,
    )
    grok_binary = _make_candidate(
        engine=OCREngine.GROK,
        variant="sharpened_binary",
        full_text="Q. Solve log3 2x + log9 x = 12\nA. x = 6561",
        ocr_confidence=0.69,
    )
    azure_candidate = _make_candidate(
        engine=OCREngine.AZURE,
        variant="original",
        full_text="Q. Solve log_3 x + log_9 x = 12\nA. x = 3^8\nx = 6561",
        ocr_confidence=0.97,
    )

    grok_client = FakeGrokClient(
        ocr_by_variant={
            "original": grok_original,
            "grayscale_autocontrast": grok_gray,
            "sharpened_binary": grok_binary,
        },
        split_results=[
            _make_structured(
                question_raw="Solve log3 2x + log9 x = 12",
                answer_raw="x = 6561",
                subject=SubjectLabel.MATH,
                split_confidence=0.74,
            ),
            _make_structured(
                question_raw="Solve log_3 x + log_9 x = 12",
                answer_raw="x = 3^8\nx = 6561",
                subject=SubjectLabel.MATH,
                split_confidence=0.97,
            ),
        ],
    )
    azure_client = FakeAzureClient(candidate=azure_candidate)

    result = ExtractionPipeline(
        settings=_settings(),
        grok_client=grok_client,
        azure_client=azure_client,
    ).extract(str(image_path))

    assert azure_client.calls == [str(image_path)]
    assert result.question_raw == "Solve log_3 x + log_9 x = 12"
    assert result.diagnostics is not None
    assert str(result.diagnostics.selected_ocr_engine) == "azure"
    assert result.diagnostics.selected_variant == "original"


def test_pipeline_triggers_azure_fallback_when_fraction_chain_ambiguity_flagged(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    image_path.write_bytes(PNG_BYTES)

    garbled_answer = (
        "Answer: log (9) = log (3^2) = 2log (3) log (4) log (2^2) 2log (2) = log (3) = log_2 (3) log (2)"
    )
    question_line = "Show that log_4 9 = log_2 3"
    full_grok = f"{question_line}\n{garbled_answer}"

    grok_original = _make_candidate(
        engine=OCREngine.GROK,
        variant="original",
        full_text=full_grok,
        ocr_confidence=0.95,
    )
    grok_gray = grok_original.model_copy(update={"variant": "grayscale_autocontrast"})
    grok_binary = grok_original.model_copy(update={"variant": "sharpened_binary"})

    azure_full = (
        f"{question_line}\n"
        r"Answer: \frac{\log 9}{\log 4} = \frac{\log 3}{\log 2} = \log_2 3"
    )
    azure_candidate = _make_candidate(
        engine=OCREngine.AZURE,
        variant="original",
        full_text=azure_full,
        ocr_confidence=0.97,
    )

    grok_client = FakeGrokClient(
        ocr_by_variant={
            "original": grok_original,
            "grayscale_autocontrast": grok_gray,
            "sharpened_binary": grok_binary,
        },
        split_results=[
            _make_structured(
                question_raw=question_line,
                answer_raw=garbled_answer,
                subject=SubjectLabel.MATH,
                split_confidence=0.95,
                whole_text_raw=full_grok,
            ),
            _make_structured(
                question_raw=question_line,
                answer_raw=r"\frac{\log 9}{\log 4} = \frac{\log 3}{\log 2} = \log_2 3",
                subject=SubjectLabel.MATH,
                split_confidence=0.96,
                whole_text_raw=azure_full,
            ),
        ],
    )
    azure_client = FakeAzureClient(candidate=azure_candidate)

    result = ExtractionPipeline(
        settings=_settings(),
        grok_client=grok_client,
        azure_client=azure_client,
    ).extract(str(image_path))

    assert azure_client.calls == [str(image_path)]
    assert result.diagnostics is not None


def test_pipeline_applies_split_retry_when_initial_split_is_weak(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    image_path.write_bytes(PNG_BYTES)

    candidate = _make_candidate(
        engine=OCREngine.GROK,
        variant="original",
        full_text="Q. State Newton's second law\nA. Force equals mass times acceleration",
        ocr_confidence=0.96,
    )
    grok_client = FakeGrokClient(
        ocr_by_variant={
            "original": candidate,
            "grayscale_autocontrast": candidate.model_copy(update={"variant": "grayscale_autocontrast"}),
            "sharpened_binary": candidate.model_copy(update={"variant": "sharpened_binary"}),
        },
        split_results=[
            _make_structured(
                question_raw="State Newton's second law",
                answer_raw="Force equals mass times acceleration",
                subject=SubjectLabel.PHYSICS,
                split_confidence=0.94,
            ),
            _make_structured(
                question_raw="Q. State Newton's second law\nA. Force equals mass times acceleration",
                answer_raw="",
                subject=SubjectLabel.PHYSICS,
                split_confidence=0.70,
            ),
        ],
        split_retry_result=SplitRetryResult(
            assignments=[
                LineAssignment(page_number=1, line_index=1, target=LineTarget.QUESTION),
                LineAssignment(page_number=1, line_index=2, target=LineTarget.ANSWER),
            ],
            split_confidence=0.98,
        ),
    )

    result = ExtractionPipeline(
        settings=_settings(enable_azure_fallback=False),
        grok_client=grok_client,
        azure_client=FakeAzureClient(available=False),
    ).extract(str(image_path))

    assert grok_client.retry_calls == 1
    assert result.question_raw == "Q. State Newton's second law"
    assert result.answer_raw == "A. Force equals mass times acceleration"
    assert result.confidence.split == 0.98
    assert result.diagnostics is not None
    assert result.diagnostics.split_retry_applied is True


def test_pipeline_rejects_repair_when_it_lowers_candidate_quality(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    image_path.write_bytes(PNG_BYTES)

    original = _make_candidate(
        engine=OCREngine.GROK,
        variant="original",
        full_text="Q. Solve log_3 x = 9\nA. x = 3^2",
        ocr_confidence=0.96,
    )
    gray = _make_candidate(
        engine=OCREngine.GROK,
        variant="grayscale_autocontrast",
        full_text="Q. Solve log3 x = 9\nA. x = 3^2",
        ocr_confidence=0.95,
    )
    binary = _make_candidate(
        engine=OCREngine.GROK,
        variant="sharpened_binary",
        full_text="Q. Solve log3 x = 9\nA. x = 3^2",
        ocr_confidence=0.94,
    )

    grok_client = FakeGrokClient(
        ocr_by_variant={
            "original": original,
            "grayscale_autocontrast": gray,
            "sharpened_binary": binary,
        },
        split_results=[
            _make_structured(
                question_raw="Solve log_3 x = 9",
                answer_raw="x = 3^2",
                subject=SubjectLabel.MATH,
                split_confidence=0.96,
            ),
            _make_structured(
                question_raw="Solve log_3 x = 9",
                answer_raw="x = 3^2",
                subject=SubjectLabel.MATH,
                split_confidence=0.96,
            ),
        ],
        repair_actions=[
            RepairAction(
                page_number=1,
                line_index=1,
                before_text="Q. Solve log_3 x = 9",
                after_text="Q Solve x 9",
                source="grok_repair",
                accepted=True,
                confidence=0.99,
                rationale="Bad repair for rejection test.",
            )
        ],
    )

    result = ExtractionPipeline(
        settings=_settings(enable_azure_fallback=False, enable_targeted_repair=True),
        grok_client=grok_client,
        azure_client=FakeAzureClient(available=False),
    ).extract(str(image_path))

    assert grok_client.repair_call_count == 1
    assert result.whole_text_raw.startswith("Q. Solve log_3 x = 9")
    assert result.diagnostics is not None
    assert any(action.accepted is False for action in result.diagnostics.repair_actions)


def test_pipeline_preserves_pdf_page_count_and_uses_pdf_fallback(tmp_path: Path) -> None:
    pdf_path = tmp_path / "input.pdf"
    document = fitz.open()
    try:
        first_page = document.new_page()
        first_page.insert_text((72, 72), "Question page")
        second_page = document.new_page()
        second_page.insert_text((72, 72), "Answer page")
        document.save(pdf_path)
    finally:
        document.close()

    grok_candidate = _make_candidate(
        engine=OCREngine.GROK,
        variant="original",
        full_text="Question page [[PAGE_BREAK]] Answer page",
        ocr_confidence=0.95,
    )
    azure_candidate = _make_candidate(
        engine=OCREngine.AZURE,
        variant="original",
        full_text="Question page [[PAGE_BREAK]] Answer page",
        ocr_confidence=0.98,
    )
    grok_client = FakeGrokClient(
        ocr_by_variant={
            "original": grok_candidate,
            "grayscale_autocontrast": grok_candidate.model_copy(update={"variant": "grayscale_autocontrast"}),
            "sharpened_binary": grok_candidate.model_copy(update={"variant": "sharpened_binary"}),
        },
        split_results=[
            _make_structured(
                question_raw="Question page",
                answer_raw="Answer page",
                subject=SubjectLabel.ENGLISH,
                split_confidence=0.95,
            ),
            _make_structured(
                question_raw="Question page",
                answer_raw="Answer page",
                subject=SubjectLabel.ENGLISH,
                split_confidence=0.98,
            ),
        ],
    )

    result = ExtractionPipeline(
        settings=_settings(),
        grok_client=grok_client,
        azure_client=FakeAzureClient(candidate=azure_candidate),
    ).extract(str(pdf_path))

    assert result.input_type == "pdf"
    assert result.page_count == 2


def test_fixture_oa1_recovers_expected_math_notation() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    image_path = repo_root / "OA-images" / "oa_1.png"
    expected = json.loads((repo_root / "tests" / "fixtures" / "oa_1_expected.json").read_text(encoding="utf-8"))

    grok_original = _make_candidate(
        engine=OCREngine.GROK,
        variant="original",
        full_text=(
            "Q. Solve log3 2x + log9 x = 12\n"
            "A. log3 2x + log3 x / 2 = 12\n"
            "x = 6561"
        ),
        ocr_confidence=0.79,
    )
    grok_gray = _make_candidate(
        engine=OCREngine.GROK,
        variant="grayscale_autocontrast",
        full_text=(
            "Q. Solve log3 x + log9 x = 12\n"
            "A. log3 x + log3 x / 2 = 12\n"
            "x = 3^8\n"
            "x = 6561"
        ),
        ocr_confidence=0.84,
    )
    grok_binary = _make_candidate(
        engine=OCREngine.GROK,
        variant="sharpened_binary",
        full_text=(
            "Q. Solve log3 x + log9 x = 12\n"
            "A. log3 x + log3 x / 2 = 12\n"
            "x = 3^8\n"
            "x = 6561"
        ),
        ocr_confidence=0.82,
    )
    azure_candidate = _make_candidate(
        engine=OCREngine.AZURE,
        variant="original",
        full_text=(
            "Q. Solve log_3 x + log_9 x = 12\n"
            "A. log_3 x + log_3 x / log_3 9 = 12\n"
            "log_3 x + log_3 x / 2 = 12\n"
            "2log_3 x + log_3 x = 24\n"
            "3log_3 x = 24\n"
            "log_3 x = 24 / 3\n"
            "log_3 x = 8\n"
            "x = 3^8\n"
            "x = 6561"
        ),
        ocr_confidence=0.97,
    )
    grok_client = FakeGrokClient(
        ocr_by_variant={
            "original": grok_original,
            "grayscale_autocontrast": grok_gray,
            "sharpened_binary": grok_binary,
        },
        split_results=[
            _make_structured(
                question_raw="Solve log3 x + log9 x = 12",
                answer_raw="x = 6561",
                subject=SubjectLabel.MATH,
                split_confidence=0.71,
            ),
            _make_structured(
                question_raw=expected["question_raw"],
                answer_raw=expected["answer_raw"],
                subject=SubjectLabel.MATH,
                split_confidence=0.98,
            ),
        ],
    )

    result = ExtractionPipeline(
        settings=_settings(),
        grok_client=grok_client,
        azure_client=FakeAzureClient(candidate=azure_candidate),
    ).extract(str(image_path))

    assert result.question_raw == expected["question_raw"]
    assert "x = 3^8" in result.answer_raw
    assert result.subject == SubjectLabel.MATH
    assert result.diagnostics is not None
    assert str(result.diagnostics.selected_ocr_engine) == "azure"
    assert len(result.diagnostics.ocr_candidates) >= 4


def test_pipeline_applies_math_answer_refine_when_enabled(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    image_path.write_bytes(PNG_BYTES)

    garbled_answer = (
        "Answer: log (9) = log (3^2) = 2log (3) log (4) log (2^2) 2log (2) = log (3) = log_2 (3) log (2)"
    )
    question_line = "Show that log_4 9 = log_2 3"
    full_text = f"{question_line}\n{garbled_answer}"
    refined = r"\frac{\log 9}{\log 4} = \frac{\log 3}{\log 2} = \log_2 3"

    candidate = _make_candidate(
        engine=OCREngine.GROK,
        variant="original",
        full_text=full_text,
        ocr_confidence=0.96,
    )
    grok_client = FakeGrokClient(
        ocr_by_variant={
            "original": candidate,
            "grayscale_autocontrast": candidate.model_copy(update={"variant": "grayscale_autocontrast"}),
            "sharpened_binary": candidate.model_copy(update={"variant": "sharpened_binary"}),
        },
        split_results=[
            _make_structured(
                question_raw=question_line,
                answer_raw=garbled_answer,
                subject=SubjectLabel.MATH,
                split_confidence=0.95,
                whole_text_raw=full_text,
            ),
            _make_structured(
                question_raw=question_line,
                answer_raw=garbled_answer,
                subject=SubjectLabel.MATH,
                split_confidence=0.95,
                whole_text_raw=full_text,
            ),
        ],
        math_refine_result=MathAnswerRefineResult(
            refined_answer=refined,
            confidence=0.92,
            rationale="explicit fractions",
        ),
    )

    result = ExtractionPipeline(
        settings=_settings(enable_azure_fallback=False, enable_math_answer_refine=True),
        grok_client=grok_client,
        azure_client=FakeAzureClient(available=False),
    ).extract(str(image_path))

    assert grok_client.refine_math_calls == 1
    assert result.answer_raw == refined
    assert result.diagnostics is not None
    assert result.diagnostics.math_answer_refine is not None
    assert result.diagnostics.math_answer_refine.applied is True
    assert result.diagnostics.math_answer_refine.confidence == 0.92


def test_pipeline_skips_math_answer_refine_when_disabled(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    image_path.write_bytes(PNG_BYTES)

    garbled_answer = (
        "Answer: log (9) = log (3^2) = 2log (3) log (4) log (2^2) 2log (2) = log (3) = log_2 (3) log (2)"
    )
    question_line = "Show that log_4 9 = log_2 3"
    full_text = f"{question_line}\n{garbled_answer}"

    candidate = _make_candidate(
        engine=OCREngine.GROK,
        variant="original",
        full_text=full_text,
        ocr_confidence=0.96,
    )
    grok_client = FakeGrokClient(
        ocr_by_variant={
            "original": candidate,
            "grayscale_autocontrast": candidate.model_copy(update={"variant": "grayscale_autocontrast"}),
            "sharpened_binary": candidate.model_copy(update={"variant": "sharpened_binary"}),
        },
        split_results=[
            _make_structured(
                question_raw=question_line,
                answer_raw=garbled_answer,
                subject=SubjectLabel.MATH,
                split_confidence=0.95,
                whole_text_raw=full_text,
            ),
            _make_structured(
                question_raw=question_line,
                answer_raw=garbled_answer,
                subject=SubjectLabel.MATH,
                split_confidence=0.95,
                whole_text_raw=full_text,
            ),
        ],
        math_refine_result=MathAnswerRefineResult(refined_answer="should-not-apply", confidence=0.99, rationale=""),
    )

    result = ExtractionPipeline(
        settings=_settings(enable_azure_fallback=False, enable_math_answer_refine=False),
        grok_client=grok_client,
        azure_client=FakeAzureClient(available=False),
    ).extract(str(image_path))

    assert grok_client.refine_math_calls == 0
    assert result.answer_raw == garbled_answer
    assert result.diagnostics is not None
    assert result.diagnostics.math_answer_refine is None


def test_pipeline_rejects_low_confidence_math_answer_refine(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    image_path.write_bytes(PNG_BYTES)

    garbled_answer = (
        "Answer: log (9) = log (3^2) = 2log (3) log (4) log (2^2) 2log (2) = log (3) = log_2 (3) log (2)"
    )
    question_line = "Show that log_4 9 = log_2 3"
    full_text = f"{question_line}\n{garbled_answer}"

    candidate = _make_candidate(
        engine=OCREngine.GROK,
        variant="original",
        full_text=full_text,
        ocr_confidence=0.96,
    )
    grok_client = FakeGrokClient(
        ocr_by_variant={
            "original": candidate,
            "grayscale_autocontrast": candidate.model_copy(update={"variant": "grayscale_autocontrast"}),
            "sharpened_binary": candidate.model_copy(update={"variant": "sharpened_binary"}),
        },
        split_results=[
            _make_structured(
                question_raw=question_line,
                answer_raw=garbled_answer,
                subject=SubjectLabel.MATH,
                split_confidence=0.95,
                whole_text_raw=full_text,
            ),
            _make_structured(
                question_raw=question_line,
                answer_raw=garbled_answer,
                subject=SubjectLabel.MATH,
                split_confidence=0.95,
                whole_text_raw=full_text,
            ),
        ],
        math_refine_result=MathAnswerRefineResult(
            refined_answer=r"\frac{9}{4}",
            confidence=0.40,
            rationale="unsure",
        ),
    )

    result = ExtractionPipeline(
        settings=_settings(
            enable_azure_fallback=False,
            enable_math_answer_refine=True,
            math_answer_refine_min_confidence=0.85,
        ),
        grok_client=grok_client,
        azure_client=FakeAzureClient(available=False),
    ).extract(str(image_path))

    assert grok_client.refine_math_calls == 1
    assert result.answer_raw == garbled_answer
    assert result.diagnostics is not None
    assert result.diagnostics.math_answer_refine is not None
    assert result.diagnostics.math_answer_refine.applied is False
    assert result.diagnostics.math_answer_refine.skipped_reason == "below_confidence_threshold"
    assert any("below confidence threshold" in r for r in result.diagnostics.selection_reasons)
