from __future__ import annotations

import base64
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SubjectLabel(str, Enum):
    MATH = "Math"
    PHYSICS = "Physics"
    CHEMISTRY = "Chemistry"
    BIOLOGY = "Biology"
    ENGLISH = "English"
    COMPUTER_SCIENCE = "Computer Science"
    HISTORY = "History"
    GEOGRAPHY = "Geography"
    ECONOMICS = "Economics"
    OTHER = "Other"
    UNKNOWN = "Unknown"


class FlagSeverity(str, Enum):
    WARNING = "warning"
    ERROR = "error"


class OCREngine(str, Enum):
    GROK = "grok"
    AZURE = "azure"


class OCRVariant(str, Enum):
    ORIGINAL = "original"
    GRAYSCALE_AUTOCONTRAST = "grayscale_autocontrast"
    SHARPENED_BINARY = "sharpened_binary"


class LineTarget(str, Enum):
    QUESTION = "question"
    ANSWER = "answer"
    UNKNOWN = "unknown"


class ValidationFlag(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    code: str = Field(description="Stable machine-readable flag code.")
    severity: FlagSeverity = Field(description="Warning or error severity.")
    message: str = Field(description="Human-readable explanation for the flag.")


class ConfidenceScores(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ocr: float = Field(ge=0.0, le=1.0)
    split: float = Field(ge=0.0, le=1.0)
    classification: float = Field(ge=0.0, le=1.0)
    overall: float = Field(ge=0.0, le=1.0)


class LineOCR(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_number: int
    line_index: int
    text: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class UncertainSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_number: int
    text: str
    reason: str
    line_index: int | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class OCRCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    engine: OCREngine
    variant: str
    full_text: str
    lines: list[LineOCR] = Field(default_factory=list)
    ocr_confidence: float = Field(ge=0.0, le=1.0)
    uncertain_spans: list[UncertainSpan] = Field(default_factory=list)
    selection_score: float | None = Field(default=None, ge=0.0, le=1.0)


class DisagreementSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_number: int
    line_index: int
    selected_text: str
    alternate_texts: list[str]
    disagreement_score: float = Field(ge=0.0, le=1.0)
    unresolved: bool = True


class RepairAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_number: int
    line_index: int
    before_text: str
    after_text: str
    source: str
    accepted: bool
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


class MathAnswerRefineResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    refined_answer: str = Field(description="Re-transcribed answer only, with explicit fractions.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the refined transcription.")
    rationale: str = Field(default="", description="Brief note on what was fixed or left unchanged.")


class MathAnswerRefineDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    applied: bool = Field(description="Whether answer_raw was replaced with the refined string.")
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    rationale: str = ""
    skipped_reason: str | None = Field(
        default=None,
        description="Why refine was skipped or not applied (e.g. disabled, below threshold, error).",
    )


class ExtractionDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    selected_ocr_engine: OCREngine
    selected_variant: str
    ocr_candidates: list[OCRCandidate] = Field(default_factory=list)
    disagreement_spans: list[DisagreementSpan] = Field(default_factory=list)
    repair_actions: list[RepairAction] = Field(default_factory=list)
    selection_reasons: list[str] = Field(default_factory=list)
    split_retry_applied: bool = False
    math_answer_refine: MathAnswerRefineDiagnostics | None = None


class ExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    input_type: str
    page_count: int
    whole_text_raw: str
    question_raw: str
    answer_raw: str
    question_normalized: str
    answer_normalized: str
    subject: SubjectLabel
    confidence: ConfidenceScores
    flags: list[ValidationFlag]
    needs_review: bool
    diagnostics: ExtractionDiagnostics | None = None


class OCRPagePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_number: int
    full_text: str
    lines: list[str] = Field(default_factory=list)
    ocr_confidence: float = Field(ge=0.0, le=1.0)
    uncertain_spans: list[str] = Field(default_factory=list)


class StructuredOCRResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    full_text: str
    pages: list[OCRPagePayload]
    ocr_confidence: float = Field(ge=0.0, le=1.0)


class StructuredExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    whole_text_raw: str = Field(description="Full transcription across all pages.")
    question_raw: str = Field(description="Extracted question text only.")
    answer_raw: str = Field(description="Extracted answer text only.")
    subject: SubjectLabel = Field(description="Broad subject label from the allowed taxonomy.")
    ocr_confidence: float = Field(ge=0.0, le=1.0)
    split_confidence: float = Field(ge=0.0, le=1.0)
    classification_confidence: float = Field(ge=0.0, le=1.0)


class RepairDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_number: int
    line_index: int
    repaired_text: str
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


class RepairDecisionSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actions: list[RepairDecision] = Field(default_factory=list)


class LineAssignment(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    page_number: int
    line_index: int
    target: LineTarget


class SplitRetryResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assignments: list[LineAssignment] = Field(default_factory=list)
    split_confidence: float = Field(ge=0.0, le=1.0)


@dataclass(frozen=True)
class DocumentPage:
    page_number: int
    mime_type: str
    content_bytes: bytes
    source_name: str
    variant_name: str = OCRVariant.ORIGINAL.value

    def to_data_url(self) -> str:
        payload = base64.b64encode(self.content_bytes).decode("ascii")
        return f"data:{self.mime_type};base64,{payload}"


@dataclass(frozen=True)
class DocumentInput:
    input_type: str
    source_path: Path
    pages: tuple[DocumentPage, ...]

    @property
    def page_count(self) -> int:
        return len(self.pages)


@dataclass(frozen=True)
class DocumentVariant:
    name: str
    pages: tuple[DocumentPage, ...]


class OAExtractionError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "oa_extraction_error",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}


class ConfigurationError(OAExtractionError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="configuration_error")


class InputDocumentError(OAExtractionError):
    def __init__(self, message: str, *, path: str | None = None) -> None:
        details = {"path": path} if path else None
        super().__init__(message, code="input_document_error", details=details)


class GrokAPIError(OAExtractionError):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        details = {"status_code": status_code} if status_code is not None else None
        super().__init__(message, code="grok_api_error", details=details)
