from __future__ import annotations

from pathlib import Path

from .arbitration import apply_repair_actions, detect_disagreement_spans, score_candidates
from .azure_client import AzureDocumentIntelligenceClient
from .config import Settings
from .grok_client import GrokClient
from .ingest import load_document
from .preprocess import build_variants
from .types import (
    ExtractionDiagnostics,
    ExtractionResult,
    LineTarget,
    OCRCandidate,
    RepairAction,
    StructuredExtraction,
)
from .validators import (
    build_confidence,
    needs_review,
    normalize_text,
    validate_diagnostics,
    validate_extraction,
)


class ExtractionPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        grok_client: GrokClient | None = None,
        azure_client: AzureDocumentIntelligenceClient | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env(Path.cwd())
        self._owns_grok_client = grok_client is None
        self._owns_azure_client = azure_client is None
        self.grok_client = grok_client or GrokClient(self.settings)
        self.azure_client = azure_client or AzureDocumentIntelligenceClient(self.settings)

    def close(self) -> None:
        if self._owns_grok_client and hasattr(self.grok_client, "close"):
            self.grok_client.close()
        if self._owns_azure_client and hasattr(self.azure_client, "close"):
            self.azure_client.close()

    def extract(self, input_path: str, *, page_number: int | None = None) -> ExtractionResult:
        document = load_document(input_path, self.settings, page_number=page_number)
        variants = build_variants(document, self.settings)
        variant_pages = {variant.name: variant.pages for variant in variants}

        grok_candidates = [
            self.grok_client.ocr_pages(variant.pages, variant_name=variant.name)
            for variant in variants
        ]
        ranked_candidates, selection_reasons = score_candidates(grok_candidates)
        selected_candidate = ranked_candidates[0]
        selected_pages = variant_pages.get(selected_candidate.variant, document.pages)

        provisional_extraction = self.grok_client.split_and_classify(selected_pages, selected_candidate)
        provisional_result = self._finalize_extraction(
            input_type=document.input_type,
            page_count=document.page_count,
            selected_candidate=selected_candidate,
            structured=provisional_extraction,
            diagnostics=None,
        )

        all_candidates = list(ranked_candidates)
        if self._should_use_azure_fallback(document.input_type, provisional_result):
            if self.azure_client.is_available:
                try:
                    azure_candidate = self.azure_client.analyze_path(document.source_path)
                except Exception as exc:
                    selection_reasons.append(f"Azure fallback OCR failed and was skipped: {exc}")
                else:
                    all_candidates.append(azure_candidate)
                    ranked_candidates, selection_reasons = score_candidates(all_candidates)
                    selected_candidate = ranked_candidates[0]
                    selected_pages = variant_pages.get(selected_candidate.variant, document.pages)
            else:
                selection_reasons.append(
                    "Azure fallback conditions were met, but Azure credentials were unavailable so Grok remained the only OCR source."
                )

        disagreements = detect_disagreement_spans(selected_candidate, ranked_candidates[1:])
        repair_actions: list[RepairAction] = []
        if self.settings.enable_targeted_repair and disagreements:
            proposed_actions = self.grok_client.repair_disagreements(
                selected_pages,
                selected_candidate,
                disagreements,
            )
            selected_candidate, accepted_actions = self._accept_repair_actions(
                selected_candidate,
                ranked_candidates[1:],
                proposed_actions,
            )
            repair_actions = accepted_actions
            ranked_candidates = [selected_candidate, *ranked_candidates[1:]]
            if any(action.accepted for action in repair_actions):
                selection_reasons.append("Accepted targeted repair actions for disputed OCR lines.")
            elif repair_actions:
                selection_reasons.append("Rejected targeted repair actions because they lowered candidate quality.")
            disagreements = detect_disagreement_spans(selected_candidate, ranked_candidates[1:])

        diagnostics = ExtractionDiagnostics(
            selected_ocr_engine=selected_candidate.engine,
            selected_variant=selected_candidate.variant,
            ocr_candidates=ranked_candidates,
            disagreement_spans=disagreements,
            repair_actions=repair_actions,
            selection_reasons=selection_reasons,
        )

        structured = self.grok_client.split_and_classify(selected_pages, selected_candidate)
        if structured.split_confidence < self.settings.grok_fallback_split_threshold:
            structured = self._retry_split_if_helpful(selected_pages, selected_candidate, structured, diagnostics)

        return self._finalize_extraction(
            input_type=document.input_type,
            page_count=document.page_count,
            selected_candidate=selected_candidate,
            structured=structured,
            diagnostics=diagnostics,
        )

    def _accept_repair_actions(
        self,
        selected_candidate: OCRCandidate,
        alternates: list[OCRCandidate],
        proposed_actions: list[RepairAction],
    ) -> tuple[OCRCandidate, list[RepairAction]]:
        accepted_actions = [action for action in proposed_actions if action.accepted]
        if not accepted_actions:
            return selected_candidate, proposed_actions

        repaired_candidate = apply_repair_actions(selected_candidate, accepted_actions)
        repaired_ranking, _ = score_candidates([repaired_candidate, *alternates])
        repaired_selected = repaired_ranking[0]
        repaired_score = repaired_selected.selection_score or 0.0
        current_score = selected_candidate.selection_score or 0.0
        if repaired_selected.full_text == repaired_candidate.full_text and repaired_score >= current_score:
            return repaired_selected, proposed_actions

        rejected_actions = [
            action.model_copy(update={"accepted": False}) if action.accepted else action
            for action in proposed_actions
        ]
        return selected_candidate, rejected_actions

    def _retry_split_if_helpful(
        self,
        selected_pages,
        selected_candidate: OCRCandidate,
        structured: StructuredExtraction,
        diagnostics: ExtractionDiagnostics,
    ) -> StructuredExtraction:
        split_retry = self.grok_client.retry_split(selected_pages, selected_candidate)
        retry_question, retry_answer = self._materialize_retry_assignment(selected_candidate, split_retry)
        if not retry_question and not retry_answer:
            diagnostics.selection_reasons.append("Split retry returned no usable line assignments.")
            return structured

        if split_retry.split_confidence >= structured.split_confidence:
            diagnostics.selection_reasons.append(
                "Accepted split-only retry because it improved or matched split confidence with explicit line assignments."
            )
            return structured.model_copy(
                update={
                    "whole_text_raw": selected_candidate.full_text,
                    "question_raw": retry_question,
                    "answer_raw": retry_answer,
                    "ocr_confidence": selected_candidate.ocr_confidence,
                    "split_confidence": split_retry.split_confidence,
                }
            )

        diagnostics.selection_reasons.append("Rejected split-only retry because it did not improve split confidence.")
        return structured

    def _materialize_retry_assignment(self, candidate: OCRCandidate, split_retry) -> tuple[str, str]:
        line_lookup = {(line.page_number, line.line_index): line.text for line in candidate.lines}
        question_lines: list[str] = []
        answer_lines: list[str] = []
        for assignment in split_retry.assignments:
            text = line_lookup.get((assignment.page_number, assignment.line_index), "")
            if not text:
                continue
            if assignment.target == LineTarget.QUESTION:
                question_lines.append(text)
            elif assignment.target == LineTarget.ANSWER:
                answer_lines.append(text)
        return normalize_text("\n".join(question_lines)), normalize_text("\n".join(answer_lines))

    def _finalize_extraction(
        self,
        input_type: str,
        page_count: int,
        selected_candidate: OCRCandidate,
        structured: StructuredExtraction,
        diagnostics: ExtractionDiagnostics | None,
    ) -> ExtractionResult:
        whole_text_raw = normalize_text(selected_candidate.full_text or structured.whole_text_raw)
        question_raw = normalize_text(structured.question_raw)
        answer_raw = normalize_text(structured.answer_raw)
        confidence = build_confidence(
            selected_candidate.ocr_confidence,
            structured.split_confidence,
            structured.classification_confidence,
        )
        flags = validate_extraction(
            whole_text_raw=whole_text_raw,
            question_raw=question_raw,
            answer_raw=answer_raw,
            subject=structured.subject,
            confidence=confidence,
            settings=self.settings,
        )
        if diagnostics is not None:
            flags.extend(validate_diagnostics(diagnostics, self.settings))

        return ExtractionResult(
            input_type=input_type,
            page_count=page_count,
            whole_text_raw=whole_text_raw,
            question_raw=question_raw,
            answer_raw=answer_raw,
            question_normalized=normalize_text(question_raw),
            answer_normalized=normalize_text(answer_raw),
            subject=structured.subject,
            confidence=confidence,
            flags=flags,
            needs_review=needs_review(flags, confidence, self.settings, diagnostics=diagnostics),
            diagnostics=diagnostics,
        )

    def _should_use_azure_fallback(self, input_type: str, provisional_result: ExtractionResult) -> bool:
        if not self.settings.enable_azure_fallback:
            return False
        if input_type == "pdf":
            return True
        if provisional_result.confidence.ocr < self.settings.grok_fallback_ocr_threshold:
            return True
        if provisional_result.confidence.split < self.settings.grok_fallback_split_threshold:
            return True
        ambiguity_flags = {
            "potential_subscript_loss",
            "potential_superscript_loss",
            "potential_equation_structure_loss",
            "potential_symbol_ambiguity",
            "potential_log_base_mismatch",
            "potential_fraction_or_chain_ambiguity",
        }
        return any(flag.code in ambiguity_flags for flag in provisional_result.flags)


def extract_qa(input_path: str, *, page_number: int | None = None) -> ExtractionResult:
    pipeline = ExtractionPipeline()
    try:
        return pipeline.extract(input_path, page_number=page_number)
    finally:
        pipeline.close()
