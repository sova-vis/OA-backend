from __future__ import annotations

import re
from difflib import SequenceMatcher

from .types import (
    DisagreementSpan,
    OCRCandidate,
    OCREngine,
    RepairAction,
)


def score_candidates(candidates: list[OCRCandidate]) -> tuple[list[OCRCandidate], list[str]]:
    if not candidates:
        return [], []

    scored: list[OCRCandidate] = []
    for candidate in candidates:
        peers = [peer for peer in candidates if peer is not candidate]
        score = _score_candidate(candidate, peers)
        scored.append(candidate.model_copy(update={"selection_score": round(score, 4)}))

    ranked = sorted(
        scored,
        key=lambda item: (item.selection_score or 0.0, item.ocr_confidence),
        reverse=True,
    )
    selected = ranked[0]
    reasons = _build_selection_reasons(selected, ranked[1:])
    return ranked, reasons


def detect_disagreement_spans(
    selected: OCRCandidate,
    alternates: list[OCRCandidate],
) -> list[DisagreementSpan]:
    if not selected.lines:
        return []

    by_key: dict[tuple[int, int], DisagreementSpan] = {}
    for alternate in alternates:
        alternate_lines = {
            (line.page_number, line.line_index): line.text.strip()
            for line in alternate.lines
            if line.text.strip()
        }
        for line in selected.lines:
            key = (line.page_number, line.line_index)
            alternate_text = alternate_lines.get(key, "").strip()
            selected_text = line.text.strip()
            if not alternate_text or not selected_text:
                continue
            similarity = _similarity(selected_text, alternate_text)
            if similarity >= 0.995 and not _is_math_sensitive_difference(selected_text, alternate_text):
                continue

            disagreement = by_key.get(key)
            if disagreement is None:
                disagreement = DisagreementSpan(
                    page_number=line.page_number,
                    line_index=line.line_index,
                    selected_text=selected_text,
                    alternate_texts=[],
                    disagreement_score=round(1.0 - similarity, 4),
                    unresolved=True,
                )
            else:
                disagreement = disagreement.model_copy(
                    update={
                        "disagreement_score": round(
                            max(disagreement.disagreement_score, 1.0 - similarity),
                            4,
                        )
                    }
                )

            if alternate_text not in disagreement.alternate_texts:
                disagreement.alternate_texts.append(alternate_text)
            by_key[key] = disagreement

    return sorted(by_key.values(), key=lambda item: (item.page_number, item.line_index))


def apply_repair_actions(candidate: OCRCandidate, repair_actions: list[RepairAction]) -> OCRCandidate:
    accepted = {(action.page_number, action.line_index): action.after_text for action in repair_actions if action.accepted}
    if not accepted:
        return candidate

    updated_lines = []
    for line in candidate.lines:
        replacement = accepted.get((line.page_number, line.line_index))
        if replacement is None:
            updated_lines.append(line)
            continue
        updated_lines.append(line.model_copy(update={"text": replacement}))

    full_text = rebuild_full_text(updated_lines)
    return candidate.model_copy(update={"lines": updated_lines, "full_text": full_text})


def rebuild_full_text(lines) -> str:
    pages: dict[int, list[str]] = {}
    for line in lines:
        pages.setdefault(line.page_number, []).append(line.text)

    page_texts = []
    for page_number in sorted(pages):
        page_texts.append("\n".join(text for text in pages[page_number] if text))
    return "\n\n[[PAGE_BREAK]]\n\n".join(text for text in page_texts if text).strip()


def render_indexed_lines(candidate: OCRCandidate) -> str:
    rendered = []
    for line in candidate.lines:
        rendered.append(f"[P{line.page_number}:L{line.line_index}] {line.text}")
    return "\n".join(rendered)


def render_disagreement_report(disagreements: list[DisagreementSpan]) -> str:
    chunks: list[str] = []
    for disagreement in disagreements:
        alternates = " | ".join(disagreement.alternate_texts)
        chunks.append(
            f"[P{disagreement.page_number}:L{disagreement.line_index}] "
            f"selected=`{disagreement.selected_text}` alternates=`{alternates}`"
        )
    return "\n".join(chunks)


def candidate_summary(candidate: OCRCandidate) -> str:
    return (
        f"{candidate.engine}:{candidate.variant} "
        f"(ocr_confidence={candidate.ocr_confidence:.2f}, selection_score={(candidate.selection_score or 0.0):.2f})"
    )


def _score_candidate(candidate: OCRCandidate, peers: list[OCRCandidate]) -> float:
    confidence = candidate.ocr_confidence
    symbol_score = _symbol_preservation_score(candidate.full_text)
    math_score = _math_token_score(candidate.full_text)
    qa_score = _qa_anchor_score(candidate.full_text)
    agreement = _agreement_score(candidate.full_text, peers)
    uncertainty_penalty = _uncertainty_penalty(candidate)
    fragmentation_penalty = _fragmentation_penalty(candidate)

    score = (
        0.38 * confidence
        + 0.20 * symbol_score
        + 0.18 * math_score
        + 0.12 * qa_score
        + 0.12 * agreement
    )
    score -= 0.12 * uncertainty_penalty
    score -= 0.08 * fragmentation_penalty
    if candidate.engine == OCREngine.GROK:
        score += 0.01
    return max(0.0, min(score, 1.0))


def _symbol_preservation_score(text: str) -> float:
    if not text.strip():
        return 0.0
    symbol_count = len(re.findall(r"[=+\-/*^_()\[\]{}|]", text))
    ratio = symbol_count / max(len(text), 1)
    balanced_bonus = 0.1 if _balanced(text) else 0.0
    return min(1.0, ratio * 20.0 + balanced_bonus + 0.35)


def _math_token_score(text: str) -> float:
    if not text.strip():
        return 0.0
    tokens = re.findall(r"(log|sqrt|sin|cos|tan|[A-Za-z]\^\d+|[A-Za-z]_\d+|\d+[A-Za-z]|[A-Za-z]+\d+)", text)
    if not tokens:
        return 0.5
    unique_tokens = len({token.lower() for token in tokens})
    return min(1.0, 0.4 + unique_tokens / 10.0)


def _qa_anchor_score(text: str) -> float:
    has_question = bool(re.search(r"(^|\n)\s*(q[.)]|question[:.]?)", text, re.IGNORECASE))
    has_answer = bool(re.search(r"(^|\n)\s*(a[.)]|answer[:.]?)", text, re.IGNORECASE))
    if has_question and has_answer:
        return 1.0
    if has_question or has_answer:
        return 0.75
    if re.search(r"solve|find|prove|show that|explain|state", text, re.IGNORECASE):
        return 0.55
    return 0.3


def _agreement_score(text: str, peers: list[OCRCandidate]) -> float:
    if not peers:
        return 0.5
    similarities = [_similarity(text, peer.full_text) for peer in peers if peer.full_text.strip()]
    if not similarities:
        return 0.5
    return sum(similarities) / len(similarities)


def _build_selection_reasons(selected: OCRCandidate, alternates: list[OCRCandidate]) -> list[str]:
    reasons = [
        f"Selected {candidate_summary(selected)} as the highest-scoring OCR candidate.",
        "Selection score combines OCR confidence, symbol preservation, math-token preservation, QA anchors, and peer agreement.",
    ]
    if alternates:
        top_alternate = alternates[0]
        reasons.append(
            f"Best alternate was {candidate_summary(top_alternate)}."
        )
    return reasons


def _similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, _normalize(left), _normalize(right)).ratio()


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _balanced(text: str) -> bool:
    for opener, closer in (("(", ")"), ("[", "]"), ("{", "}")):
        if text.count(opener) != text.count(closer):
            return False
    return True


def _is_math_sensitive_difference(left: str, right: str) -> bool:
    if _normalize(left) == _normalize(right):
        return False
    math_markers = ("log", "^", "_", "/", "=", "+", "-")
    return any(marker in left or marker in right for marker in math_markers)


def _uncertainty_penalty(candidate: OCRCandidate) -> float:
    if not candidate.lines:
        return 0.0
    ratio = len(candidate.uncertain_spans) / max(len(candidate.lines), 1)
    return min(1.0, ratio)


def _fragmentation_penalty(candidate: OCRCandidate) -> float:
    if not candidate.lines:
        return 0.0
    short_or_fragmented = 0
    for line in candidate.lines:
        text = line.text.strip()
        if len(text) <= 3 or re.fullmatch(r"[=+\-/*^_0-9., ]+", text):
            short_or_fragmented += 1
    return min(1.0, short_or_fragmented / max(len(candidate.lines), 1))
