from typing import Any, Dict, List, Tuple, Optional
import io
import gc
import datetime
import os
import re
import sys

import cv2
import numpy as np
import fitz
from PIL import Image

# Try to import psutil for cross-platform memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    # Fallback to resource module (Unix only)
    try:
        import resource
        RESOURCE_AVAILABLE = True
    except ImportError:
        RESOURCE_AVAILABLE = False

# -----------------------------
# CONSTANTS
# -----------------------------
MAX_DIMENSION_BEFORE_NUMPY = 6500
MAX_DIMENSION_BEFORE_RESIZE = 4000
MAX_PIL_PIXELS = 80_000_000
LOW_MEMORY_MB = 500.0
LOW_MEMORY_WARN_MB = 200.0
MIN_PAGE_HEIGHT = 2800
SIDE_MARGIN_RATIO = 0.40
MARGIN_RATIO = 0.03
PAGE_DPI = 200
COLOR_RED_BGR = (0, 0, 255)
COLOR_GREEN_BGR = (0, 180, 0)
COLOR_SUGGESTION_BGR = (255, 140, 0)
MAX_SUGGESTIONS_PER_PAGE = 6
HEADING_MATCH_THRESHOLD = 0.8
FUZZY_MATCH_THRESHOLD = 0.5


def _get_available_memory_mb() -> Optional[float]:
    """
    Get available system memory in MB.
    Returns None if memory information is not available.
    """
    try:
        if PSUTIL_AVAILABLE:
            # Cross-platform memory info
            mem = psutil.virtual_memory()
            return mem.available / (1024 * 1024)  # Convert to MB
        elif RESOURCE_AVAILABLE and sys.platform != 'win32':
            # Unix-only: get process memory limit
            # Note: This gives process limit, not system available
            # For system memory, we'd need to parse /proc/meminfo
            try:
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if line.startswith('MemAvailable:'):
                            parts = line.split()
                            if len(parts) >= 2:
                                # Value is in KB, convert to MB
                                return float(parts[1]) / 1024.0
            except (IOError, ValueError):
                pass
    except Exception:
        pass
    return None


def _get_process_memory_mb() -> Optional[float]:
    """
    Get current process memory usage in MB.
    Returns None if memory information is not available.
    """
    try:
        if PSUTIL_AVAILABLE:
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            return mem_info.rss / (1024 * 1024)  # Convert to MB
        elif RESOURCE_AVAILABLE and sys.platform != 'win32':
            # Unix-only: get process memory usage
            usage = resource.getrusage(resource.RUSAGE_SELF)
            # maxrss is in KB on Linux, convert to MB
            return usage.ru_maxrss / 1024.0
    except Exception:
        pass
    return None


def _estimate_memory_requirements(
    page_count: int,
    avg_page_size_mb: float = 2.0,
    processing_copies: int = 4,
    safety_margin: float = 2.0,
) -> float:
    """
    Estimate memory requirements for annotation processing.
    
    Args:
        page_count: Number of pages to process
        avg_page_size_mb: Average size of one page image in MB (default: 2.0)
        processing_copies: Number of copies created during processing (default: 4)
        safety_margin: Safety margin multiplier (default: 2.0)
    
    Returns:
        Estimated memory requirement in MB
    """
    # Memory for processing one page at a time
    processing_memory = avg_page_size_mb * processing_copies
    
    # Memory for accumulated annotated pages (output)
    output_memory = page_count * avg_page_size_mb
    
    # Total estimated memory
    total_memory = processing_memory + output_memory
    
    # Apply safety margin
    estimated_memory = total_memory * safety_margin
    
    return estimated_memory


def _check_memory_before_processing(
    page_count: int,
    pdf_size_mb: Optional[float] = None,
    warn_threshold_mb: float = LOW_MEMORY_MB,
    fail_threshold_mb: float = LOW_MEMORY_WARN_MB,
) -> Tuple[bool, Optional[str]]:
    """
    Check if there's sufficient memory before starting processing.
    
    Args:
        page_count: Number of pages to process
        pdf_size_mb: Size of PDF file in MB (optional, for better estimation)
        warn_threshold_mb: Warn if available memory is below this (MB)
        fail_threshold_mb: Fail if available memory is below this (MB)
    
    Returns:
        Tuple of (should_proceed, warning_message)
        - should_proceed: True if processing should continue, False if should fail
        - warning_message: Optional warning message if memory is low
    """
    available_memory = _get_available_memory_mb()
    process_memory = _get_process_memory_mb()
    
    # Estimate memory requirements
    if pdf_size_mb:
        # Use actual PDF size to estimate page size
        avg_page_size = max(1.0, pdf_size_mb / max(1, page_count))
    else:
        # Use default estimate
        avg_page_size = 2.0
    
    estimated_required = _estimate_memory_requirements(
        page_count=page_count,
        avg_page_size_mb=avg_page_size,
    )
    
    # If we can't get memory info, proceed with warning
    if available_memory is None:
        return True, "Memory information not available - proceeding with caution"
    
    # Check if we have enough memory
    if available_memory < fail_threshold_mb:
        return False, (
            f"Insufficient memory: {available_memory:.1f} MB available, "
            f"estimated {estimated_required:.1f} MB required. "
            f"Processing may fail. Consider processing smaller files or increasing system memory."
        )
    
    if available_memory < estimated_required:
        return False, (
            f"Low memory: {available_memory:.1f} MB available, "
            f"estimated {estimated_required:.1f} MB required. "
            f"Processing may fail."
        )
    
    if available_memory < warn_threshold_mb:
        return True, (
            f"Low available memory: {available_memory:.1f} MB. "
            f"Estimated requirement: {estimated_required:.1f} MB. "
            f"Processing may be slow or fail with very large files."
        )
    
    # Memory looks good
    if process_memory:
        return True, (
            f"Memory check: {available_memory:.1f} MB available, "
            f"process using {process_memory:.1f} MB, "
            f"estimated {estimated_required:.1f} MB required"
        )
    
    return True, f"Memory check: {available_memory:.1f} MB available, estimated {estimated_required:.1f} MB required"


def _get_page_ocr(ocr_data: Dict[str, Any], page_number: int) -> Optional[Dict[str, Any]]:
    for p in ocr_data.get("pages", []):
        if p.get("page_number") == page_number:
            return p
    return None


def _bbox_to_rect(
    bbox: List[Tuple[int, int]],
    pad: int,
    w: int,
    h: int,
) -> Tuple[int, int, int, int]:
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    x1 = max(0, min(xs) - pad)
    y1 = max(0, min(ys) - pad)
    x2 = min(w - 1, max(xs) + pad)
    y2 = min(h - 1, max(ys) + pad)
    return x1, y1, x2, y2


def _wrap_text_cv2(
    text: str,
    max_width_px: int,
    font_face: int,
    font_scale: float,
    thickness: int,
) -> List[str]:
    words = text.split()
    lines: List[str] = []
    current = ""
    for w in words:
        trial = (current + " " + w).strip()
        if not trial:
            continue
        size, _ = cv2.getTextSize(trial, font_face, font_scale, thickness)
        if size[0] <= max_width_px or not current:
            current = trial
        else:
            # Check if single word is too long
            word_size, _ = cv2.getTextSize(w, font_face, font_scale, thickness)
            if word_size[0] > max_width_px:
                # Word too long, needs character-level breaking
                if current:
                    lines.append(current)
                # Break word into chunks that fit
                for i in range(0, len(w), max(1, int(len(w) * max_width_px / word_size[0]))):
                    chunk = w[i:i + max(1, int(len(w) * max_width_px / word_size[0]))]
                    chunk_size, _ = cv2.getTextSize(chunk, font_face, font_scale, thickness)
                    if chunk_size[0] <= max_width_px:
                        lines.append(chunk)
                    else:
                        # Even chunk too long, force break with hyphen
                        for j in range(len(chunk)):
                            test = chunk[:j+1]
                            test_size, _ = cv2.getTextSize(test + "-", font_face, font_scale, thickness)
                            if test_size[0] > max_width_px and j > 0:
                                lines.append(chunk[:j] + "-")
                                chunk = chunk[j:]
                                break
                        if chunk:
                            lines.append(chunk)
                current = ""
            else:
                lines.append(current)
                current = w
    if current:
        lines.append(current)
    return lines


def _normalize(text: str) -> str:
    return (text or "").strip().lower()


def _find_heading_bbox_on_page(
    title: str,
    page_ocr: Dict[str, Any],
) -> Optional[List[Tuple[int, int]]]:
    target = _normalize(title)
    if not target:
        return None

    best_bbox = None
    best_score = 0.0

    for line in page_ocr.get("lines", []):
        text = (line.get("text") or "").strip()
        if not text:
            continue

        # Skip very long lines (likely body text, not headings)
        if len(text) > 150:
            continue

        candidate = _normalize(text)

        # Exact match is best
        if candidate == target:
            return line.get("bbox")

        # For OCR variations, check token overlap but with higher threshold
        # Headings should match most of their tokens
        t_tokens = target.split()
        c_tokens = candidate.split()

        if len(t_tokens) == 0:
            continue

        # Only consider if all target tokens are present in candidate
        common = len(set(t_tokens) & set(c_tokens))
        score = common / len(t_tokens)

        # Higher threshold to avoid matching random body text
        if score > best_score and score >= HEADING_MATCH_THRESHOLD:
            best_score = score
            best_bbox = line.get("bbox")

    return best_bbox


def _find_word_or_line_rect(
    page_ocr: Dict[str, Any],
    target_text: str,
    w: int,
    h: int,
) -> Optional[Tuple[int, int, int, int]]:
    if not target_text:
        return None

    target_norm = _normalize(target_text)
    if not target_norm:
        return None

    tokens = [t for t in target_norm.split() if t]
    if not tokens:
        return None

    # 1) Try word-level matching for each token and union boxes
    matched_word_boxes: List[List[Tuple[int, int]]] = []
    for line in page_ocr.get("lines", []):
        for w_entry in line.get("words") or []:
            w_text = _normalize(w_entry.get("text") or "")
            if not w_text:
                continue
            if w_text in tokens or w_text == target_norm:
                if w_entry.get("bbox"):
                    matched_word_boxes.append(w_entry["bbox"])

    if matched_word_boxes:
        xs: List[int] = []
        ys: List[int] = []
        for bbox in matched_word_boxes:
            for x, y in bbox:
                xs.append(x)
                ys.append(y)
        if xs and ys:
            x1 = max(0, min(xs) - 2)
            y1 = max(0, min(ys) - 2)
            x2 = min(w - 1, max(xs) + 2)
            y2 = min(h - 1, max(ys) + 2)
            return x1, y1, x2, y2

    # 2) Fallback: line-level fuzzy matching, but reject huge boxes
    best_rect = None
    best_score = 0.0

    for line in page_ocr.get("lines", []):
        text = _normalize(line.get("text") or "")
        if not text:
            continue

        if target_norm in text or text in target_norm:
            score = 1.0
        else:
            t_tokens = set(tokens)
            l_tokens = set(text.split())
            common = len(t_tokens & l_tokens)
            score = common / max(len(tokens), 1)

        if score < FUZZY_MATCH_THRESHOLD or score <= best_score:
            continue

        bbox = line.get("bbox")
        if not bbox:
            continue

        x1, y1, x2, y2 = _bbox_to_rect(bbox, pad=2, w=w, h=h)
        box_w = x2 - x1
        box_h = y2 - y1

        # Reject boxes that are 'too big' (heuristic thresholds)
        if box_h > 0.3 * h or box_w > 0.9 * w:
            continue

        best_score = score
        best_rect = (x1, y1, x2, y2)

    return best_rect


def _find_anchor_rect_on_page(
    page_ocr: Dict[str, Any],
    anchor_quote: str,
    w: int,
    h: int,
    already_found: Optional[List[Tuple[int, int, int, int]]] = None,
) -> Optional[Tuple[int, int, int, int]]:
    """Resolve an anchor quote to a compact bbox using exact/fuzzy OCR matching."""
    anchor_norm = _normalize(anchor_quote)
    if not anchor_norm:
        return None

    already_found = already_found or []

    # Prefer precise token-sequence matching first.
    rect = _find_precise_word_rect_with_context(
        page_ocr=page_ocr,
        target_text=anchor_quote,
        context_before="",
        context_after="",
        w=w,
        h=h,
        already_found=already_found,
    )
    if rect is not None:
        return rect

    # Fallback to strong line-level overlap.
    anchor_tokens = set(anchor_norm.split())
    if not anchor_tokens:
        return None

    best_rect: Optional[Tuple[int, int, int, int]] = None
    best_score = 0.0

    for line in page_ocr.get("lines", []):
        text = _normalize(line.get("text") or "")
        bbox = line.get("bbox")
        if not text or not bbox:
            continue

        if anchor_norm in text:
            score = 1.0
        else:
            line_tokens = set(text.split())
            score = len(anchor_tokens & line_tokens) / max(len(anchor_tokens), 1)

        if score < 0.7 or score <= best_score:
            continue

        x1, y1, x2, y2 = _bbox_to_rect(bbox, pad=2, w=w, h=h)
        box_w = x2 - x1
        box_h = y2 - y1
        if box_h > 0.35 * h or box_w > 0.95 * w:
            continue

        is_duplicate = False
        for found_rect in already_found:
            fx1, fy1, fx2, fy2 = found_rect
            overlap_x = max(0, min(x2, fx2) - max(x1, fx1))
            overlap_y = max(0, min(y2, fy2) - max(y1, fy1))
            if overlap_x > 0 and overlap_y > 0:
                is_duplicate = True
                break
        if is_duplicate:
            continue

        best_score = score
        best_rect = (x1, y1, x2, y2)

    return best_rect


def _find_precise_word_rect_with_context(
    page_ocr: Dict[str, Any],
    target_text: str,
    context_before: str,
    context_after: str,
    w: int,
    h: int,
    already_found: Optional[List[Tuple[int, int, int, int]]] = None,
) -> Optional[Tuple[int, int, int, int]]:
    if not target_text:
        return None

    target_norm = _normalize(target_text)
    before_norm = _normalize(context_before)
    after_norm = _normalize(context_after)

    if not target_norm:
        return None

    already_found = already_found or []
    target_tokens = target_norm.split()

    # Build word sequence for the whole page
    all_words: List[Dict[str, Any]] = []
    for line in page_ocr.get("lines", []):
        for w_entry in line.get("words") or []:
            w_text = _normalize(w_entry.get("text") or "")
            if w_text and w_entry.get("bbox"):
                all_words.append({
                    "text": w_text,
                    "bbox": w_entry["bbox"]
                })

    if not all_words:
        return None

    # Find all positions where target appears
    best_rect = None
    best_score = 0.0

    for i in range(len(all_words) - len(target_tokens) + 1):
        # Check if target matches at position i
        matches = True
        for j, target_token in enumerate(target_tokens):
            if target_token != all_words[i + j]["text"]:
                matches = False
                break

        if not matches:
            continue

        # Get bboxes for matched words
        matched_bboxes = [all_words[i + j]["bbox"] for j in range(len(target_tokens))]

        # Union the bboxes
        xs: List[int] = []
        ys: List[int] = []
        for bbox in matched_bboxes:
            for x, y in bbox:
                xs.append(x)
                ys.append(y)

        if not xs or not ys:
            continue

        x1 = max(0, min(xs) - 2)
        y1 = max(0, min(ys) - 2)
        x2 = min(w - 1, max(xs) + 2)
        y2 = min(h - 1, max(ys) + 2)

        # Check if already found
        is_duplicate = False
        for found_rect in already_found:
            fx1, fy1, fx2, fy2 = found_rect
            overlap_x = max(0, min(x2, fx2) - max(x1, fx1))
            overlap_y = max(0, min(y2, fy2) - max(y1, fy1))
            if overlap_x > 0 and overlap_y > 0:
                is_duplicate = True
                break

        if is_duplicate:
            continue

        # Score based on context
        score = 1.0

        # Check context_before (look at previous words)
        if before_norm:
            prev_words = []
            for k in range(max(0, i - 7), i):
                prev_words.append(all_words[k]["text"])
            prev_text = " ".join(prev_words)

            if before_norm in prev_text:
                score += 2.0
            else:
                before_tokens = set(before_norm.split())
                prev_tokens = set(prev_words)
                if before_tokens:
                    overlap_ratio = len(before_tokens & prev_tokens) / len(before_tokens)
                    score += overlap_ratio * 1.0

        # Check context_after (look at next words)
        if after_norm:
            next_words = []
            for k in range(i + len(target_tokens), min(len(all_words), i + len(target_tokens) + 7)):
                next_words.append(all_words[k]["text"])
            next_text = " ".join(next_words)

            if after_norm in next_text:
                score += 2.0
            else:
                after_tokens = set(after_norm.split())
                next_tokens = set(next_words)
                if after_tokens:
                    overlap_ratio = len(after_tokens & next_tokens) / len(after_tokens)
                    score += overlap_ratio * 1.0

        # Update best match
        if score > best_score:
            best_score = score
            best_rect = (x1, y1, x2, y2)

    return best_rect


def _find_annotation_rect_with_context(
    page_ocr: Dict[str, Any],
    target_text: str,
    context_before: str,
    context_after: str,
    w: int,
    h: int,
    already_found: Optional[List[Tuple[int, int, int, int]]] = None,
) -> Optional[Tuple[int, int, int, int]]:
    if not target_text:
        return None

    target_norm = _normalize(target_text)
    before_norm = _normalize(context_before)
    after_norm = _normalize(context_after)

    if not target_norm:
        return None

    already_found = already_found or []

    # Build a full text representation of the page to find context matches
    full_text_parts: List[str] = []
    line_to_bbox: Dict[int, List[Tuple[int, int]]] = {}

    for idx, line in enumerate(page_ocr.get("lines", [])):
        text = line.get("text") or ""
        full_text_parts.append(text)
        line_to_bbox[idx] = line.get("bbox")

    full_text = " ".join(full_text_parts)
    full_text_norm = _normalize(full_text)

    # Strategy: Find all occurrences of target in the full text with context
    # Then map back to line indices

    best_rect: Optional[Tuple[int, int, int, int]] = None
    best_score = 0.0
    best_line_idx = -1

    # Pass 1: Search with context to find the right occurrence
    for line_idx, line in enumerate(page_ocr.get("lines", [])):
        text = _normalize(line.get("text") or "")
        if not text:
            continue

        # Check if target is in this line
        if target_norm not in text:
            continue

        bbox = line.get("bbox")
        if not bbox:
            continue

        x1, y1, x2, y2 = _bbox_to_rect(bbox, pad=3, w=w, h=h)
        box_w = x2 - x1
        box_h = y2 - y1

        # Avoid 'giant' boxes that span too much of the page
        if box_h > 0.35 * h or box_w > 0.95 * w:
            continue

        # Check if this rect was already found (for handling duplicates)
        is_duplicate = False
        for found_rect in already_found:
            fx1, fy1, fx2, fy2 = found_rect
            # Check if rects overlap significantly
            overlap_x = max(0, min(x2, fx2) - max(x1, fx1))
            overlap_y = max(0, min(y2, fy2) - max(y1, fy1))
            if overlap_x > 0 and overlap_y > 0:
                is_duplicate = True
                break

        if is_duplicate:
            continue

        # Score this match based on context
        score = 1.0  # Base score for having the target

        # Check context before
        if before_norm:
            # Look at previous lines
            prev_texts = []
            for i in range(max(0, line_idx - 2), line_idx):
                prev_texts.append(_normalize(page_ocr.get("lines", [])[i].get("text") or ""))
            prev_context = " ".join(prev_texts) + " " + text

            if before_norm in prev_context:
                score += 2.0
            else:
                # Partial match with context_before tokens
                before_tokens = set(before_norm.split())
                context_tokens = set(prev_context.split())
                if before_tokens:
                    overlap = len(before_tokens & context_tokens) / len(before_tokens)
                    score += overlap * 1.0

        # Check context after
        if after_norm:
            # Look at next lines
            next_texts = [text]
            for i in range(line_idx + 1, min(len(page_ocr.get("lines", [])), line_idx + 3)):
                next_texts.append(_normalize(page_ocr.get("lines", [])[i].get("text") or ""))
            next_context = " ".join(next_texts)

            if after_norm in next_context:
                score += 2.0
            else:
                # Partial match with context_after tokens
                after_tokens = set(after_norm.split())
                context_tokens = set(next_context.split())
                if after_tokens:
                    overlap = len(after_tokens & context_tokens) / len(after_tokens)
                    score += overlap * 1.0

        # Update best match
        if score > best_score:
            best_score = score
            best_rect = (x1, y1, x2, y2)
            best_line_idx = line_idx

    if best_rect is not None:
        return best_rect

    # Pass 2: Fallback to word-level matching without context
    return _find_word_or_line_rect(page_ocr, target_text, w=w, h=h)



def _compute_section_region_on_page(
    section: Dict[str, Any],
    page_number: int,
    ocr_data: Dict[str, Any],
    orig_w: int,
    orig_h: int,
    all_sections: List[Dict[str, Any]],
) -> Optional[Tuple[int, int, int, int]]:
    page_ocr = _get_page_ocr(ocr_data, page_number)
    if not page_ocr:
        return None

    title = section.get("title") or ""
    heading_bbox = _find_heading_bbox_on_page(title, page_ocr)

    # Determine y_start
    first_section_page = min(section.get("page_numbers") or [page_number])
    if heading_bbox and page_number == first_section_page:
        xh1, yh1, xh2, yh2 = _bbox_to_rect(heading_bbox, pad=6, w=orig_w, h=orig_h)
        y_start = max(0, yh1 - 4)
    else:
        y_start = 0

    # Determine y_end from next section on this page
    y_end = orig_h - 1
    this_index = None
    for idx, sec in enumerate(all_sections):
        if sec is section:
            this_index = idx
            break

    if this_index is not None:
        for next_sec in all_sections[this_index + 1:]:
            if page_number not in (next_sec.get("page_numbers") or []):
                continue
            next_page_ocr = _get_page_ocr(ocr_data, page_number)
            if not next_page_ocr:
                continue
            next_bbox = _find_heading_bbox_on_page(next_sec.get("title") or "", next_page_ocr)
            if next_bbox:
                _, ny1, _, _ = _bbox_to_rect(next_bbox, pad=4, w=orig_w, h=orig_h)
                if ny1 > y_start:
                    y_end = ny1 - 4
                break

    # Collect all lines between y_start and y_end
    xs: List[int] = []
    ys: List[int] = []
    for line in page_ocr.get("lines", []):
        lbbox = line.get("bbox")
        if not lbbox:
            continue
        lx1, ly1, lx2, ly2 = _bbox_to_rect(lbbox, pad=0, w=orig_w, h=orig_h)
        mid_y = (ly1 + ly2) // 2
        if y_start <= mid_y <= y_end:
            xs.extend([lx1, lx2])
            ys.extend([ly1, ly2])

    if not xs or not ys:
        return None

    x1 = max(0, min(xs))
    x2 = min(orig_w - 1, max(xs))
    y1 = max(0, min(ys))
    y2 = min(orig_h - 1, max(ys))
    return x1, y1, x2, y2


# ---------- MAIN ANNOTATION FUNCTION ----------


def annotate_pdf_answer_pages(
    pdf_path: str,
    ocr_data: Dict[str, Any],
    sections: List[Dict[str, Any]],
    annotations: List[Dict[str, Any]],
    page_suggestions: Optional[List[Dict[str, Any]]] = None,
    log_path: Optional[str] = None,
    request_id: Optional[str] = None,
) -> List[Image.Image]:
    """
    Create annotated versions of the answer pages.

    NEW LAYOUT:
      - [LEFT MARGIN with improvement suggestions][Original Answer Page]
      - Left margin shows specific, actionable suggestions for each page
      - Annotations still mark errors/issues on the answer itself

    Annotation behaviour:
      - introduction_comment → big red box for whole introduction + red comment in side padding
      - heading_issue (negative) → red box on heading line + red comment in side padding
      - factual_error → red box on sentence line + red comment in side padding
      - grammar_language → small red box on word/sentence + red correction near box (no side comment)
      - repetition → red box on repeated section region, text "repeated on page X" in red near box
      - For each 'correct' section on a page (no negative annotation on that page):
          → draw a red ✓ near the heading and near the section body region.
    """
    page_suggestions = page_suggestions or []

    def _append_skipped_annotations_log(
        skipped: List[Dict[str, Any]],
        *,
        page_number: int,
    ) -> None:
        """
        Write skipped-annotation diagnostics to a dedicated log file.

        This is intentionally separate from the main OCR log to keep it readable.
        """
        if not skipped:
            return
        if not log_path:
            return
        try:
            log_dir = os.path.dirname(log_path)
            if not log_dir:
                return
            os.makedirs(log_dir, exist_ok=True)
            out_path = os.path.join(log_dir, "skipped_annotations_log.txt")
            ts = datetime.datetime.utcnow().isoformat() + "Z"
            rid = request_id or "unknown"
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(f"{ts} request={rid} page={page_number} skipped={len(skipped)}\n")
                for item in skipped:
                    atype = (item.get("type") or "").strip()
                    reason = (item.get("reason") or "").strip().replace("\n", " ")
                    target = (item.get("target") or "").strip().replace("\n", " ")
                    # Keep lines short-ish to avoid huge logs.
                    if len(target) > 200:
                        target = target[:200] + "..."
                    f.write(
                        f"{ts} request={rid} page={page_number} type={atype} reason={reason} target={target}\n"
                    )
        except Exception:
            # Never fail annotation due to logging.
            pass
    
    # Get PDF file size for memory estimation
    pdf_size_mb = None
    try:
        pdf_size_bytes = os.path.getsize(pdf_path)
        pdf_size_mb = pdf_size_bytes / (1024 * 1024)
    except Exception:
        pass
    
    # Load PDF pages via PyMuPDF
    doc = fitz.open(pdf_path)
    try:
        page_count = len(doc)
        
        # Check memory before processing
        should_proceed, memory_message = _check_memory_before_processing(
            page_count=page_count,
            pdf_size_mb=pdf_size_mb,
        )
        
        if not should_proceed:
            # Fail gracefully with clear error message
            raise MemoryError(
                f"Cannot process PDF: {memory_message}. "
                f"PDF has {page_count} pages ({pdf_size_mb:.1f} MB if available). "
                f"Please try with a smaller file or increase available system memory."
            )
        
        # Log memory status (warning or info)
        if memory_message:
            print(f"Memory check: {memory_message}")
            if "Low" in memory_message or "caution" in memory_message.lower():
                print(f"WARNING: {memory_message}")
        
        annotated_pages: List[Image.Image] = []

        # Precompute OCR by page
        ocr_pages_by_num: Dict[int, Dict[str, Any]] = {
            p.get("page_number"): p for p in ocr_data.get("pages", [])
        }
        print(f"DEBUG: OCR pages available: {list(ocr_pages_by_num.keys())}")
        for page_num, page_data in ocr_pages_by_num.items():
            line_count = len(page_data.get("lines", []))
            print(f"  Page {page_num}: {line_count} lines")

        # Map sections by id for quick lookup
        sections_by_id: Dict[str, Dict[str, Any]] = {}
        for sec in sections:
            sid = sec.get("id") or sec.get("section_id") or sec.get("title")
            if sid:
                sections_by_id[str(sid)] = sec

        print(f"DEBUG: Sections found: {len(sections)}")
        if not sections:
            print("⚠️  WARNING: No sections provided to annotate_pdf_answer_pages!")
            print("  This will cause headings and section annotations to be missing.")
        else:
            for idx, sec in enumerate(sections):
                title = sec.get("title", "Untitled")
                pages = sec.get("page_numbers", [])
                print(f"  Section {idx+1}: '{title}' on pages {pages}")
                # Validate page_numbers format
                if pages:
                    invalid_pages = [p for p in pages if not isinstance(p, (int, float)) and not str(p).isdigit()]
                    if invalid_pages:
                        print(f"    WARNING: Section '{title}' has invalid page numbers: {invalid_pages}")

        # Drawing constants
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        RED = COLOR_RED_BGR
        GREEN = COLOR_GREEN_BGR  # slightly darker green for readability (BGR)

        for page_idx, page in enumerate(doc):
            page_number = page_idx + 1
            
            # Load this page as PIL image (process one at a time to reduce memory)
            pix = page.get_pixmap(dpi=PAGE_DPI)
            img_bytes = pix.tobytes("png")
            pil_img = Image.open(io.BytesIO(img_bytes))
            
            # Check image size and downscale if too large BEFORE converting to numpy
            # This prevents MemoryError when converting very large images to numpy arrays
            max_dimension_before_numpy = 6500  # Maximum dimension before downscaling
            pil_w, pil_h = pil_img.size
            max_dim = max(pil_w, pil_h)
            
            if max_dim > max_dimension_before_numpy:
                # Calculate scale factor
                scale = max_dimension_before_numpy / max_dim
                new_w = int(pil_w * scale)
                new_h = int(pil_h * scale)
                print(f"WARNING: Page {page_number} is very large ({pil_w}x{pil_h}), downscaling to {new_w}x{new_h} before processing")
                # Downscale using high-quality resampling
                pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # Convert PIL to NumPy array (RGB format)
            # Use asarray to avoid copy if possible, then convert BGR to RGB efficiently
            try:
                orig_cv_rgb = np.asarray(pil_img)
            except MemoryError as mem_error:
                # If still fails, try even smaller size
                print(f"WARNING: MemoryError converting page {page_number} to numpy, trying smaller size. Error: {mem_error}")
                # Try 50% of current size
                new_w = int(pil_img.size[0] * 0.5)
                new_h = int(pil_img.size[1] * 0.5)
                pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                orig_cv_rgb = np.asarray(pil_img)
            
            if len(orig_cv_rgb.shape) == 3 and orig_cv_rgb.shape[2] == 3:
                # Convert RGB to BGR for OpenCV (creates a view, not a copy)
                orig_cv = orig_cv_rgb[:, :, ::-1]
            else:
                # Handle RGBA or grayscale
                if len(orig_cv_rgb.shape) == 3 and orig_cv_rgb.shape[2] == 4:
                    # RGBA: convert to RGB first
                    pil_img = pil_img.convert('RGB')
                    try:
                        orig_cv_rgb = np.asarray(pil_img)
                    except MemoryError as mem_error:
                        # If still fails, try even smaller
                        print(f"WARNING: MemoryError converting RGBA page {page_number}, trying smaller size. Error: {mem_error}")
                        new_w = int(pil_img.size[0] * 0.5)
                        new_h = int(pil_img.size[1] * 0.5)
                        pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                        orig_cv_rgb = np.asarray(pil_img)
                orig_cv = orig_cv_rgb[:, :, ::-1] if len(orig_cv_rgb.shape) == 3 else orig_cv_rgb
            
            orig_h, orig_w, _ = orig_cv.shape
            content_h = orig_h
            # Reduced min_page_height for answer pages to minimize extra space
            min_page_height = MIN_PAGE_HEIGHT  # Reduced from 3500
            
            # Explicitly delete pix and img_bytes to free memory immediately
            del pix, img_bytes

            # Extended canvas: [left margin][answer][right margin]
            left_width = int(SIDE_MARGIN_RATIO * orig_w)
            right_width = int(SIDE_MARGIN_RATIO * orig_w)
            new_w = left_width + orig_w + right_width
            margin = int(MARGIN_RATIO * orig_w)
            # Changed from vertical centering to top-aligned with small margin
            # This reduces top/bottom space while keeping suggestions/annotations working
            y_offset = margin  # Changed from (h - content_h) // 2
            
            # Ensure canvas height is sufficient: must accommodate y_offset + content_h
            # For very large pages (like test 5), ensure we have enough space
            # Calculate required height: margin at top + content height + small bottom margin
            required_h = y_offset + content_h + margin  # Add bottom margin for safety
            h = max(orig_h, min_page_height, required_h)  # Ensure h >= required_h

            cv_img = np.full((h, new_w, 3), 255, dtype=np.uint8)
            # Place answer at top with small margin
            # Canvas is guaranteed to be tall enough, so this will always fit
            cv_img[
                y_offset:y_offset + content_h,
                left_width:left_width + orig_w,
                :
            ] = orig_cv

            # Track all comment boxes for collision detection
            comment_boxes: List[Tuple[int, int, int, int]] = []

            # Left-side padding area for improvement suggestions
            suggestion_x1 = margin
            suggestion_x2 = left_width - margin
            suggestion_y = margin

            # Right-side padding area for error/issue annotations
            comment_x1 = left_width + orig_w
            comment_x2 = new_w - margin
            comment_x = comment_x1 + margin
            comment_y = margin

            # Get suggestions for this page
            page_suggestion_data = None
            for ps in page_suggestions:
                if ps.get("page") == page_number:
                    page_suggestion_data = ps
                    break

            page_ocr = ocr_pages_by_num.get(page_number)
            if not page_ocr:
                # Check if image is too large BEFORE color conversion
                max_dimension = MAX_DIMENSION_BEFORE_RESIZE
                h_img, w_img = cv_img.shape[:2]
                
                # Downscale BEFORE color conversion to reduce memory pressure
                if max(h_img, w_img) > max_dimension:
                    scale = max_dimension / max(h_img, w_img)
                    new_w = int(w_img * scale)
                    new_h = int(h_img * scale)
                    
                    # Check available memory before resize
                    available_memory = _get_available_memory_mb()
                    # Estimate memory needed for resize (rough estimate: 3x the output size)
                    estimated_mb = (new_w * new_h * 3 * 3) / (1024 * 1024)  # 3 channels, 3 bytes per pixel
                    
                    # Use more memory-efficient interpolation if memory is low
                    interpolation = cv2.INTER_LINEAR  # More memory-efficient than INTER_LANCZOS4
                    if available_memory and available_memory < LOW_MEMORY_MB:
                        interpolation = cv2.INTER_AREA  # Most memory-efficient
                    
                    try:
                        cv_img = cv2.resize(cv_img, (new_w, new_h), interpolation=interpolation)
                    except cv2.error as resize_error:
                        # If resize fails due to memory, try even smaller size
                        print(f"WARNING: Resize failed for page {page_number}, trying smaller size. Error: {resize_error}")
                        # Try half the target size
                        new_w = int(new_w * 0.5)
                        new_h = int(new_h * 0.5)
                        try:
                            cv_img = cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                        except cv2.error as resize_error2:
                            # Last resort: use PIL for resize (more memory-efficient for very large images)
                            print(f"WARNING: OpenCV resize failed again, using PIL resize for page {page_number}")
                            pil_temp = Image.fromarray(cv_img[:, :, ::-1])  # BGR to RGB
                            pil_temp = pil_temp.resize((new_w, new_h), Image.Resampling.LANCZOS)
                            cv_img = np.array(pil_temp)[:, :, ::-1]  # RGB back to BGR
                
                # Convert BGR to RGB efficiently using cv2.cvtColor (more memory efficient)
                try:
                    cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                except cv2.error as color_error:
                    # Fallback: use array slicing if cvtColor fails
                    print(f"WARNING: cvtColor failed for page {page_number}, using array slicing. Error: {color_error}")
                    cv_img_rgb = cv_img[:, :, ::-1]
                
                pil_result = Image.fromarray(cv_img_rgb)
                annotated_pages.append(pil_result)
                # Free memory immediately
                del cv_img_rgb, orig_cv, pil_img, pil_result
                if 'cv_img' in locals():
                    del cv_img
                gc.collect()
                continue 

            # Draw config based on page size (bigger fonts + thicker lines)
            font_scale = max(0.9, min(orig_w, h) / 1200.0)
            text_thickness = 3
            box_thickness = 5
            line_height = int(32 * font_scale)
            suggestion_max_width = suggestion_x2 - suggestion_x1 - 10
            comment_max_width = comment_x2 - comment_x - 10

            # RENDER IMPROVEMENT SUGGESTIONS ON LEFT MARGIN
            BLUE = COLOR_SUGGESTION_BGR  # Deep sky blue color in BGR
            if page_suggestion_data:
                suggestions = page_suggestion_data.get("suggestions", [])

                # Title
                title_text = f"Page {page_number} - Suggestions:"
                cv2.putText(
                    cv_img,
                    title_text,
                    (suggestion_x1, suggestion_y),
                    font_face,
                    font_scale * 0.9,
                    BLUE,
                    text_thickness + 1,
                    cv2.LINE_AA,
                )
                suggestion_y += int(line_height * 1.5)

                # Draw each suggestion as a numbered bullet with blue box
                for idx, suggestion in enumerate(suggestions[:MAX_SUGGESTIONS_PER_PAGE], 1):  # Max suggestions per page
                    if isinstance(suggestion, dict):
                        suggestion_text = str(suggestion.get("suggestion", "")).strip()
                        suggestion_anchor = str(suggestion.get("anchor_quote", "")).strip()
                    else:
                        suggestion_text = str(suggestion).strip()
                        suggestion_anchor = ""

                    if not suggestion_text:
                        continue

                    bullet = f"{idx}. {suggestion_text}"
                    wrapped_lines = _wrap_text_cv2(
                        bullet, suggestion_max_width, font_face, font_scale * 0.85, text_thickness
                    )

                    # Calculate box height for this suggestion
                    box_start_y = suggestion_y - int(line_height * 0.8)
                    box_height = len(wrapped_lines) * int(line_height * 1.2) + int(line_height * 0.4)

                    # Draw blue box around suggestion
                    suggestion_box = (suggestion_x1 - 5, box_start_y, suggestion_x2 + 5, box_start_y + box_height)
                    cv2.rectangle(
                        cv_img,
                        (suggestion_box[0], suggestion_box[1]),
                        (suggestion_box[2], suggestion_box[3]),
                        BLUE,
                        3,  # Box thickness
                    )
                    
                    # Add suggestion box to collision detection list
                    comment_boxes.append(suggestion_box)

                    for line in wrapped_lines:
                        cv2.putText(
                            cv_img,
                            line,
                            (suggestion_x1, suggestion_y),
                            font_face,
                            font_scale * 0.85,
                            BLUE,
                            text_thickness,
                            cv2.LINE_AA,
                        )
                        suggestion_y += int(line_height * 1.2)
                    suggestion_y += int(line_height * 1.2)  # Increased gap between suggestion boxes

                    # Match left-side suggestion anchor to answer text and draw connector to the suggestion box.
                    if suggestion_anchor and page_ocr:
                        matched = _find_anchor_rect_on_page(
                            page_ocr=page_ocr,
                            anchor_quote=suggestion_anchor,
                            w=orig_w,
                            h=orig_h,
                            already_found=[],
                        )
                        if matched:
                            m_x1 = matched[0] + left_width
                            m_y1 = matched[1] + y_offset
                            m_x2 = matched[2] + left_width
                            m_y2 = matched[3] + y_offset

                            # Highlight the matched text span in blue for visible linking.
                            cv2.rectangle(cv_img, (m_x1, m_y1), (m_x2, m_y2), BLUE, 3)

                            text_y = (m_y1 + m_y2) // 2
                            box_target_x = suggestion_box[2]
                            box_target_y = (suggestion_box[1] + suggestion_box[3]) // 2
                            text_start_x = max(left_width + 4, m_x1 - 8)
                            cv2.line(
                                cv_img,
                                (text_start_x, text_y),
                                (box_target_x, box_target_y),
                                BLUE,
                                3,
                                cv2.LINE_AA,
                            )

            suggestion_end_y = max(suggestion_y, margin)

            # Filter annotations relevant to this page
            page_anns: List[Dict[str, Any]] = []
            for ann in annotations or []:
                atype = (ann.get("type") or "").lower()

                if atype == "repetition":
                    rep_page = ann.get("repeated_page")
                    if rep_page == page_number:
                        page_anns.append(ann)
                    continue

                apage = ann.get("page")
                if apage == page_number:
                    page_anns.append(ann)

            # Track which sections are problematic on this page
            bad_section_ids: set[str] = set()
            for ann in page_anns:
                atype = (ann.get("type") or "").lower()
                if atype == "heading_issue":
                    sid = ann.get("section_id")
                    if sid:
                        bad_section_ids.add(str(sid))
                elif atype == "repetition":
                    sid = ann.get("section_id")
                    if sid:
                        bad_section_ids.add(str(sid))
                elif atype == "introduction_comment":
                    sid = ann.get("target_section_id")
                    if sid:
                        bad_section_ids.add(str(sid))

            # Helper function to shift coordinates for answer page (now in center)
            def shift_rect(rect: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
                """Shift rectangle coordinates to account for left margin and vertical centering."""
                x1, y1, x2, y2 = rect
                return (x1 + left_width, y1 + y_offset, x2 + left_width, y2 + y_offset)

            def _rects_overlap(
                a: Tuple[int, int, int, int],
                b: Tuple[int, int, int, int],
                pad: int = 0,
            ) -> bool:
                """
                Check if two rectangles overlap with optional padding.
                
                Args:
                    a: First rectangle (x1, y1, x2, y2)
                    b: Second rectangle (x1, y1, x2, y2)
                    pad: Optional padding to add to overlap check
                
                Returns:
                    True if rectangles overlap, False otherwise
                """
                ax1, ay1, ax2, ay2 = a
                bx1, by1, bx2, by2 = b
                return not (
                    ax2 + pad < bx1
                    or ax1 - pad > bx2
                    or ay2 + pad < by1
                    or ay1 - pad > by2
                )

            def _find_box_in_column(
                x1: int,
                x2: int,
                start_y: int,
                min_y: int,
                max_y: int,
                height: int,
            ) -> Optional[Tuple[int, int, int, int]]:
                """
                Find non-overlapping position for a comment box in a column.
                
                Args:
                    x1: Left edge of column
                    x2: Right edge of column
                    start_y: Preferred starting Y position
                    min_y: Minimum allowed Y position
                    max_y: Maximum allowed Y position
                    height: Height of the box to place
                
                Returns:
                    Tuple of (x1, y1, x2, y2) if position found, None otherwise
                """
                y = max(start_y, min_y)
                while y + height <= max_y:
                    candidate = (x1, y, x2, y + height)
                    overlapping = [
                        rect for rect in comment_boxes
                        if _rects_overlap(candidate, rect, pad=4)
                    ]
                    if not overlapping:
                        return candidate
                    y = max(rect[3] for rect in overlapping) + int(line_height * 0.6)
                return None

            def add_side_comment(
                header: str,
                text: str,
                draw_box: bool = True,
            ) -> Tuple[Tuple[int, int, int, int], bool]:
                """
                Add a side comment with collision detection.
                
                Args:
                    header: Header text for the comment
                    text: Body text for the comment
                    draw_box: Whether to draw red box around comment (default: True)
                
                Returns:
                    Tuple of ((x1, y1, x2, y2), is_right) where:
                    - (x1, y1, x2, y2) is the comment box coordinates
                    - is_right is True if box is on right side, False if on left
                """
                nonlocal comment_y
                
                def build_lines(
                    max_width: int,
                ) -> Tuple[List[str], List[Tuple[str, Tuple[int, int, int]]], int]:
                    """Build header lines + colored body lines, return total height needed."""
                    header_lines = _wrap_text_cv2(
                        header, max_width - 20, font_face, font_scale * 0.95, text_thickness
                    )
                    body_lines: List[Tuple[str, Tuple[int, int, int]]] = []
                    if text:
                        # Preserve explicit newlines (e.g., "Heading:" / "Rephrased:" / "- comment")
                        # and allow styling specific lines without affecting annotation boxes.
                        raw_lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
                        for raw in raw_lines:
                            is_rephrased = raw.lower().startswith("rephrased:")
                            color = GREEN if is_rephrased else RED
                            wrapped = _wrap_text_cv2(
                                raw, max_width - 20, font_face, font_scale * 0.85, text_thickness
                            )
                            for wline in wrapped:
                                body_lines.append((wline, color))
                    height = int(
                        len(header_lines) * line_height * 1.2
                        + len(body_lines) * line_height * 1.0
                        + line_height * 1.4
                    )
                    return header_lines, body_lines, height

                # Try right side first
                right_x1 = comment_x - 10
                right_x2 = comment_x2 - 5
                right_min_y = margin
                right_max_y = h - margin
                preferred_y = max(comment_y - int(line_height * 1.0), right_min_y)

                header_lines, body_lines, box_height = build_lines(comment_max_width)
                box = _find_box_in_column(
                    right_x1,
                    right_x2,
                    preferred_y,
                    right_min_y,
                    right_max_y,
                    box_height,
                )
                
                # If right side is full, try left side
                if not box:
                    box = _find_box_in_column(
                        right_x1,
                        right_x2,
                        right_min_y,
                        right_min_y,
                        right_max_y,
                        box_height,
                    )
                
                if not box:
                    header_lines, body_lines, box_height = build_lines(suggestion_max_width)
                    left_x1 = suggestion_x1 - 5
                    left_x2 = suggestion_x2 + 5
                    left_min_y = max(suggestion_end_y, margin)
                    left_max_y = h - margin
                    box = _find_box_in_column(
                        left_x1,
                        left_x2,
                        left_min_y,
                        left_min_y,
                        left_max_y,
                        box_height,
                    )

                # Last resort: place at bottom
                if not box:
                    y1 = max(right_min_y, right_max_y - box_height)
                    box = (right_x1, y1, right_x2, min(right_max_y, y1 + box_height))

                # Track this box for collision detection
                comment_boxes.append(box)
                box_x1, box_y1, box_x2, box_y2 = box

                if draw_box:
                    # Draw red box around the entire comment
                    cv2.rectangle(
                        cv_img,
                        (box_x1, box_y1),
                        (box_x2, box_y2),
                        RED,
                        3,  # Box thickness
                    )

                text_x = box_x1 + 10
                text_y = box_y1 + int(line_height * 1.0)

                # Header (red, bold-ish)
                for line in header_lines:
                    if text_y > box_y2 - int(line_height * 0.5):
                        break
                    cv2.putText(
                        cv_img,
                        line,
                        (text_x, text_y),
                        font_face,
                        font_scale * 0.95,
                        RED,
                        text_thickness,
                        cv2.LINE_AA,
                    )
                    text_y += int(line_height * 1.2)

                # Body text (red, except Rephrased:* lines in green)
                for line, color in body_lines:
                    if text_y > box_y2 - int(line_height * 0.5):
                        break
                    cv2.putText(
                        cv_img,
                        line,
                        (text_x, text_y),
                        font_face,
                        font_scale * 0.85,
                        color,
                        text_thickness,
                        cv2.LINE_AA,
                    )
                    text_y += int(line_height * 1.0)

                comment_y = max(comment_y, box_y2 + int(line_height * 0.6))
                is_right = box_x1 >= comment_x - 5
                return (box_x1, box_y1, box_x2, box_y2), is_right

            def draw_correction_near_box(
                rect: Tuple[int, int, int, int],
                correction: str,
            ):
                if not correction:
                    return
                x1, y1, x2, y2 = rect
                cx = x1
                cy = max(15, y1 - 8)  # Increased minimum to avoid negative coords

                size, _ = cv2.getTextSize(
                    correction, font_face, font_scale * 0.85, text_thickness
                )
                tx, ty = size
                canvas_max_x = left_width + orig_w - 5
                bg_x2 = min(canvas_max_x, cx + tx + 8)
                bg_y2 = max(5, cy - ty - 6)  # Ensure >= 5 instead of 0

                # Background white box for correction
                cv2.rectangle(
                    cv_img,
                    (cx - 2, bg_y2),
                    (bg_x2, cy + 4),
                    (255, 255, 255),
                    thickness=-1,
                )
                cv2.putText(
                    cv_img,
                    correction,
                    (cx, cy),
                    font_face,
                    font_scale * 0.85,
                    RED,
                    text_thickness,
                    cv2.LINE_AA,
                )

            def draw_connector(
                rect: Tuple[int, int, int, int],
                target_x: int,
                target_y_center: int,
            ):
                """
                Draw a red line from the annotation box to the comment box.
                
                Args:
                    rect: Annotation rectangle (x1, y1, x2, y2)
                    target_x: X position of target comment box
                    target_y_center: Y center position of target comment box
                """
                x1, y1, x2, y2 = rect
                rect_center_y = (y1 + y2) // 2
                rect_right_x = min(left_width + orig_w - int(0.02 * orig_w), x2 + 10)
                cv2.line(
                    cv_img,
                    (rect_right_x, rect_center_y),
                    (target_x, target_y_center),
                    RED,
                    3,
                )


        
            def draw_tick_shape(
                img,
                x: int,
                y: int,
                color: Tuple[int, int, int],
                thickness: int,
            ):
                """
                Draw a simple tick (checkmark) using two anti-aliased lines.
                The size scales with font_scale so it looks consistent.
                (x, y) is the 'start' of the tick.
                """
                # Scale lengths based on font_scale so it looks good on large/small pages
                down_dx = int( 20 * font_scale)
                down_dy = int( 2 * 16 * font_scale)
                up_dx = int(10 * 26 * font_scale)
                up_dy = int(3 * 8 * font_scale)

                # Points of the tick
                p1 = (x, y)  # start
                p2 = (x + down_dx, y + down_dy)  # bottom of the tick
                p3 = (x + up_dx, y - up_dy)      # upper end

                # Draw two lines to make a checkmark
                cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)
                cv2.line(img, p2, p3, color, thickness, cv2.LINE_AA)



            def draw_tick_at_rect(rect: Tuple[int, int, int, int]):
                """
                Draw a red tick near the top-left of the given rectangle using two lines.
                """
                x1, y1, x2, y2 = rect

                # Choose a base point slightly inside the top-left of the rect
                tick_x = max(5, x1 + 10)
                tick_y = max(35, y1 + int(0.05 * h))  # Increased minimum to 35 to account for upward stroke

                # Use our line-based tick drawer
                draw_tick_shape(
                    img=cv_img,
                    x=tick_x,
                    y=tick_y,
                    color=RED,
                    thickness=text_thickness + 1,
                )


            # Precompute section regions for this page (for intro + repetition + ticks)
            section_regions: Dict[str, Tuple[int, int, int, int]] = {}
            for sec in sections:
                pages = sec.get("page_numbers") or []
                if page_number not in pages:
                    continue
                region = _compute_section_region_on_page(
                    section=sec,
                    page_number=page_number,
                    ocr_data=ocr_data,
                    orig_w=orig_w,
                    orig_h=orig_h,
                    all_sections=sections,
                )
                if region:
                    sid = str(sec.get("id") or sec.get("section_id") or sec.get("title"))
                    if sid:
                        section_regions[sid] = region

            # Track rectangles already found on this page to handle duplicates
            found_rects: List[Tuple[int, int, int, int]] = []
            
            # Track skipped annotations for debugging
            skipped_annotations: List[Dict[str, Any]] = []

            # Draw all negative/problem annotations in red
            for ann in page_anns:
                atype = (ann.get("type") or "").lower()
                rubric_point = ann.get("rubric_point") or ""
                target_text = ann.get("target_word_or_sentence") or ""
                context_before = ann.get("context_before") or ""
                context_after = ann.get("context_after") or ""
                anchor_quote = ann.get("anchor_quote") or ""
                correction = ann.get("correction") or ""
                comment = ann.get("comment") or ""

                # 1) Introduction comment – big box over intro + right-side comment
                if atype == "introduction_comment":
                    # Try to find intro section region first
                    intro_sections = [s for s in sections if "introduction" in (s.get("title") or "").lower()]
                    intro_rect = None
                    if intro_sections:
                        intro_sec = intro_sections[0]
                        sid = str(intro_sec.get("id") or intro_sec.get("section_id") or intro_sec.get("title"))
                        intro_rect = section_regions.get(sid)

                    if not intro_rect and anchor_quote:
                        intro_rect = _find_anchor_rect_on_page(
                            page_ocr=page_ocr,
                            anchor_quote=anchor_quote,
                            w=orig_w,
                            h=orig_h,
                            already_found=found_rects,
                        )

                    # Fallback 1: Try target_text if section region not found
                    if not intro_rect and target_text:
                        rect = _find_annotation_rect_with_context(
                            page_ocr=page_ocr,
                            target_text=target_text,
                            context_before=context_before,
                            context_after=context_after,
                            w=orig_w,
                            h=orig_h,
                            already_found=found_rects,
                        )
                        if not rect:
                            # Fallback 2: Try word/line matching
                            rect = _find_word_or_line_rect(
                                page_ocr=page_ocr,
                                target_text=target_text,
                                w=orig_w,
                                h=orig_h,
                            )
                        intro_rect = rect

                    # Fallback 3: Default rect
                    if not intro_rect:
                        intro_rect = (0, 0, orig_w - 1, min(orig_h - 1, int(0.22 * content_h)))

                    x1, y1, x2, y2 = intro_rect
                    x1 = 0
                    x2 = orig_w - 1
                    pad_top = int(0.03 * content_h)
                    pad_bottom = int(0.08 * content_h)
                    y1 = max(0, y1 - pad_top)
                    y2 = min(content_h - 1, y2 + pad_bottom)

                    shifted = shift_rect((x1, y1, x2, y2))
                    cv2.rectangle(cv_img, (shifted[0], shifted[1]), (shifted[2], shifted[3]), RED, box_thickness)

                    # NEW: Remove brackets, cleaner format
                    header = f"Introduction - {rubric_point}".strip()
                    body = comment
                    comment_box, comment_on_right = add_side_comment(header, body)
                    if comment_on_right:
                        draw_connector(
                            shifted,
                            comment_box[0],
                            (comment_box[1] + comment_box[3]) // 2,
                        )
                    found_rects.append((x1, y1, x2, y2))  # Store original for duplicate detection

                # 2) Heading issue – box on heading + right-side comment
                elif atype == "heading_issue":
                    sentiment = (ann.get("sentiment") or "").lower()
                    if sentiment not in ("negative", "weak", "problematic"):
                        skipped_annotations.append({
                            "type": atype,
                            "reason": f"Sentiment '{sentiment}' not in negative/weak/problematic",
                            "target": target_text[:50],
                            "page": page_number
                        })
                        continue

                    if not target_text and not anchor_quote:
                        skipped_annotations.append({
                            "type": atype,
                            "reason": "Missing target_text and anchor_quote",
                            "page": page_number
                        })
                        continue

                    rect = None
                    if anchor_quote:
                        rect = _find_anchor_rect_on_page(
                            page_ocr=page_ocr,
                            anchor_quote=anchor_quote,
                            w=orig_w,
                            h=orig_h,
                            already_found=found_rects,
                        )
                    if not rect and target_text:
                        # Use precise word-level matching for headings (they're short)
                        rect = _find_precise_word_rect_with_context(
                            page_ocr=page_ocr,
                            target_text=target_text,
                            context_before=context_before,
                            context_after=context_after,
                            w=orig_w,
                            h=orig_h,
                            already_found=found_rects,
                        )
                    
                    # Always show annotation comment, even if text not found
                    # If rect found: draw box + connector
                    # If rect not found: just show comment on side (no box/connector)
                    header = f"Heading Issue - {rubric_point}".strip()
                    body_lines = [f"Heading: {target_text}"]
                    if correction:
                        body_lines.append(f"Rephrased: {correction}")
                    body_lines.append(f"- {comment}")
                    body = "\n".join(body_lines)
                    comment_box, comment_on_right = add_side_comment(header, body)
                    
                    if rect:
                        # Text found: draw box and connector
                        shifted = shift_rect(rect)
                        x1, y1, x2, y2 = shifted
                        cv2.rectangle(cv_img, (x1, y1), (x2, y2), RED, box_thickness)
                        if comment_on_right:
                            draw_connector(
                                shifted,
                                comment_box[0],
                                (comment_box[1] + comment_box[3]) // 2,
                            )
                        found_rects.append(rect)
                    # If rect not found, we still show the comment (already added above)

                # 3) Factual error – precise box on error phrase + right-side comment
                elif atype == "factual_error":
                    if not target_text and not anchor_quote:
                        skipped_annotations.append({
                            "type": atype,
                            "reason": "Missing target_text and anchor_quote",
                            "page": page_number
                        })
                        continue

                    rect = None
                    if anchor_quote:
                        rect = _find_anchor_rect_on_page(
                            page_ocr=page_ocr,
                            anchor_quote=anchor_quote,
                            w=orig_w,
                            h=orig_h,
                            already_found=found_rects,
                        )
                    if not rect and target_text:
                        # ALWAYS use precise word-level matching for factual errors
                        rect = _find_precise_word_rect_with_context(
                            page_ocr=page_ocr,
                            target_text=target_text,
                            context_before=context_before,
                            context_after=context_after,
                            w=orig_w,
                            h=orig_h,
                            already_found=found_rects,
                        )

                    # Always show annotation comment, even if text not found
                    # If rect found: draw box + connector
                    # If rect not found: just show comment on side (no box/connector)
                    header = f"Factual Error - {rubric_point}".strip()
                    body = comment
                    if correction:
                        body = f"Correction: {correction}\n{body}"
                    comment_box, comment_on_right = add_side_comment(header, body)
                    
                    if rect:
                        # Text found: draw box and connector
                        shifted = shift_rect(rect)
                        x1, y1, x2, y2 = shifted
                        cv2.rectangle(cv_img, (x1, y1), (x2, y2), RED, box_thickness)
                        if comment_on_right:
                            draw_connector(
                                shifted,
                                comment_box[0],
                                (comment_box[1] + comment_box[3]) // 2,
                            )
                        found_rects.append(rect)
                    # If rect not found, we still show the comment (already added above)

                # 4) Grammar & language – small box + inline correction
                elif atype == "grammar_language":
                    if not target_text and not anchor_quote:
                        skipped_annotations.append({
                            "type": atype,
                            "reason": "Missing target_text and anchor_quote",
                            "page": page_number
                        })
                        continue

                    rect = None
                    if anchor_quote:
                        rect = _find_anchor_rect_on_page(
                            page_ocr=page_ocr,
                            anchor_quote=anchor_quote,
                            w=orig_w,
                            h=orig_h,
                            already_found=found_rects,
                        )
                    if not rect and target_text:
                        # ALWAYS use precise word-level matching for spelling errors
                        rect = _find_precise_word_rect_with_context(
                            page_ocr=page_ocr,
                            target_text=target_text,
                            context_before=context_before,
                            context_after=context_after,
                            w=orig_w,
                            h=orig_h,
                            already_found=found_rects,
                        )
                    
                    # For Grammar/Language: only show box on text (no side comment)
                    # If rect found: draw box + inline correction only
                    # If rect not found: skip (don't show side comment)
                    if rect:
                        # Text found: draw box and inline correction only (no side comment)
                        shifted = shift_rect(rect)
                        x1, y1, x2, y2 = shifted
                        cv2.rectangle(cv_img, (x1, y1), (x2, y2), RED, box_thickness)
                        draw_correction_near_box(shifted, correction)
                        found_rects.append(rect)
                    # If rect not found, skip this annotation (no side comment for grammar/language)

                # 5) Repetition – box on repeated content + text "repeated"
                elif atype == "repetition":
                    if not target_text:
                        skipped_annotations.append({
                            "type": atype,
                            "reason": "Missing target_text",
                            "page": page_number
                        })
                        continue

                    # Use context-aware matching
                    rect = _find_annotation_rect_with_context(
                        page_ocr=page_ocr,
                        target_text=target_text,
                        context_before=context_before,
                        context_after=context_after,
                        w=orig_w,
                        h=orig_h,
                        already_found=found_rects,
                    )
                    
                    # Always show annotation, even if text not found
                    # If rect found: draw box + "repeated" label
                    # If rect not found: show as side comment
                    if rect:
                        # Text found: draw box and "repeated" label
                        shifted = shift_rect(rect)
                        x1, y1, x2, y2 = shifted
                        cv2.rectangle(cv_img, (x1, y1), (x2, y2), RED, box_thickness)

                        rep_text = comment if comment else "repeated"
                        size, _ = cv2.getTextSize(
                            rep_text, font_face, font_scale * 0.9, text_thickness
                        )
                        tx, ty = size
                        tx1 = x1
                        ty1 = max(0, y1 - ty - 6)
                        # Keep text within answer area bounds
                        answer_area_right = left_width + orig_w - 5
                        tx2 = min(answer_area_right, x1 + tx + 8)
                        ty2 = y1

                        cv2.rectangle(
                            cv_img,
                            (tx1 - 2, ty1),
                            (tx2, ty2),
                            (255, 255, 255),
                            thickness=-1,
                        )
                        cv2.putText(
                            cv_img,
                            rep_text,
                            (tx1, y1 - 4),
                            font_face,
                            font_scale * 0.9,
                            RED,
                            text_thickness,
                            cv2.LINE_AA,
                        )
                        found_rects.append(rect)
                    else:
                        # Text not found: show as side comment
                        header = f"Repetition - {rubric_point}".strip()
                        body_lines = [f"Text: {target_text}"]
                        if comment:
                            body_lines.append(f"- {comment}")
                        body = "\n".join(body_lines)
                        add_side_comment(header, body)
                    found_rects.append(rect)

            # NEW: Add heading comments only on the first page of each section
            for sec in sections:
                pages = sec.get("page_numbers") or []
                if not pages:
                    continue
                # Only show heading on the first page of the section.
                # Safely extract numeric page numbers (robust to 0-based vs 1-based).
                try:
                    numeric_pages = [int(p) for p in pages if isinstance(p, (int, float))]
                    if not numeric_pages:
                        # Try converting string page numbers
                        numeric_pages = [int(p) for p in pages if str(p).isdigit()]
                    if not numeric_pages:
                        continue  # Skip if no valid page numbers found
                    # If these look 0-based (common in some pipelines), shift to 1-based.
                    if min(numeric_pages) == 0:
                        numeric_pages = [p + 1 for p in numeric_pages]
                    first_page = min(numeric_pages)
                except (ValueError, TypeError) as e:
                    # If page number extraction fails, skip this section
                    print(f"WARNING: Failed to extract page numbers for section '{sec.get('title', 'UNKNOWN')}': {e}")
                    continue
                
                if page_number != first_page:
                    continue

                # For heading labels:
                # - exact_ocr_heading: ORIGINAL heading text from OCR (for matching + display)
                # - rephrased_heading: improved/natural heading suggestion (display in green)
                title_text = (sec.get("title") or "").strip()
                exact_heading = (sec.get("exact_ocr_heading") or "").strip()
                rephrased_heading = (sec.get("rephrased_heading") or "").strip()

                display_heading = exact_heading or title_text
                if not display_heading:
                    continue
                # NOTE: We will keep using the raw heading for bbox matching (to avoid breaking boxes),
                # but we can safely clean the DISPLAY text to remove leading enumerations like "e)".
                heading_norm = display_heading.strip().lower()
                if "introduction" in heading_norm or "conclusion" in heading_norm:
                    continue

                def _strip_heading_enumeration_prefix(s: str) -> str:
                    """
                    Remove leading enumeration/bullets like:
                    '(2)', '2.', '2)', 'a)', 'b.', 'i)', '-', '•'
                    from rephrased headings only (keep original OCR heading as-is).
                    """
                    t = (s or "").strip()
                    # Strip repeatedly because headings can have stacked prefixes like:
                    # "I (ii) ..." or "(V) ..." or "e) (2) ..."
                    while True:
                        before = t
                        # common bullets
                        t = re.sub(r"^\s*[-•]+\s*", "", t)
                        # (2) / (12)
                        t = re.sub(r"^\s*\(\s*\d+\s*\)\s*", "", t)
                        # (V) / (iv) etc.
                        t = re.sub(r"^\s*\(\s*([ivxlcdm]+)\s*\)\s*", "", t, flags=re.IGNORECASE)
                        # 2. / 2) / 12.
                        t = re.sub(r"^\s*\d+\s*[\.\)]\s*", "", t)
                        # a) / b. / A)
                        t = re.sub(r"^\s*[A-Za-z]\s*[\.\)]\s*", "", t)
                        # roman numerals i) ii) IV.
                        t = re.sub(r"^\s*([ivxlcdm]+)\s*[\.\)]\s*", "", t, flags=re.IGNORECASE)
                        # roman numeral WITHOUT punctuation, e.g. "I (ii) Heading ..."
                        t = re.sub(r"^\s*([ivxlcdm]+)\s+(?=\()", "", t, flags=re.IGNORECASE)
                        # Trim again after stripping
                        t = t.strip()
                        if t == before.strip():
                            break
                    return t.strip()

                # For detected headings:
                # - RED header shows the ORIGINAL OCR heading (or title fallback if OCR missing)
                # - GREEN body shows exactly one line: "Rephrased: <rephrased_heading>"
                #   (we do NOT include sec["comment"] here because it duplicates rephrasing and can add bullets like (2)/a)/b)).
                display_heading_clean = _strip_heading_enumeration_prefix(display_heading)
                header = f"Heading: {display_heading_clean or display_heading}".strip()

                body = ""
                clean_rephrased = _strip_heading_enumeration_prefix(rephrased_heading)
                if clean_rephrased and clean_rephrased.strip().lower() != display_heading.strip().lower():
                    body = f"Rephrased: {clean_rephrased}"

                comment_box, comment_on_right = add_side_comment(header, body)

                heading_bbox = _find_heading_bbox_on_page(exact_heading or display_heading, page_ocr)
                if heading_bbox and comment_on_right:
                    hx1, hy1, hx2, hy2 = _bbox_to_rect(heading_bbox, pad=4, w=orig_w, h=orig_h)
                    shifted = shift_rect((hx1, hy1, hx2, hy2))
                    draw_connector(
                        shifted,
                        comment_box[0],
                        (comment_box[1] + comment_box[3]) // 2,
                    )

            # After drawing negative annotations, add ticks for ALL headings of good sections
            for sec in sections:
                pages = sec.get("page_numbers") or []
                if page_number not in pages:
                    continue

                sid = str(sec.get("id") or sec.get("section_id") or sec.get("title") or "")
                # Always draw ticks for headings/subheadings regardless of any
                # annotations marking the section as 'bad'. Only skip if we
                # cannot derive an identifier/title for the section.
                if not sid:
                    continue

                tick_drawn = False

                # Try 1: Use exact_ocr_heading if available
                exact_heading = sec.get("exact_ocr_heading") or ""
                if exact_heading:
                    heading_bbox = _find_heading_bbox_on_page(exact_heading, page_ocr)
                    if heading_bbox:
                        hx1, hy1, hx2, hy2 = _bbox_to_rect(heading_bbox, pad=4, w=orig_w, h=orig_h)
                        shifted = shift_rect((hx1, hy1, hx2, hy2))
                        draw_tick_at_rect(shifted)
                        tick_drawn = True

                # Try 2: Fall back to title
                if not tick_drawn:
                    title = sec.get("title") or ""
                    if title and title != exact_heading:
                        heading_bbox = _find_heading_bbox_on_page(title, page_ocr)
                        if heading_bbox:
                            hx1, hy1, hx2, hy2 = _bbox_to_rect(heading_bbox, pad=4, w=orig_w, h=orig_h)
                            shifted = shift_rect((hx1, hy1, hx2, hy2))
                            draw_tick_at_rect(shifted)
                            tick_drawn = True

                # Try 3: If still not found, draw tick on section region (first part of section content)
                if not tick_drawn:
                    sec_region = section_regions.get(sid)
                    if sec_region:
                        # Draw tick at the top-left of the section region
                        shifted = shift_rect(sec_region)
                        draw_tick_at_rect(shifted)
                        tick_drawn = True

                # Try 4: Last resort - search for any line containing key words from title
                if not tick_drawn and page_ocr:
                    title = sec.get("title") or ""
                    title_words = set(title.lower().split()[:3])  # Use first 3 words
                    title_words.discard("introduction")
                    title_words.discard("conclusion")

                    if title_words:
                        for line in page_ocr.get("lines", []):
                            line_text = (line.get("text") or "").lower()
                            line_words = set(line_text.split())

                            # If at least 2 words match and line is short (likely a heading)
                            overlap = len(title_words & line_words)
                            if overlap >= min(2, len(title_words)) and len(line_text) < 100:
                                line_bbox = line.get("bbox")
                                if line_bbox:
                                    hx1, hy1, hx2, hy2 = _bbox_to_rect(line_bbox, pad=4, w=orig_w, h=orig_h)
                                    shifted = shift_rect((hx1, hy1, hx2, hy2))
                                    draw_tick_at_rect(shifted)
                                    tick_drawn = True
                                    break

            # Check if image is too large BEFORE color conversion to prevent MemoryError
            # Large images can cause MemoryError when converting colors or to PIL Image
            # 268MB allocation failure suggests image is ~9000x9000 pixels or larger
            max_dimension = MAX_DIMENSION_BEFORE_RESIZE  # Maximum dimension before downscaling
            h_img, w_img = cv_img.shape[:2]
            
            # Downscale BEFORE color conversion to reduce memory pressure
            if max(h_img, w_img) > max_dimension:
                # Calculate scale factor
                scale = max_dimension / max(h_img, w_img)
                new_w = int(w_img * scale)
                new_h = int(h_img * scale)
                
                # Check available memory before resize
                available_memory = _get_available_memory_mb()
                # Estimate memory needed for resize (rough estimate: 3x the output size)
                estimated_mb = (new_w * new_h * 3 * 3) / (1024 * 1024)  # 3 channels, 3 bytes per pixel
                
                # Use more memory-efficient interpolation if memory is low
                interpolation = cv2.INTER_LINEAR  # More memory-efficient than INTER_LANCZOS4
                if available_memory and available_memory < LOW_MEMORY_MB:
                    interpolation = cv2.INTER_AREA  # Most memory-efficient
                
                try:
                    # Downscale using memory-efficient interpolation
                    cv_img = cv2.resize(cv_img, (new_w, new_h), interpolation=interpolation)
                    # Update dimensions after resize
                    h_img, w_img = cv_img.shape[:2]
                except cv2.error as resize_error:
                    # If resize fails due to memory, try even smaller size
                    print(f"WARNING: Resize failed for page {page_number}, trying smaller size. Error: {resize_error}")
                    # Try half the target size
                    new_w = int(new_w * 0.5)
                    new_h = int(new_h * 0.5)
                    try:
                        cv_img = cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                        h_img, w_img = cv_img.shape[:2]
                    except cv2.error as resize_error2:
                        # Last resort: use PIL for resize (more memory-efficient for very large images)
                        print(f"WARNING: OpenCV resize failed again, using PIL resize for page {page_number}")
                        pil_temp = Image.fromarray(cv_img[:, :, ::-1])  # BGR to RGB
                        pil_temp = pil_temp.resize((new_w, new_h), Image.Resampling.LANCZOS)
                        cv_img = np.array(pil_temp)[:, :, ::-1]  # RGB back to BGR
                        h_img, w_img = cv_img.shape[:2]
            
            # Convert BGR to RGB efficiently using cv2.cvtColor (more memory efficient than slicing)
            # This avoids creating a copy of the entire array
            try:
                cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            except cv2.error as color_error:
                # Fallback: use array slicing if cvtColor fails
                print(f"WARNING: cvtColor failed for page {page_number}, using array slicing. Error: {color_error}")
                cv_img_rgb = cv_img[:, :, ::-1]
            
            # Final safety check: ensure image is small enough for PIL to handle
            # PIL can fail with MemoryError if image is too large (typically > 100MP or ~10,000x10,000)
            h_rgb, w_rgb = cv_img_rgb.shape[:2]
            total_pixels = h_rgb * w_rgb
            max_pil_pixels = MAX_PIL_PIXELS  # ~80MP limit for PIL (conservative)
            
            if total_pixels > max_pil_pixels:
                # Calculate scale factor to fit within PIL limit
                scale = (max_pil_pixels / total_pixels) ** 0.5
                new_w = int(w_rgb * scale)
                new_h = int(h_rgb * scale)
                print(f"WARNING: Image too large for PIL ({w_rgb}x{h_rgb}, ~{total_pixels/1e6:.1f}MP). "
                      f"Downscaling to {new_w}x{new_h} before PIL conversion.")
                
                # Use memory-efficient interpolation
                available_memory = _get_available_memory_mb()
                interpolation = cv2.INTER_AREA if (available_memory and available_memory < LOW_MEMORY_MB) else cv2.INTER_LINEAR
                
                try:
                    cv_img_rgb = cv2.resize(cv_img_rgb, (new_w, new_h), interpolation=interpolation)
                except Exception as resize_error:
                    # If resize fails, try even smaller
                    print(f"WARNING: Resize failed before PIL conversion, trying 50% size. Error: {resize_error}")
                    new_w = int(w_rgb * 0.5)
                    new_h = int(h_rgb * 0.5)
                    cv_img_rgb = cv2.resize(cv_img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Convert to PIL Image with error handling
            try:
                pil_result = Image.fromarray(cv_img_rgb)
            except MemoryError as pil_error:
                # If still fails, try even smaller size
                print(f"WARNING: MemoryError converting to PIL Image ({cv_img_rgb.shape[1]}x{cv_img_rgb.shape[0]}). "
                      f"Trying 50% size. Error: {pil_error}")
                h_rgb, w_rgb = cv_img_rgb.shape[:2]
                new_w = int(w_rgb * 0.5)
                new_h = int(h_rgb * 0.5)
                cv_img_rgb = cv2.resize(cv_img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
                pil_result = Image.fromarray(cv_img_rgb)
            
            # Log skipped annotations for this page (separate log file + lightweight console hint)
            _append_skipped_annotations_log(skipped_annotations, page_number=page_number)
            if skipped_annotations:
                print(f"  ⚠ Page {page_number}: Skipped {len(skipped_annotations)} annotations (see skipped_annotations_log.txt).")
            
            annotated_pages.append(pil_result)
            
            # Explicitly release memory after each page to prevent accumulation
            del pil_img, orig_cv, cv_img, cv_img_rgb, pil_result
            gc.collect()
            
            # Monitor memory usage periodically (every 5 pages)
            if (page_idx + 1) % 5 == 0:
                process_memory = _get_process_memory_mb()
                available_memory = _get_available_memory_mb()
                if process_memory and available_memory:
                    print(f"Memory status after page {page_number}: "
                          f"process={process_memory:.1f} MB, "
                          f"available={available_memory:.1f} MB")
                    # Warn if memory is getting low
                    if available_memory < LOW_MEMORY_WARN_MB:
                        print(f"WARNING: Low available memory ({available_memory:.1f} MB) "
                              f"after processing {page_number} pages")

        return annotated_pages
    finally:
        doc.close()  # Always close the document to release file handle
