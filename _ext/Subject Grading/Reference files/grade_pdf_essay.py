# grade_pdf_essay.py
#
# ESSAY pipeline (CSS English Essay) - STRICT RANGE MARKING:
#   - Structure assumption:
#       (1) Outline section first (expected)
#       (2) Essay body is mostly paragraphs; headings/markers may appear
#   - Marking is VERY strict:
#       - Even a very strong essay should land around 38-40/100 max
#   - DO NOT output exact marks, output mark ranges (e.g., "6-8").
#
# Outputs:
#   - JSON: structure + grading + annotations
#   - PDF: report pages + annotated essay pages
#
# Env (.env):
#   Grok_API=...
#   AZURE_ENDPOINT=...
#   AZURE_KEY=...
#
# Usage:
#   python3 grade_pdf_essay.py --pdf Essay.pdf --output-json essay_result.json --output-pdf essay_annotated.pdf

import argparse
import base64
import gc
import io
import json
import os
import random
import re
import time
from typing import Any, Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import requests
from dotenv import load_dotenv
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
import fitz  # PyMuPDF
from docx import Document
from PIL import Image, ImageDraw, ImageFont

# Import PDF compression function
try:
    from compressPdf import compress_pdf_if_needed
except (ImportError, ModuleNotFoundError):
    try:
        from .compressPdf import compress_pdf_if_needed
    except (ImportError, ModuleNotFoundError):
        # Fallback: try to load from file location
        try:
            import importlib.util
            current_dir = os.path.dirname(os.path.abspath(__file__))
            compress_pdf_path = os.path.join(current_dir, "compressPdf.py")
            if os.path.exists(compress_pdf_path):
                spec = importlib.util.spec_from_file_location("compressPdf", compress_pdf_path)
                compress_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(compress_module)
                compress_pdf_if_needed = compress_module.compress_pdf_if_needed
            else:
                raise ImportError("compressPdf.py not found")
        except Exception:
            # No-op function if compression module is not available
            def compress_pdf_if_needed(*args, **kwargs):
                print("  Warning: PDF compression module not available. Skipping compression.")
                return False

# Helper to get eng_essay directory for temp folders
def _get_eng_essay_dir() -> str:
    """Get the eng_essay directory path for temp folders."""
    return os.path.dirname(os.path.abspath(__file__))

try:
    from .annotate_pdf_with_essay_rubric import annotate_pdf_essay_pages
except (ImportError, ModuleNotFoundError):
    try:
        from annotate_pdf_with_essay_rubric import annotate_pdf_essay_pages  # type: ignore
    except (ImportError, ModuleNotFoundError):
        raise ImportError(
            "Cannot import 'annotate_pdf_essay_pages'. "
            "Ensure 'annotate_pdf_with_essay_rubric.py' exists in backend/eng_essay/ directory."
        )

# Import spell correction function from backend/ocr/ocr-spell-correction.py
try:
    import sys
    import importlib.util
    # Get the correct path relative to this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.dirname(current_dir)
    spell_correction_path = os.path.join(backend_dir, "ocr", "ocr-spell-correction.py")
    spec = importlib.util.spec_from_file_location("ocr_spell_correction", spell_correction_path)
    if spec and spec.loader:
        ocr_spell_module = importlib.util.module_from_spec(spec)
        sys.modules["ocr_spell_correction"] = ocr_spell_module
        spec.loader.exec_module(ocr_spell_module)
        detect_spelling_grammar_errors = ocr_spell_module.detect_spelling_grammar_errors
        _filter_errors = ocr_spell_module._filter_errors
        print(f"✓ OCR Spell Correction Module: ENABLED")
        print(f"✓ Loaded from: {spell_correction_path}")
    else:
        def detect_spelling_grammar_errors(grok_key, ocr_data):
            return []
        def _filter_errors(errors):
            return errors
except Exception as e:
    print(f"Warning: Could not import spell correction module: {e}")
    def detect_spelling_grammar_errors(grok_key, ocr_data):
        return []
    def _filter_errors(errors):
        return errors


# -----------------------------
# Helpers
# -----------------------------

def _format_duration(seconds: float) -> str:
    """Format elapsed time as 'Xs' or 'Xm Y.Ys' for display."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    m = int(seconds // 60)
    s = seconds - m * 60
    return f"{m}m {s:.2f}s"


def clean_json_from_llm(text: str) -> str:
    """Clean JSON response from LLM by removing markdown code blocks."""
    text = (text or "").strip()
    if not text:
        return ""
    
    # Remove markdown code blocks
    if text.startswith("```"):
        # Remove opening ```json or ```
        text = re.sub(r"^```[a-zA-Z0-9]*\n?", "", text)
        # Remove closing ```
        text = re.sub(r"\n?```$", "", text)
    
    return text.strip()


def _load_docx_text(path: str) -> str:
    doc = Document(path)
    parts: List[str] = []
    for p in doc.paragraphs:
        t = (p.text or "").strip()
        if t:
            parts.append(t)
    return "\n".join(parts)


def load_environment() -> Tuple[str, DocumentAnalysisClient]:
    load_dotenv()
    grok_key = os.getenv("Grok_API")
    azure_endpoint = os.getenv("AZURE_ENDPOINT")
    azure_key = os.getenv("AZURE_KEY")
    missing = []
    if not grok_key:
        missing.append("Grok_API")
    if not azure_endpoint:
        missing.append("AZURE_ENDPOINT")
    if not azure_key:
        missing.append("AZURE_KEY")
    if missing:
        raise EnvironmentError(
            f"Missing environment variable(s): {', '.join(missing)}. Please set them in your .env file."
        )
    doc_client = DocumentAnalysisClient(endpoint=azure_endpoint, credential=AzureKeyCredential(azure_key))
    return grok_key, doc_client


def validate_input_paths(pdf_path: str, output_json_path: str, output_pdf_path: str) -> None:
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    with open(pdf_path, "rb") as f:
        if f.read(4) != b"%PDF":
            raise ValueError(f"Not a valid PDF: {pdf_path}")

    for outp in [output_json_path, output_pdf_path]:
        d = os.path.dirname(outp)
        if d:
            os.makedirs(d, exist_ok=True)
        try:
            with open(outp, "w", encoding="utf-8") as wf:
                wf.write("")
            os.remove(outp)
        except Exception as e:
            raise ValueError(f"Cannot write to {outp}: {e}")

def parse_json_with_repair(
    grok_api_key: str,
    raw_text: str,
    *,
    debug_tag: str = "grok",
    max_fix_attempts: int = 2,
    debug_dir_override: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Try strict JSON parse.
    If fails, ask Grok to output valid JSON only (repair mode).
    Also saves raw + repaired outputs for debugging.
    """
    raw_clean = clean_json_from_llm(raw_text)

    # Use override debug_dir if provided, otherwise create default
    if debug_dir_override:
        debug_dir = debug_dir_override
    else:
        eng_essay_dir = _get_eng_essay_dir()
        debug_dir = os.path.join(eng_essay_dir, "debug_llm")
    os.makedirs(debug_dir, exist_ok=True)
    raw_path = os.path.join(debug_dir, f"{debug_tag}_raw.txt")
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(raw_text or "")

    def _extract_json_candidate(text: str) -> str:
        s = (text or "").strip()
        if not s:
            return s
        if s.startswith("{") and s.endswith("}"):
            return s
        if "{" in s and "}" in s:
            start = s.find("{")
            end = s.rfind("}")
            if end > start:
                return s[start : end + 1]
        if re.search(r'"[^"]+"\s*:', s):
            return "{" + s.strip().strip(",") + "}"
        return s

    # 1) direct parse (with light extraction)
    try:
        candidate = _extract_json_candidate(raw_clean)
        return json.loads(candidate)
    except Exception as e:
        err = str(e)

    # 2) repair loop
    last_text = raw_clean
    for attempt in range(1, max_fix_attempts + 1):
        fix_prompt = {
            "role": "user",
            "content": (
                "You previously produced invalid JSON.\n"
                "Fix it and return VALID JSON ONLY. No markdown, no comments, no extra text.\n\n"
                "Rules:\n"
                "- Use double quotes for all keys and strings.\n"
                "- Escape any inner quotes.\n"
                "- No trailing commas.\n"
                "- Output must be a single JSON object.\n\n"
                "Here is the invalid JSON:\n"
                f"{last_text}"
            ),
        }

        data = _grok_chat(
            grok_api_key,
            messages=[{"role": "system", "content": "Return valid JSON only."}, fix_prompt],
            temperature=0.0,
        )
        repaired = data["choices"][0]["message"]["content"]
        repaired_clean = clean_json_from_llm(repaired)

        repaired_path = os.path.join(debug_dir, f"{debug_tag}_repaired_attempt{attempt}.txt")
        with open(repaired_path, "w", encoding="utf-8") as f:
            f.write(repaired)

        try:
            candidate = _extract_json_candidate(repaired_clean)
            return json.loads(candidate)
        except Exception as e:
            last_text = repaired_clean
            err = str(e)

    raise ValueError(
        f"Failed to parse Grok JSON after repair attempts. Last error: {err}. "
        f"See {raw_path} and debug_llm/{debug_tag}_repaired_attempt*.txt"
    )


# -----------------------------
# PDF  Images for Grok
# -----------------------------

def pdf_to_page_images_for_grok(
    pdf_path: str,
    max_pages: Optional[int] = None,
    max_dim: int = 800,
    base64_cap: Optional[int] = None,
    output_dir: Optional[str] = None,
    max_total_base64_chars: int = 240_000,
) -> List[Dict[str, Any]]:
    """
    Render PDF pages to JPEG and encode them for Grok.
    Automatically downsizes/lowers quality until the combined base64 payload
    stays under `max_total_base64_chars` to avoid Grok API size/context errors.
    """

    # Create grok_images_essay inside eng_essay directory if not specified
    if output_dir is None:
        eng_essay_dir = _get_eng_essay_dir()
        output_dir = os.path.join(eng_essay_dir, "grok_images_essay")
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    try:
        total_pages = doc.page_count if max_pages is None else min(doc.page_count, max_pages)
        pil_pages: List[Image.Image] = []
        for idx in range(total_pages):
            pix = doc[idx].get_pixmap(dpi=200)
            pil_pages.append(Image.open(io.BytesIO(pix.tobytes("png"))))
    finally:
        doc.close()

    # Start from the requested max_dim, then progressively reduce size/quality if needed.
    dim_candidates_base = [800, 640, 560, 512, 448, 384, 360, 320]
    dim_candidates = [max_dim] + [d for d in dim_candidates_base if d < max_dim]
    dim_candidates = [d for i, d in enumerate(dim_candidates) if d not in dim_candidates[:i]]
    quality_candidates = [65, 55, 45, 40, 35]

    def _encode_pages(dim: int, quality: int, save_files: bool) -> Tuple[List[Dict[str, Any]], int]:
        encoded_pages: List[Dict[str, Any]] = []
        total_chars = 0
        for idx, pil_img in enumerate(pil_pages):
            img = pil_img.copy()
            img.thumbnail((dim, dim))

            if img.mode in ("RGBA", "LA", "P"):
                rgb = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "P":
                    img = img.convert("RGBA")
                rgb.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
                img = rgb
            elif img.mode != "RGB":
                img = img.convert("RGB")

            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality, optimize=True)

            encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
            truncated = False
            if base64_cap is not None and len(encoded) > base64_cap:
                encoded = encoded[:base64_cap]
                truncated = True

            total_chars += len(encoded)
            file_path = None
            if save_files:
                file_path = os.path.join(output_dir, f"page_{idx+1:03d}.jpg")
                with open(file_path, "wb") as f:
                    f.write(buffer.getvalue())

            encoded_pages.append(
                {"page": idx + 1, "image_base64": encoded, "file_path": file_path, "truncated": truncated}
            )
        return encoded_pages, total_chars

    chosen: Optional[Tuple[List[Dict[str, Any]], int, int, int]] = None
    for dim in dim_candidates:
        for quality in quality_candidates:
            pages_tmp, total_chars = _encode_pages(dim, quality, save_files=False)
            chosen = (pages_tmp, total_chars, dim, quality)
            if max_total_base64_chars and total_chars > max_total_base64_chars:
                continue
            final_pages, final_total = _encode_pages(dim, quality, save_files=True)
            print(
                f"Saved {len(final_pages)} page images to '{output_dir}/' "
                f"(dim={dim}, quality={quality}, total_base64_chars={final_total})"
            )
            return final_pages

    # Fallback to the smallest attempted settings if nothing met the budget.
    if chosen:
        pages_tmp, total_chars, dim, quality = chosen
        final_pages, final_total = _encode_pages(dim, quality, save_files=True)
        print(
            f"Saved {len(final_pages)} page images to '{output_dir}/' "
            f"(dim={dim}, quality={quality}, total_base64_chars={final_total}) [fallback]"
        )
        return final_pages

    return []


def get_report_page_size(
    pdf_path: str,
    dpi: int = 220,
    margin_ratio: float = 0.35,
    min_height: int = 3500,
    max_width: int = 9000,
    max_height: int = 12000,
    max_pixels: int = 50000000,  # ~50MP limit (e.g., 5000x10000)
    fallback: Tuple[int, int] = (2977, 4211),
) -> Tuple[int, int]:
    """
    Match report page size to the annotated canvas.

    This project uses equal side margins around the essay body, so annotated width is:
      total_width = orig_w * (1 + 2 * margin_ratio)

    This implementation is adapted from `grade_pdf_answer.py`:
    - Uses progressive DPI reduction for very large PDFs (prevents MemoryError)
    - Caps max width/height and total pixel count
    - Keeps THIS project's report width aligned to `annotate_pdf_essay_pages` canvas
      (width multiplier = 2.0 of the rendered PDF page width at the chosen DPI)
    """
    doc = fitz.open(pdf_path)
    try:
        if doc.page_count == 0:
            return fallback

        # Check page size first to estimate memory requirements
        page = doc[0]
        page_rect = page.rect
        page_width_pts = page_rect.width
        page_height_pts = page_rect.height

        # Calculate expected pixmap size at target DPI (1 point = 1/72 inch)
        expected_width = int(page_width_pts * (dpi / 72))
        expected_height = int(page_height_pts * (dpi / 72))
        expected_pixels = expected_width * expected_height
        expected_mb = (expected_pixels * 4) / (1024 * 1024)  # RGBA = 4 bytes per pixel

        # Progressive DPI reduction if page is too large
        target_dpi = dpi
        max_safe_mb = 50  # ~50MB per pixmap is reasonable
        if expected_mb > max_safe_mb:
            safe_dpi = int(dpi * (max_safe_mb / expected_mb) ** 0.5)
            target_dpi = max(100, safe_dpi)  # don't go too low
            print(
                f"WARNING: First page is very large ({expected_width}x{expected_height} at {dpi} DPI, "
                f"~{expected_mb:.1f}MB). Using {target_dpi} DPI instead to prevent MemoryError."
            )

        # Try to get pixmap with progressive DPI reduction
        dpi_options = [target_dpi, 150, 100, 75] if target_dpi < dpi else [dpi, 150, 100, 75]
        pix = None
        last_error = None
        for attempt_dpi in dpi_options:
            try:
                pix = page.get_pixmap(dpi=attempt_dpi)
                if attempt_dpi < dpi:
                    print(f"Successfully created pixmap at {attempt_dpi} DPI (reduced from {dpi} DPI)")
                break
            except Exception as e:
                last_error = e
                if attempt_dpi == dpi_options[-1]:
                    print(f"WARNING: Failed to create pixmap even at {attempt_dpi} DPI. Using fallback size. Error: {e}")
                    return fallback
                continue

        if pix is None:
            print(f"WARNING: Failed to create pixmap. Using fallback size. Error: {last_error}")
            return fallback

        orig_w, orig_h = pix.width, pix.height
        del pix
        gc.collect()

        # Match annotate_pdf_essay_pages canvas width: orig_w + left + right
        total_width = int(orig_w * (1.0 + 2.0 * margin_ratio))
        total_height = max(orig_h, min_height)

        # Cap width/height to prevent extremely large images
        if total_width > max_width:
            total_width = max_width
        if total_height > max_height:
            total_height = max_height

        # Check total pixel count (RGB images ~3 bytes per pixel)
        total_pixels = total_width * total_height
        if total_pixels > max_pixels:
            scale = (max_pixels / float(total_pixels)) ** 0.5
            total_width = int(total_width * scale)
            total_height = int(total_height * scale)
            print(
                f"WARNING: Calculated page size exceeds pixel limit. "
                f"Scaled down to ({total_width}x{total_height}) to prevent MemoryError."
            )

        return (total_width, total_height)
    except Exception as e:
        print(f"WARNING: Error calculating report page size: {e}. Using fallback size.")
        return fallback
    finally:
        doc.close()


# -----------------------------
# OCR (Azure Document Intelligence)
# -----------------------------

def _is_noise_text(text: str, bbox: List[Tuple[int, int]], page_w: int, page_h: int) -> bool:
    if not text:
        return True
    if len(text.strip()) <= 2:
        return True
    # If Azure doesn't provide a polygon, keep the text (we can't judge size)
    if not bbox:
        return False
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    if not xs or not ys:
        return False
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    if page_w and page_h:
        rel_w = w / max(1e-6, page_w)
        rel_h = h / max(1e-6, page_h)
        if rel_w < 0.002 or rel_h < 0.002:
            return True
    else:
        if w < 2 or h < 2:
            return True
    return False


def run_ocr_on_pdf(
    doc_client: DocumentAnalysisClient,
    pdf_path: str,
    *,
    workers: int = 3,
    render_dpi: int = 220,
    debug_pages_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run Azure OCR page by page to avoid document size limits.
    Each page is rendered to JPEG, optionally resized/compressed on retry if Azure rejects size.
    Runs pages in parallel (workers>1). Saves per-page debug JSON with bboxes if debug_pages_dir is provided.
    """
    def _analyze_image_bytes(img_bytes: bytes) -> Any:
        poller = doc_client.begin_analyze_document("prebuilt-read", document=img_bytes)
        return poller.result()

    def _encode_page_img(pil_img: Image.Image, scale: float, quality: int) -> bytes:
        img = pil_img.copy()
        if scale != 1.0:
            new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
            img = img.resize(new_size, Image.LANCZOS)
        if img.mode != "RGB":
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        return buf.getvalue()

    doc = fitz.open(pdf_path)
    try:
        pil_pages: List[Tuple[int, Image.Image]] = []
        for idx in range(doc.page_count):
            page = doc[idx]
            pix = page.get_pixmap(dpi=render_dpi)
            pil_pages.append((idx + 1, Image.open(io.BytesIO(pix.tobytes("png")))))
    finally:
        doc.close()

    if debug_pages_dir:
        os.makedirs(debug_pages_dir, exist_ok=True)

    def _process_page(page_number: int, pil_img: Image.Image) -> Dict[str, Any]:
        result = None
        attempts = [(1.0, 75), (0.85, 70), (0.7, 60)]
        last_err: Optional[Exception] = None
        for scale, quality in attempts:
            try:
                img_bytes = _encode_page_img(pil_img, scale=scale, quality=quality)
                result = _analyze_image_bytes(img_bytes)
                used = {"scale": scale, "quality": quality}
                break
            except HttpResponseError as e:
                last_err = e
                if "InvalidContentLength" in str(e):
                    continue
                raise
        if result is None:
            raise RuntimeError(f"OCR failed on page {page_number}: {last_err}")

        first_page = result.pages[0] if result.pages else None
        page_w = float(getattr(first_page, "width", 0.0) or 0.0) if first_page else float(pil_img.width)
        page_h = float(getattr(first_page, "height", 0.0) or 0.0) if first_page else float(pil_img.height)
        page_lines: List[Dict[str, Any]] = []
        page_text_parts: List[str] = []
        page_words_flat = []  # Extract ALL words for spelling annotation

        for p in result.pages:
            # Extract ALL words directly from Azure API (before any filtering)
            for w in (p.words or []):
                txt = (w.content or "").strip()
                if not txt:
                    continue
                poly = []
                if w.polygon:
                    poly = [(int(pt.x), int(pt.y)) for pt in w.polygon]
                page_words_flat.append({
                    "text": txt,
                    "bbox": poly,
                    "confidence": float(getattr(w, "confidence", 1.0) or 1.0),
                })
            
            page_words = list(p.words or [])
            for line in p.lines or []:
                text = (line.content or "").strip()
                line_bbox = []
                if line.polygon:
                    line_bbox = [(int(pt.x), int(pt.y)) for pt in line.polygon]
                if _is_noise_text(text, line_bbox, page_w, page_h):
                    continue

                matched_words = []
                if not line.spans:
                    page_lines.append({"text": text, "bbox": line_bbox, "words": []})
                    continue

                for word in page_words:
                    wsp = getattr(word, "span", None)
                    if not wsp:
                        continue
                    for lsp in line.spans:
                        l_start = lsp.offset
                        l_end = l_start + lsp.length
                        w_start = wsp.offset
                        w_end = w_start + wsp.length
                        if w_start >= l_start and w_end <= l_end:
                            w_bbox = []
                            if word.polygon:
                                w_bbox = [(int(pt.x), int(pt.y)) for pt in word.polygon]
                            if _is_noise_text(word.content, w_bbox, page_w, page_h):
                                continue
                            matched_words.append({
                                "text": word.content,
                                "bbox": w_bbox,
                                "confidence": float(getattr(word, "confidence", 1.0) or 1.0)
                            })
                            break
                    else:
                        continue
                    break
                else:
                    for word in page_words:
                        w_bbox = [(int(pt.x), int(pt.y)) for pt in word.polygon] if word.polygon else []
                        if _is_noise_text(word.content, w_bbox, page_w, page_h):
                            continue
                        matched_words.append({
                            "text": word.content,
                            "bbox": w_bbox,
                            "confidence": float(getattr(word, "confidence", 1.0) or 1.0)
                        })

                if matched_words:
                    page_lines.append({"text": text, "bbox": line_bbox, "words": matched_words})

            # collect full page text in order
            for ln in (p.lines or []):
                t = (ln.content or "").strip()
                if t:
                    page_text_parts.append(t)

        page_text = " ".join(page_text_parts)
        
        debug_payload = {
            "page_number": page_number,
            "page_width": page_w,
            "page_height": page_h,
            "unit": "pixel",
            "lines": page_lines,
            "words": page_words_flat,  # Use the flat array extracted from Azure API
            "ocr_full_text_page": page_text,
            "attempt": used if result else {},
        }

        return debug_payload

    pages_output: List[Dict[str, Any]] = []
    full_text_parts: List[str] = []

    worker_count = max(1, int(workers or 1))
    with ThreadPoolExecutor(max_workers=worker_count) as ex:
        futures = {ex.submit(_process_page, num, img): num for num, img in pil_pages}
        for future in as_completed(futures):
            page_number = futures[future]
            data = future.result()
            pages_output.append({
                "page_number": data["page_number"],
                "page_width": data.get("page_width"),
                "page_height": data.get("page_height"),
                "unit": data.get("unit", "pixel"),
                "ocr_page_text": data.get("ocr_full_text_page", ""),
                "lines": data["lines"],
                "words": data.get("words", []),
            })
            full_text_parts.append(data.get("ocr_full_text_page", ""))
            if debug_pages_dir:
                out_path = os.path.join(debug_pages_dir, f"page_{page_number:03d}.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

    pages_output.sort(key=lambda x: x.get("page_number", 0))
    return {"pages": pages_output, "full_text": "\n".join([t for t in full_text_parts if t]).strip()}


# -----------------------------
# Load Rubrics + Report Format
# -----------------------------

def load_essay_rubric_text(path: str) -> str:
    """Load essay rubric text. If path is relative, look in eng_essay directory."""
    if not os.path.isabs(path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(script_dir, path)
    return _load_docx_text(path)


def load_annotations_rubric_text(path: str) -> str:
    """Load annotations rubric text. If path is relative, look in eng_essay directory."""
    if not os.path.isabs(path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(script_dir, path)
    return _load_docx_text(path)


def load_report_format_text(path: str) -> str:
    """Load report format text. If path is relative, look in eng_essay directory."""
    if not os.path.isabs(path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(script_dir, path)
    return _load_docx_text(path)


# -----------------------------
# Grok Calls (chunked payloads to avoid 503 / size limits)
# -----------------------------

# Max page images per structure request; grading uses at most MAX_PAGES_FOR_GRADING when doc is longer.
STRUCTURE_CHUNK_PAGES = 5
MAX_PAGES_FOR_GRADING = 20


def _chunk_page_images(
    page_images: List[Dict[str, Any]],
    chunk_size: int,
) -> List[Tuple[List[Dict[str, Any]], List[int]]]:
    """
    Split page_images into chunks by page number. Returns list of (chunk_pages, page_numbers).
    Preserves order; every page appears in exactly one chunk.
    """
    if not page_images:
        return []
    by_page = sorted(page_images, key=lambda p: p.get("page", 0))
    chunks: List[Tuple[List[Dict[str, Any]], List[int]]] = []
    for i in range(0, len(by_page), chunk_size):
        chunk = by_page[i : i + chunk_size]
        pages = [p.get("page") for p in chunk if p.get("page") is not None]
        chunks.append((chunk, pages))
    return chunks


def _subset_page_images_for_grading(
    page_images: List[Dict[str, Any]],
    max_pages: int,
) -> List[Dict[str, Any]]:
    """
    When there are many pages, return a representative subset (first, last, middle)
    so the grading payload stays small. Nothing is lost: full OCR and structure are still sent.
    """
    if len(page_images) <= max_pages:
        return page_images
    by_page = sorted(page_images, key=lambda p: p.get("page", 0))
    n = len(by_page)
    # first 3, last 3, and evenly spaced from middle
    head = 3
    tail = 3
    mid_count = max_pages - head - tail
    if mid_count <= 0:
        return by_page[:max_pages]
    indices = list(range(head))
    # middle indices evenly spaced
    step = (n - head - tail) / (mid_count + 1)
    for i in range(mid_count):
        idx = head + int((i + 1) * step)
        if idx < n - tail:
            indices.append(idx)
    indices.extend(range(n - tail, n))
    indices = sorted(set(indices))[:max_pages]
    return [by_page[i] for i in indices]


def _grok_chat(
    grok_api_key: str,
    messages: List[Dict[str, str]],
    model: str = "grok-4-1-fast-reasoning",
    temperature: float = 0.15,
    max_tokens: Optional[int] = None,
    timeout: int = 180,
    max_retries: int = 10,
    backoff_base: float = 2.0,
    backoff_max: float = 60.0,
) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {grok_api_key}"}
    payload = {"model": model, "messages": messages, "temperature": temperature}
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    last_err: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(
                "https://api.x.ai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=(30, timeout),
            )
            if resp.status_code >= 300:
                err = RuntimeError(f"Grok API error {resp.status_code}: {resp.text}")
                if resp.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
                    last_err = err
                    delay = min(backoff_max, backoff_base ** attempt)
                    print(f"  Grok API {resp.status_code} (attempt {attempt + 1}/{max_retries + 1}). Retrying in {delay:.0f}s...")
                    time.sleep(delay)
                    continue
                raise err
            return resp.json()
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_err = e
            if attempt >= max_retries:
                raise
            time.sleep(min(backoff_max, backoff_base ** attempt))

    raise RuntimeError(f"Grok request failed after retries: {last_err}")



def call_grok_for_essay_structure_paragraphs_only(
    grok_api_key: str,
    ocr_data: Dict[str, Any],
    page_images: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Essay structure for this pipeline:
      - Outline first (expected) across ~3-4 pages; can include headings and short paragraph-style bullets/sections
      - Then essay as paragraphs (~10-12 pages); headings or section markers may appear
      - Identify where the outline ends and the main essay begins

    Output schema:
    {
      "topic": "string",
      "outline": {
        "present": true/false,
        "pages": [1],
        "quality": "Weak|Average|Good|Excellent",
        "issues": ["..."],
        "strengths": ["..."]
      },
      "outline_span": {"start_page": 1, "end_page": 3},
      "outline_sections": [{"title": "string", "page": 1, "notes": "string"}],
      "essay_start_page": 4,
      "paragraph_map": [
        {"page": 1, "role_guess": "outline|intro|body|conclusion|mixed", "notes": "short"}
      ],
      "overall_flow_comment": "short"
    }
    """
    system = {
        "role": "system",
        "content": (
            "You are an expert CSS English Essay examiner.\n"
            "Essay may include headings or section markers. Do not invent headings; only report if visible.\n"
            "First part is Outline, then Intro/Body/Conclusion as paragraph blocks.\n"
            "Primary truth = page images. OCR is only helper; ignore OCR errors and never mention them.\n"
            "When returning the topic/title, use the exact wording written in the answer—no rephrasing or additions.\n"
            "IMPORTANT: Do NOT treat events from 2025 or later years as speculation. "
            "If you encounter dates/events you don't have knowledge about, simply ignore them and focus on content detection. "
            "Never comment on whether content is speculative based on your knowledge cutoff.\n"
            "Return JSON only."
        ),
    }

    # lightweight OCR summary
    sanitized_pages = []
    for p in ocr_data.get("pages", []):
        lines = []
        for line in p.get("lines", []):
            lines.append((line.get("text") or ""))
        sanitized_pages.append({"page_number": p.get("page_number"), "lines_preview": lines})

    chunks = _chunk_page_images(page_images, STRUCTURE_CHUNK_PAGES)
    if not chunks:
        return {
            "topic": "",
            "outline": {"present": False, "pages": [], "quality": "Weak", "issues": [], "strengths": []},
            "outline_span": {},
            "outline_sections": [],
            "essay_start_page": 1,
            "paragraph_map": [],
            "overall_flow_comment": "",
        }

    merged: Dict[str, Any] = {
        "topic": "",
        "outline": {"present": False, "pages": [], "quality": "Weak", "issues": [], "strengths": []},
        "outline_span": {},
        "outline_sections": [],
        "essay_start_page": 1,
        "paragraph_map": [],
        "overall_flow_comment": "",
    }
    all_pages = sorted({p.get("page") for p in page_images if p.get("page") is not None})
    last_page_set = {max(all_pages)} if all_pages else set()

    for idx, (chunk_pages, chunk_page_nums) in enumerate(chunks):
        chunk_has_page_one = 1 in chunk_page_nums or (chunk_page_nums and min(chunk_page_nums) == 1)
        chunk_has_last_page = bool(chunk_page_nums and max(chunk_page_nums) in last_page_set)

        user_payload = {
            "task": (
                "Detect topic/title, identify outline pages first, and map each page's role "
                "(outline/intro/body/conclusion/mixed) for the essay. "
                "You are given ONLY page images for pages " + str(chunk_page_nums) + ". "
                "Return paragraph_map entries ONLY for these page numbers. "
                + ("This chunk includes page 1: also return topic, outline, outline_span, outline_sections, essay_start_page." if chunk_has_page_one else "")
                + (" This chunk includes the last page: also return overall_flow_comment." if chunk_has_last_page else "")
            ),
            "rules": [
                "Do NOT invent headings or sections; only report if visible.",
                "Outline is typically a numbered/roman list or bullet plan early (often page 1) spanning ~3-4 pages; may include headings and short paragraphs.",
                "If outline is missing or weak, say so strongly.",
                "Do NOT comment on the numbering format, numeral structure, bullet style, or point-numbering convention used. Focus only on content quality.",
                "role_guess is best-effort: outline, intro, body, conclusion, mixed.",
                "Ignore OCR errors; do not mention OCR quality, legibility, scanning, handwriting, blurring, or smudging anywhere.",
                "Topic must be verbatim as written in the essay; never expand or paraphrase.",
                "After the outline, the main essay continues for ~10-12 pages as paragraphs; identify the page where the outline ends and essay begins.",
                "List each outline section with its page number; use the visible heading/phrase as the title (do not invent).",
                "If parts are unreadable, say 'content unclear' without blaming OCR/scan/handwriting.",
            ],
            "ocr_pages_preview": sanitized_pages,
            "ocr_full_text": (ocr_data.get("full_text") or ""),
            "page_images": chunk_pages,
            "output_schema": {
                "topic": "string",
                "outline": {
                    "present": True,
                    "pages": [1],
                    "quality": "Weak",
                    "issues": ["..."],
                    "strengths": ["..."],
                },
                "outline_span": {"start_page": 1, "end_page": 3},
                "outline_sections": [{"title": "Section title", "page": 1, "notes": "short"}],
                "essay_start_page": 4,
                "paragraph_map": [{"page": 1, "role_guess": "outline", "notes": "short"}],
                "overall_flow_comment": "short",
            },
        }

        print(f"  Structure chunk {idx + 1}/{len(chunks)} (pages {chunk_page_nums})...")
        data = _grok_chat(
            grok_api_key,
            messages=[system, {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}],
            temperature=0.1,
        )
        content = data["choices"][0]["message"]["content"]
        cleaned_content = clean_json_from_llm(content)
        
        # Debug logging and error handling
        if not cleaned_content or not cleaned_content.strip():
            print(f"  WARNING: Empty response from Grok API for chunk {idx + 1}")
            print(f"  Raw response content: {content[:500] if content else 'None'}")
            raise ValueError(f"Empty or invalid response from Grok API for structure detection (chunk {idx + 1})")
        
        try:
            parsed = json.loads(cleaned_content)
        except json.JSONDecodeError as e:
            print(f"  ERROR: Failed to parse JSON from Grok response (chunk {idx + 1})")
            print(f"  Cleaned content (first 1000 chars): {cleaned_content[:1000]}")
            print(f"  JSON error: {e}")
            raise ValueError(f"Invalid JSON response from Grok API: {e}") from e

        pm = parsed.get("paragraph_map") or []
        for e in pm:
            if isinstance(e, dict) and e.get("page") is not None:
                merged["paragraph_map"].append(e)
        merged["paragraph_map"].sort(key=lambda x: (x.get("page") or 0))

        if chunk_has_page_one:
            if (parsed.get("topic") or "").strip():
                merged["topic"] = (parsed.get("topic") or "").strip()
            if parsed.get("outline") and isinstance(parsed["outline"], dict):
                merged["outline"] = parsed["outline"]
            if parsed.get("outline_span") and isinstance(parsed["outline_span"], dict):
                merged["outline_span"] = parsed["outline_span"]
            if isinstance(parsed.get("outline_sections"), list):
                merged["outline_sections"] = parsed["outline_sections"]
            if isinstance(parsed.get("essay_start_page"), int):
                merged["essay_start_page"] = parsed["essay_start_page"]
        if chunk_has_last_page and (parsed.get("overall_flow_comment") or "").strip():
            merged["overall_flow_comment"] = (parsed.get("overall_flow_comment") or "").strip()

    return merged


def _parse_range(rng: str) -> Tuple[int, int]:
    """
    Parse a mark range string into (lo, hi).

    Accepts:
    - ASCII hyphen:        '6-8'
    - En dash / em dash:   '6–8', '6—8'
    - With spaces:         '6 – 8'
    - 'to' as separator:   '6 to 8', '6   TO   8'

    On any parse failure, returns (0, 0).
    """
    s = str(rng or "").strip()
    if not s:
        return 0, 0

    # Normalise common separators to a simple hyphen
    # U+2013 (EN DASH), U+2014 (EM DASH)
    s = s.replace("–", "-").replace("—", "-")
    # Replace textual 'to' with hyphen
    s = re.sub(r"\bto\b", "-", s, flags=re.IGNORECASE)
    # Collapse whitespace
    s = re.sub(r"\s+", "", s)

    parts = s.split("-")
    if len(parts) != 2:
        return 0, 0
    try:
        lo = int(parts[0])
        hi = int(parts[1])
    except Exception:
        return 0, 0
    if hi < lo:
        lo, hi = hi, lo
    return lo, hi


def call_grok_for_essay_grading_strict_range(
    grok_api_key: str,
    essay_rubric_text: str,
    report_format_text: str,
    ocr_data: Dict[str, Any],
    structure: Dict[str, Any],
    page_images: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    STRICT range grading:
      - DO NOT output exact marks
      - output "marks_awarded_range": "x-y"
      - keep total ranges very conservative (excellent essays 38-42, lesser accordingly)
    """

    system = {
        "role": "system",
        "content": (
            "You are a strict CSS English Essay examiner (FPSC style). "
            "Be conservative: excellent essays should score in the 38-42 range out of 100. "
            "Good essays should score 30-37, Average essays 20-29, Weak essays below 20. "
            "IMPORTANT: Do NOT treat events from 2025 or later years as speculation. "
            "If you encounter dates/events you don't have knowledge about, simply ignore them and focus on the essay's structure and argumentation. "
            "Never comment on whether an event is speculative or not based on your knowledge cutoff. "
            "Return VALID JSON only; no markdown or commentary."
        ),
    }

    schema_hint = {
        "topic": "",
        "total_marks": 100,
        "overall_rating": "Weak",
        "criteria": [
            {
                "id": "outline_topic_interpretation",
                "criterion": "Essay Outline & Topic Interpretation/Clarity",
                "marks_allocated": 40,
                "marks_awarded_range": "0-0",
                "rating": "Weak",
                "key_comments": "string",
            },
            {
                "id": "introduction",
                "criterion": "Introduction",
                "marks_allocated": 15,
                "marks_awarded_range": "0-0",
                "rating": "Weak",
                "key_comments": "string",
            },
            {
                "id": "relevance_focus",
                "criterion": "Relevance & Focus (Adherence to Topic)",
                "marks_allocated": 5,
                "marks_awarded_range": "0-0",
                "rating": "Weak",
                "key_comments": "string",
            },
            {
                "id": "content_depth_originality",
                "criterion": "Content Depth & Originality",
                "marks_allocated": 10,
                "marks_awarded_range": "0-0",
                "rating": "Weak",
                "key_comments": "string",
            },
            {
                "id": "argumentation_critical_analysis",
                "criterion": "Argumentation & Critical Analysis",
                "marks_allocated": 10,
                "marks_awarded_range": "0-0",
                "rating": "Weak",
                "key_comments": "string",
            },
            {
                "id": "organization_coherence_transitions",
                "criterion": "Organization, Coherence & Transitions",
                "marks_allocated": 5,
                "marks_awarded_range": "0-0",
                "rating": "Weak",
                "key_comments": "string",
            },
            {
                "id": "expression_grammar_vocab_style",
                "criterion": "Expression, Grammar, Vocabulary & Style",
                "marks_allocated": 10,
                "marks_awarded_range": "0-0",
                "rating": "Weak",
                "key_comments": "string",
            },
            {
                "id": "conclusion_overall_impression",
                "criterion": "Conclusion & Overall Impression",
                "marks_allocated": 5,
                "marks_awarded_range": "0-0",
                "rating": "Weak",
                "key_comments": "string",
            },
        ],
        "total_awarded_range": "0-0",
        "reasons_for_low_score": ["..."],
        "suggested_improvements_for_higher_score_70_plus": ["..."],
        "overall_remarks": "string",
    }

    instructions = (
    "Grade strictly using the provided CSS English Essay rubric (weights are in the rubric/schema).\n"
    "Objective:\n"
    "- Identify ONLY the specific issues that caused loss of marks under each rubric criterion.\n"
    "- Do NOT praise, summarize, reinterpret, or rewrite any part of the essay.\n"
    "- For output, use simple and complete sentences to increase readability.\n"
    "Rules:\n"
    "- Output only mark ranges per criterion (e.g., \"6–8\"); width ≤ 3 points.\n"
    "- Hard cap: the total_awarded_range upper bound MUST NOT exceed 42; scale ranges down to stay under this cap.\n"
    "- Keep totals conservative; excellent essays score 38-42/100, good essays 30-37, average 20-29, weak below 20.\n"
    "- Overall rating must be one of: Excellent, Good, Average, Weak.\n"
    "- total_awarded_range = sum of all low bounds and high bounds across criteria.\n"
    "- Topic must be verbatim from the essay; do not rephrase or shorten.\n"
    "- Judge only what is written; do not assume intent or missing content.\n"
    "- Do not mention OCR/scan/legibility/handwriting; critique clarity, relevance, logic, and language only.\n"
    "- Headings/section markers may exist; evaluate only what is visible; do not invent content.\n"
    "- Do NOT comment on the numbering format, numeral structure, bullet style, or point-numbering convention used in the outline or essay body. "
    "Focus only on the substance and content quality, not how points are numbered or listed.\n"
    "Issue Identification Rules (Strict):\n"
    "- For EACH criterion, list ONLY concrete deficiencies observed in the essay.\n"
    "- Each issue must clearly explain why marks were lost.\n"
    "- Use simple, direct language so the student understands exactly what went wrong.\n"
    "- Avoid vague or generic phrases (e.g., 'needs improvement', 'lacks depth', 'weak analysis').\n"
    "- State precise problems (e.g., 'no clear thesis in introduction', 'arguments listed without explanation', "
    "'claims unsupported by evidence', 'irrelevant paragraphs', 'repetition of same example', "
    "'frequent grammar errors in introduction and conclusion').\n"
    "- Reasons for low score must be directly drawn from the essay (structure, argument gaps, evidence, relevance, language).\n"
    "Reasons for Low Score (ELABORATE):\n"
    "- Each reason in 'reasons_for_low_score' MUST be a detailed, well-explained sentence (2-3 lines minimum).\n"
    "- Do NOT write short or vague reasons like 'weak introduction' or 'lack of depth'.\n"
    "- Instead, explain specifically WHAT is wrong and WHY it costs marks. For example:\n"
    "  'The introduction fails to present a clear thesis statement or define the scope of discussion, "
    "which means the reader has no roadmap for the essay and the examiner cannot assess topic interpretation.'\n"
    "  'Body paragraphs repeat the same example of economic impact three times without introducing "
    "new evidence or perspectives, which shows limited research and reduces content depth marks.'\n"
    "- Each reason must reference specific parts of the essay (introduction, body paragraph, conclusion, outline) "
    "and explain the exact deficiency with its impact on the score.\n"
    "Suggested Improvements (if required by schema):\n"
    "- Provide ONLY targeted, actionable fixes directly linked to the identified issues.\n"
    "- Keep suggestions specific and exam-oriented (e.g., 'state a one-sentence thesis in the introduction', "
    "'add factual evidence to support claim X', 'remove repeated example in body paragraph 3').\n"
    "- Do NOT give general writing advice or motivational comments.\n"
    "Other Constraints:\n"
    "- Never leave any field blank.\n"
    "- If unsure, choose the lower bound.\n"
    "- Return JSON only, strictly matching the provided schema."
    )

    # Use subset of page images when many pages to keep payload under API limits; full OCR + structure still sent
    grading_images = _subset_page_images_for_grading(page_images, MAX_PAGES_FOR_GRADING)
    if len(grading_images) < len(page_images):
        print(f"  Grading: using {len(grading_images)} representative pages (of {len(page_images)}) to reduce payload size.")
    payload = {
        "essay_rubric_text": (essay_rubric_text or ""),
        "report_format_text": (report_format_text or ""),
        "structure_detected": structure,
        "ocr_full_text": (ocr_data.get("full_text") or ""),
        "page_images": grading_images,
        "output_schema": schema_hint,
    }

    def _coerce_grading_shape(data: Dict[str, Any]) -> Dict[str, Any]:
        """Coerce partial grading JSON into the expected schema so pipeline can continue."""
        parsed = data if isinstance(data, dict) else {}
        out: Dict[str, Any] = {}

        for key, value in schema_hint.items():
            out[key] = value

        for key, value in parsed.items():
            if key != "criteria":
                out[key] = value

        valid_ratings = {"Excellent", "Good", "Average", "Weak"}
        if out.get("overall_rating") not in valid_ratings:
            out["overall_rating"] = "Weak"

        out["topic"] = str(out.get("topic") or "")
        out["total_marks"] = int(out.get("total_marks") or 100)

        raw_criteria = parsed.get("criteria") if isinstance(parsed.get("criteria"), list) else []
        by_id: Dict[str, Dict[str, Any]] = {}
        for item in raw_criteria:
            if isinstance(item, dict) and item.get("id"):
                by_id[str(item.get("id"))] = item

        template_criteria = schema_hint.get("criteria") or []
        coerced_criteria: List[Dict[str, Any]] = []
        for idx, tmpl in enumerate(template_criteria):
            src = by_id.get(str(tmpl.get("id")))
            if src is None and idx < len(raw_criteria) and isinstance(raw_criteria[idx], dict):
                src = raw_criteria[idx]

            merged = dict(tmpl)
            if isinstance(src, dict):
                merged.update(src)

            merged["id"] = str(merged.get("id") or tmpl.get("id") or f"criterion_{idx + 1}")
            merged["criterion"] = str(merged.get("criterion") or tmpl.get("criterion") or "Criterion")
            merged["marks_allocated"] = int(merged.get("marks_allocated") or tmpl.get("marks_allocated") or 0)

            lo, hi = _parse_range(str(merged.get("marks_awarded_range") or "0-0"))
            if hi - lo > 3:
                hi = lo + 3
            lo = max(0, lo)
            hi = max(lo, hi)
            merged["marks_awarded_range"] = f"{lo}-{hi}"

            if merged.get("rating") not in valid_ratings:
                merged["rating"] = "Weak"

            key_comments = merged.get("key_comments")
            if isinstance(key_comments, list):
                joined = "; ".join(str(x).strip() for x in key_comments if str(x).strip())
                merged["key_comments"] = joined or "No specific criterion-wise issue extracted."
            else:
                merged["key_comments"] = str(key_comments or "No specific criterion-wise issue extracted.")

            coerced_criteria.append(merged)

        if not coerced_criteria:
            coerced_criteria = [dict(c) for c in template_criteria]

        out["criteria"] = coerced_criteria

        reasons = out.get("reasons_for_low_score")
        if not isinstance(reasons, list) or not reasons:
            out["reasons_for_low_score"] = [
                "The essay response did not provide enough rubric-grounded structure for stronger marks."
            ]

        improvements = out.get("suggested_improvements_for_higher_score_70_plus")
        if not isinstance(improvements, list) or not improvements:
            out["suggested_improvements_for_higher_score_70_plus"] = [
                "Add a clearer thesis, stronger evidence, and tighter paragraph-level argument progression."
            ]

        out["overall_remarks"] = str(out.get("overall_remarks") or "")
        return out

    def _is_valid_grading(data: Dict[str, Any]) -> bool:
        criteria = data.get("criteria")
        if not isinstance(criteria, list) or len(criteria) < 1:
            print(f"  Validation failed: criteria count = {len(criteria) if isinstance(criteria, list) else 'not a list'}")
            return False
        if not isinstance(data.get("total_awarded_range"), str):
            print(f"  Validation failed: total_awarded_range = {data.get('total_awarded_range')} (type: {type(data.get('total_awarded_range'))})")
            return False
        if data.get("topic") is None:
            print(f"  Validation failed: topic is None")
            return False
        rating = data.get("overall_rating")
        if rating not in ("Excellent", "Good", "Average", "Weak"):
            print(f"  Validation failed: overall_rating = '{rating}' (not in valid list)")
            return False

        return True

    def _enforce_range_rules(parsed: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert mark ranges to single values per criterion, then compute total with 4-point range.
        - For each criterion: pick single value (50% chance of minimum, 50% chance of maximum)
        - Sum all single values to get total
        - Apply 42 cap if needed (scale down proportionally) - excellent essays max 38-42
        - Set total range as (total-2) to (total+2) - a 4-point range
        """
        crit_list = parsed.get("criteria") or []
        
        # Process each criterion: parse range, clamp width, pick single value
        for c in crit_list:
            rng = c.get("marks_awarded_range", "0-0")
            lo, hi = _parse_range(rng)
            
            # Clamp range width to max 3
            if hi - lo > 3:
                hi = lo + 3
            lo = max(0, lo)
            hi = max(lo, hi)
            
            # Keep the range for reference
            c["marks_awarded_range"] = f"{lo}-{hi}"
            
            # Pick single value: 50% chance of minimum, 50% chance of maximum
            if random.random() < 0.5:  # 50% chance
                marks_awarded = hi
            else:  # 50% chance
                marks_awarded = lo
            
            c["marks_awarded"] = marks_awarded
        
        # Calculate total from single values
        total = sum(c.get("marks_awarded", 0) for c in crit_list)
        
        # Apply 42 cap if total exceeds 42 (excellent essays max 38-42)
        if total > 42 and crit_list:
            scale = 42.0 / float(total)
            # Scale down all marks_awarded proportionally
            for c in crit_list:
                original = c.get("marks_awarded", 0)
                scaled = max(0, int(round(original * scale)))
                c["marks_awarded"] = scaled
            
            # Recalculate total after scaling
            total = sum(c.get("marks_awarded", 0) for c in crit_list)
        
        # Set total range as (total-2) to (total+2) - 4-point range
        total_lo = max(0, total - 2)
        total_hi = min(100, total + 2)
        parsed["total_awarded_range"] = f"{total_lo}-{total_hi}"
        
        return parsed

    last_err: Optional[Exception] = None
    for attempt in range(4):
        print(f"  Grading attempt {attempt + 1}/4...")
        data = _grok_chat(
            grok_api_key,
            messages=[system, {"role": "user", "content": instructions + "\n\nDATA:\n" + json.dumps(payload, ensure_ascii=False)}],
            temperature=0.12,
        )
        content = data["choices"][0]["message"]["content"]
        parsed = parse_json_with_repair(grok_api_key, content, debug_tag="essay_grading", max_fix_attempts=3)
        if parsed is None:
            print(f"  Parse failed on attempt {attempt + 1}")
            last_err = ValueError("JSON parsing failed")
            continue
        parsed = _coerce_grading_shape(parsed)
        parsed = _enforce_range_rules(parsed)
        if _is_valid_grading(parsed):
            print(f"  Grading validated successfully on attempt {attempt + 1}")
            return parsed
        last_err = ValueError("Invalid grading JSON: missing required fields")

    print(f"  All grading attempts failed. Last parsed data: {json.dumps(parsed, indent=2) if parsed else 'None'}")
    recovered = _enforce_range_rules(_coerce_grading_shape(parsed or {}))
    print("  WARNING: Continuing with recovered grading JSON after retries.")
    return recovered



def _norm_ws(s: str) -> str:
    """Normalize whitespace for substring matching."""
    return re.sub(r"\s+", " ", (s or "").strip())


def _anchor_is_valid(anchor: str, ocr_page_text: str) -> bool:
    """Check if anchor is a valid substring of OCR page text."""
    a = _norm_ws(anchor)
    t = _norm_ws(ocr_page_text)
    if not a or len(a.split()) < 5:
        return False
    # exact substring check (whitespace-normalized)
    return a in t


def _compact_ocr_page(page: Dict[str, Any]) -> Dict[str, Any]:
    # Keep stable per-page text blob (best for anchor_quote extraction)
    page_text = (page.get("ocr_page_text") or "").strip()
    
    # Keep lines in order with exact line text
    lines_out = []
    for line in page.get("lines", []):
        line_text = (line.get("text") or line.get("content") or "").strip()
        lines_out.append({"text": line_text})
    
    return {
        "page_number": page.get("page_number"),
        "ocr_page_text": page_text,
        "lines": lines_out,
    }


ESSAY_PAGE_SUGGESTIONS_PROMPT = (
    "PAGE SUGGESTIONS RULES (CRITICAL):\n\n"
    "- page_suggestions: 2-4 items for this page only.\n"
    "- Each suggestion MUST be an object with two fields:\n"
    "    1) \"suggestion\" (the improvement text)\n"
    "    2) \"anchor_quote\" (EXACT contiguous substring from OCR_PAGE_TEXT that this suggestion refers to).\n\n"
    "ANCHOR REQUIREMENTS:\n"
    "- The anchor_quote MUST be an EXACT contiguous substring copied from OCR_PAGE_TEXT.\n"
    "- Do NOT paraphrase the anchor.\n"
    "- The anchor_quote links the suggestion directly to the specific part of the essay being improved.\n\n"
    "CORE REQUIREMENT (MANDATORY REWRITE RULE):\n"
    "- Each suggestion MUST include a FULLY WRITTEN improved version of the referenced text.\n"
    "- Do NOT only describe the problem or explain how to improve it.\n"
    "- You MUST demonstrate the improved version exactly as it should appear in the essay.\n"
    "- The improved version must be written in full sentences and in formal academic tone.\n"
    "-Use simple sentences. Avoid Using complex sentences.\n"
    "- The improved version must preserve the original intent but increase analytical depth, specificity, precision, and argument strength.\n\n"
    "SUGGESTION STRUCTURE:\n"
    "Each suggestion must:\n"
    "1) Briefly identify the issue (1–2 clear sentences).\n"
    "2) Provide the improved version in quotation marks, clearly introduced as:\n"
    "   Improved version: \"...\"\n\n"
    "WHAT TO IMPROVE:\n"
    "- If the thesis is vague → Rewrite it into a clear, analytical, arguable thesis.\n"
    "- If an outline point is broad → Rewrite it into a precise, argument-driven claim.\n"
    "- If a claim lacks evidence → Replace it with a more specific and evidence-based version.\n"
    "- If a topic sentence is weak → Rewrite it as a strong argumentative topic sentence.\n"
    "- If reasoning lacks causation or evaluation → Rewrite it to include analytical depth (cause, consequence, qualification, comparison, or evaluation).\n\n"
    "QUALITY STANDARD:\n"
    "- The improved version must demonstrate higher-order thinking (analysis, causation, evaluation, or qualification), not just clearer wording.\n"
    "- Avoid generic advice such as \"add more detail\" or \"improve clarity.\"\n"
    "- Every suggestion must contain a concrete rewritten sample.\n"
    "- Suggestions must focus only on content, structure, argumentation, evidence, and relevance.\n\n"
    "RESTRICTIONS:\n"
    "- Do NOT include grammar or spelling corrections (handled separately).\n"
    "- Do NOT comment on numbering format, numeral structure, or point-listing style.\n"
    "- Do NOT mention OCR, scan quality, handwriting, or legibility.\n"
    "- Do NOT produce generic or repetitive suggestions.\n"
    "- Ensure variation in suggestions (e.g., thesis strength, argument depth, evidence precision, structural coherence).\n\n"
    "STYLE:\n"
    "- Use clear, complete sentences for readability.\n"
    "- Maintain academic tone appropriate for high-level competitive examinations.\n\n"
    "LENGTH RULE (MANDATORY):\n"
    "- Keep each suggestion concise: 22 to 45 words total.\n"
    "- Do not exceed 45 words in a single suggestion.\n"
    "- Keep the 'Improved version' compact but complete.\n\n"
    "Return JSON only matching schema."
)


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)?", text or ""))


_EXTRA_LINE_PATTERNS = [
    re.compile(r"\bcamscanner\b", re.IGNORECASE),
    re.compile(r"\bcs\s*camscanner\b", re.IGNORECASE),
    re.compile(r"\bdate\s*[:\-]", re.IGNORECASE),
    re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", re.IGNORECASE),
    re.compile(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}\b", re.IGNORECASE),
    re.compile(r"\b\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{2,4}\b", re.IGNORECASE),
]


def _is_extra_artifact_line(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    for pat in _EXTRA_LINE_PATTERNS:
        if pat.search(t):
            return True
    return False


def filter_essay_extra_text(ocr_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    cleaned_pages: List[Dict[str, Any]] = []
    extras_pages: List[Dict[str, Any]] = []

    for page in (ocr_data.get("pages") or []):
        page_no = int(page.get("page_number") or 0)
        kept_lines: List[Dict[str, Any]] = []
        removed_lines: List[Dict[str, Any]] = []

        for ln in (page.get("lines") or []):
            txt = (ln.get("text") or "").strip()
            if _is_extra_artifact_line(txt):
                removed_lines.append({"text": txt, "bbox": ln.get("bbox") or []})
            else:
                kept_lines.append(ln)

        rebuilt_text = " ".join((x.get("text") or "").strip() for x in kept_lines if (x.get("text") or "").strip()).strip()

        new_page = dict(page)
        new_page["lines"] = kept_lines
        new_page["ocr_page_text"] = rebuilt_text
        cleaned_pages.append(new_page)

        if removed_lines:
            extras_pages.append({"page_number": page_no, "removed_lines": removed_lines})

    cleaned = {
        "pages": cleaned_pages,
        "full_text": "\n".join((p.get("ocr_page_text") or "").strip() for p in cleaned_pages if (p.get("ocr_page_text") or "").strip()).strip(),
    }
    extras = {
        "removed_line_count": sum(len(x.get("removed_lines", [])) for x in extras_pages),
        "pages": extras_pages,
    }
    return cleaned, extras


def _load_partial_annotations(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _save_partial_annotations(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _process_annotation_page(
    page: Dict[str, Any],
    page_num: int,
    payload: Dict[str, Any],
    system: Dict[str, Any],
    instructions: str,
    grok_api_key: str,
    debug_dir: str,
    lock: threading.Lock,
) -> Tuple[int, Optional[Dict[str, Any]], Optional[str]]:
    """Process a single page for annotations (used for parallel processing)."""
    ocr_page_text = (page.get("ocr_page_text") or "").strip()
    if not ocr_page_text:
        return page_num, None, "Missing ocr_page_text (fix run_ocr_on_pdf output)."
    
    max_page_attempts = 3
    last_err = None
    parsed = None
    
    for attempt in range(1, max_page_attempts + 1):
        try:
            data = _grok_chat(
                grok_api_key,
                messages=[system, {"role": "user", "content": instructions + "\n\nDATA:\n" + json.dumps(payload, ensure_ascii=False)}],
                temperature=0.12,
                timeout=200,
                max_retries=4,
            )
            content = data["choices"][0]["message"]["content"]
            parsed = parse_json_with_repair(
                grok_api_key, 
                content, 
                debug_tag=f"essay_annotations_p{page_num}",
                debug_dir_override=debug_dir
            )
            if not isinstance(parsed, dict):
                raise ValueError("Annotation JSON is not an object")
            if not isinstance(parsed.get("annotations"), list):
                raise ValueError("Annotation JSON missing annotations list")
            if "page_suggestions" in parsed:
                parsed.pop("page_suggestions", None)
            
            # VALIDATE ANCHORS: ensure they exist in OCR text
            ann = parsed.get("annotations") or []
            valid_ann = []
            invalid_count = 0
            
            for a in ann:
                if not isinstance(a, dict):
                    continue
                aq = a.get("anchor_quote", "")
                atype = (a.get("type") or "").strip()
                
                # Allow outline_quality and introduction_quality even without perfect anchors
                if atype in ["outline_quality", "introduction_quality"]:
                    valid_ann.append(a)
                    continue
                
                # For other annotation types, validate anchor
                if not aq or not _anchor_is_valid(aq, ocr_page_text):
                    invalid_count += 1
                
                valid_ann.append(a)
            
            # Log validation result for debugging
            if invalid_count > 0:
                with lock:
                    print(f"    [Page {page_num}] Warning: {invalid_count}/{len(ann)} annotations missing valid anchor_quote")
            
            parsed["annotations"] = valid_ann
            
            # Light cleanup to keep output consistent
            cleaned = []
            for a in valid_ann:
                if not isinstance(a.get("page"), int):
                    a["page"] = page_num
                for k in ["type", "rubric_point", "anchor_quote", "target_word_or_sentence", "context_before", "context_after", "correction", "comment"]:
                    if k not in a:
                        a[k] = ""
                cleaned.append(a)
            
            return page_num, {"annotations": cleaned}, None
            
        except Exception as e:
            last_err = str(e)
            if attempt == max_page_attempts:
                return page_num, None, last_err
            continue
    
    return page_num, None, last_err


def call_grok_for_essay_annotations(
    grok_api_key: str,
    annotations_rubric_text: str,
    ocr_data: Dict[str, Any],
    structure: Dict[str, Any],
    grading: Dict[str, Any],
    page_images: List[Dict[str, Any]],
    debug_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
        Returns:
        {
            "annotations":[ ... ]
        }
    """
    system = {
        "role": "system",
        "content": (
            "You generate pinpoint annotations for handwritten CSS essays.\n"
            "Primary truth = page images; OCR is helper. Ignore OCR errors and never mention them.\n"
            "IMPORTANT: Do NOT treat events from 2025 or later years as speculation. "
            "If you encounter dates/events you don't have knowledge about, ignore them and focus on essay structure and argumentation. "
            "Never comment on whether an event is speculative based on your knowledge cutoff.\n"
            "Return JSON only."
        ),
    }

    schema_hint = {
        "page": 1,
        "annotations": [
            {
                "page": 1,
                "type": "grammar_language",
                "rubric_point": "Grammar & Language",
                "anchor_quote": "EXACT substring from OCR_PAGE_TEXT (full relevant sentence/phrase)",
                "correction": "string",
                "comment": "string",
            }
        ],
    }

    instructions = (
        "Using the ANNOTATIONS RUBRIC, generate actionable annotations for ONE PAGE only.\n"
        "Rules (MUST FOLLOW):\n"
        "- For output, use simple and complete sentences to increase readability.\n"
        "- Prefer 2-5 annotations per page.\n"
        "- Every annotation MUST be LOCATABLE on the page.\n"
        "- Annotations = rubric-point issues; keep each comment to ONE concise line that states the problem and fix (no multi-line paragraphs).\n"
        "- Do NOT comment on the numbering format, numeral structure, bullet style, or point-numbering convention "
        "used in the outline or essay body. Focus only on the substance and content quality, not how points are numbered or listed.\n"
        "\n"
        "ANCHOR RULE (CRITICAL):\n"
        "- You are given OCR_PAGE_TEXT below.\n"
        "- anchor_quote MUST be an EXACT contiguous substring copied from OCR_PAGE_TEXT.\n"
        "- Use the full relevant sentence/phrase (no upper word cap); do NOT paraphrase.\n"
        "- Do NOT correct spelling inside anchor_quote.\n"
        "- If you cannot find a suitable quote in OCR_PAGE_TEXT, set anchor_quote to empty and SKIP that annotation.\n"
        "- Ignore camera/date/watermark artifacts (e.g., CamScanner/date stamps) completely.\n"
        "\n"
        "- Use these types exactly:\n"
        "  outline_quality, introduction_quality, paragraph_flow, factual_accuracy,\n"
        "  grammar_language, repetitiveness, argumentation_depth,\n"
        "  organization_coherence, conclusion_quality, relevance_focus.\n"
        "\n"
        "MANDATORY COMMENTS (MUST ALWAYS INCLUDE):\n"
        "- For pages containing or expected to contain the OUTLINE:\n"
        "  ALWAYS generate an 'outline_quality' annotation. If outline is present, assess its quality.\n"
        "  If outline is NOT present or incomplete, use suggestive tone (e.g., 'Consider including a\n"
        "  structured outline to organize main points before writing').\n"
        "- For pages containing or expected to contain the ESSAY INTRODUCTION:\n"
        "  ALWAYS generate an 'introduction_quality' annotation. If introduction is present, assess it.\n"
        "  If introduction is NOT present or weak, use suggestive tone (e.g., 'A clear introduction\n"
        "  with thesis statement would strengthen the essay opening').\n"
        "- These annotations should appear on the FIRST relevant page (outline on page 1-2, intro on early essay pages).\n"
        "Return JSON only matching schema."
    )

    # Use provided debug_dir or create default one
    if debug_dir is None:
        eng_essay_dir = _get_eng_essay_dir()
        debug_dir = os.path.join(eng_essay_dir, "debug_llm")
    os.makedirs(debug_dir, exist_ok=True)
    partial_path = os.path.join(debug_dir, "essay_annotations_partial.json")
    
    # CRITICAL: Delete old partial file to prevent cache issues
    if os.path.exists(partial_path):
        try:
            os.remove(partial_path)
            print(f"✓ Deleted old partial annotations cache: {partial_path}")
        except Exception as e:
            print(f"⚠ Failed to delete partial cache: {e}")
    
    partial = _load_partial_annotations(partial_path)
    annotations: List[Dict[str, Any]] = partial.get("annotations") or []
    page_suggestions: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = partial.get("errors") or []
    completed_pages = set(partial.get("completed_pages") or [])

    image_by_page = {p.get("page"): p for p in page_images}
    ocr_pages = ocr_data.get("pages", [])

    grading_summary = {
        "overall_rating": grading.get("overall_rating"),
        "total_awarded_range": grading.get("total_awarded_range"),
        "criteria": grading.get("criteria", []),
    }
    structure_summary = {
        "outline": structure.get("outline"),
        "paragraph_map": structure.get("paragraph_map", []),
    }

    outline_span = structure.get("outline_span") or {}
    outline_pages_set = set()
    try:
        start_p = int(outline_span.get("start_page")) if outline_span.get("start_page") else None
        end_p = int(outline_span.get("end_page")) if outline_span.get("end_page") else None
        if start_p and end_p and end_p >= start_p:
            outline_pages_set = set(range(start_p, end_p + 1))
    except Exception:
        outline_pages_set = set()

    # PARALLEL PROCESSING: Process multiple pages concurrently for faster annotations
    print(f"  Processing {len(ocr_pages)} pages with parallel annotation generation...")
    
    pages_to_process = []
    for page in ocr_pages:
        page_num = page.get("page_number")
        if not isinstance(page_num, int):
            continue
        if page_num in completed_pages:
            continue
        
        payload = {
            "annotations_rubric_text": (annotations_rubric_text or ""),
            "grading_summary": grading_summary,
            "structure_detected": structure_summary,
            "ocr_page": _compact_ocr_page(page),
            "ocr_full_text": (ocr_data.get("full_text") or ""),
            "page_image": image_by_page.get(page_num),
            "allowed_outline_pages": sorted(outline_pages_set),
            "output_schema": schema_hint,
        }
        pages_to_process.append((page, page_num, payload))
    
    # Process pages in parallel (up to 3 concurrent pages)
    lock = threading.Lock()
    max_workers = min(3, len(pages_to_process))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_annotation_page,
                page,
                page_num,
                payload,
                system,
                instructions,
                grok_api_key,
                debug_dir,
                lock
            ): page_num
            for page, page_num, payload in pages_to_process
        }
        
        for future in as_completed(futures):
            page_num = futures[future]
            try:
                result_page_num, result, error = future.result()
                
                if error:
                    with lock:
                        errors.append({"page": result_page_num, "error": error})
                        print(f"    ✗ Page {result_page_num} failed: {error}")
                elif result:
                    with lock:
                        annotations.extend(result["annotations"])
                        completed_pages.add(result_page_num)
                        print(f"    ✓ Page {result_page_num} done ({len(result['annotations'])} annotations)")
                
                # Save progress after each page completes
                with lock:
                    _save_partial_annotations(partial_path, {
                        "annotations": annotations,
                        "errors": errors,
                        "completed_pages": sorted(completed_pages),
                    })
            
            except Exception as e:
                with lock:
                    errors.append({"page": page_num, "error": str(e)})
                    print(f"    ✗ Page {page_num} exception: {e}")

    if not annotations and errors:
        raise RuntimeError(f"All annotation requests failed. See {partial_path} for details.")

    return {"annotations": annotations, "errors": errors}


def _process_suggestion_page(
    page: Dict[str, Any],
    page_num: int,
    payload: Dict[str, Any],
    system: Dict[str, Any],
    instructions: str,
    grok_api_key: str,
    debug_dir: str,
) -> Tuple[int, Optional[List[Dict[str, str]]], Optional[str]]:
    ocr_page_text = (page.get("ocr_page_text") or "").strip()
    if not ocr_page_text:
        return page_num, None, "Missing ocr_page_text"

    last_err: Optional[str] = None
    for _ in range(3):
        try:
            data = _grok_chat(
                grok_api_key,
                messages=[system, {"role": "user", "content": instructions + "\n\nDATA:\n" + json.dumps(payload, ensure_ascii=False)}],
                temperature=0.10,
                timeout=200,
                max_retries=4,
            )
            content = data["choices"][0]["message"]["content"]
            parsed = parse_json_with_repair(
                grok_api_key,
                content,
                debug_tag=f"essay_suggestions_p{page_num}",
                debug_dir_override=debug_dir,
            )
            raw_suggestions = parsed.get("page_suggestions")
            if not isinstance(raw_suggestions, list):
                raise ValueError("page_suggestions must be a list")

            cleaned: List[Dict[str, str]] = []
            for item in raw_suggestions:
                if not isinstance(item, dict):
                    continue
                suggestion = str(item.get("suggestion", "")).strip()
                anchor = str(item.get("anchor_quote", "")).strip()
                if not suggestion or not anchor:
                    continue
                if not _anchor_is_valid(anchor, ocr_page_text):
                    continue
                wc = _word_count(suggestion)
                if wc < 12 or wc > 50:
                    continue
                cleaned.append({"suggestion": suggestion, "anchor_quote": anchor})

            return page_num, cleaned[:4], None
        except Exception as e:
            last_err = str(e)
            continue

    return page_num, None, last_err


def call_grok_for_essay_page_suggestions(
    grok_api_key: str,
    ocr_data: Dict[str, Any],
    structure: Dict[str, Any],
    grading: Dict[str, Any],
    page_images: List[Dict[str, Any]],
    debug_dir: Optional[str] = None,
) -> Dict[str, Any]:
    system = {
        "role": "system",
        "content": (
            "You generate high-quality page-wise rewrite suggestions for handwritten CSS essays.\n"
            "Primary truth = page images; OCR is helper text.\n"
            "Do NOT comment on numbering format, numeral structure, bullet style, or point-listing conventions.\n"
            "Ignore camera/date/watermark artifacts (e.g., CamScanner/date stamps) completely.\n"
            "Return JSON only."
        ),
    }

    instructions = ESSAY_PAGE_SUGGESTIONS_PROMPT

    schema_hint = {
        "page": 1,
        "page_suggestions": [
            {
                "suggestion": "Issue + rewrite guidance. Improved version: \"...\"",
                "anchor_quote": "EXACT contiguous substring from OCR_PAGE_TEXT"
            }
        ],
    }

    if debug_dir is None:
        eng_essay_dir = _get_eng_essay_dir()
        debug_dir = os.path.join(eng_essay_dir, "debug_llm")
    os.makedirs(debug_dir, exist_ok=True)

    image_by_page = {p.get("page"): p for p in page_images}
    ocr_pages = ocr_data.get("pages", [])
    errors: List[Dict[str, Any]] = []
    page_suggestions: List[Dict[str, Any]] = []

    grading_summary = {
        "overall_rating": grading.get("overall_rating"),
        "total_awarded_range": grading.get("total_awarded_range"),
        "criteria": grading.get("criteria", []),
    }
    structure_summary = {
        "outline": structure.get("outline"),
        "paragraph_map": structure.get("paragraph_map", []),
    }

    tasks: List[Tuple[Dict[str, Any], int, Dict[str, Any]]] = []
    for page in ocr_pages:
        page_num = page.get("page_number")
        if not isinstance(page_num, int):
            continue
        payload = {
            "grading_summary": grading_summary,
            "structure_detected": structure_summary,
            "ocr_page": _compact_ocr_page(page),
            "ocr_full_text": (ocr_data.get("full_text") or ""),
            "page_image": image_by_page.get(page_num),
            "output_schema": schema_hint,
        }
        tasks.append((page, page_num, payload))

    max_workers = min(3, max(1, len(tasks)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_suggestion_page,
                page,
                page_num,
                payload,
                system,
                instructions,
                grok_api_key,
                debug_dir,
            ): page_num
            for page, page_num, payload in tasks
        }
        for future in as_completed(futures):
            page_num = futures[future]
            try:
                result_page, suggestions, err = future.result()
                if err:
                    errors.append({"page": result_page, "error": err})
                    continue
                page_suggestions.append({"page": result_page, "suggestions": suggestions or []})
            except Exception as e:
                errors.append({"page": page_num, "error": str(e)})

    page_suggestions.sort(key=lambda x: int(x.get("page") or 0))
    return {"page_suggestions": page_suggestions, "errors": errors}



# -----------------------------
# Report Rendering (range-based)
# -----------------------------

def _iter_font_candidates() -> List[str]:
    pil_fonts_dir = os.path.join(os.path.dirname(ImageFont.__file__), "fonts")
    return [
        os.path.join(pil_fonts_dir, "DejaVuSans.ttf"),
        os.path.join(pil_fonts_dir, "DejaVuSans-Bold.ttf"),
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\arialbd.ttf",
        "DejaVuSans.ttf",
        "arial.ttf",
    ]


def _get_font(size: int) -> ImageFont.FreeTypeFont:
    for fp in _iter_font_candidates():
        try:
            font = ImageFont.truetype(fp, size)
            # Only log once per unique font path to avoid spam
            if not hasattr(_get_font, '_logged_fonts'):
                _get_font._logged_fonts = set()
            if fp not in _get_font._logged_fonts:
                print(f"✓ Loaded font: {fp} (size={size})")
                _get_font._logged_fonts.add(fp)
            return font
        except Exception:
            continue
    
    # CRITICAL: Pillow's default font is tiny (11px bitmap). Scale it up for readability.
    print(f"WARNING: No TrueType font found. Using Pillow default font at size {size}.")
    print(f"WARNING: This may result in smaller text. Ensure DejaVuSans.ttf is available in container.")
    # Return default but log the issue
    default_font = ImageFont.load_default()
    return default_font


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
    text = (text or "").strip()
    if not text:
        return [""]
    words = text.split()
    lines: List[str] = []
    cur = ""
    for w in words:
        test = (cur + " " + w).strip()
        if draw.textlength(test, font=font) <= max_width or not cur:
            cur = test
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def render_essay_report_pages_range(
    grading: Dict[str, Any],
    page_size: Tuple[int, int] = (2977, 4211),
) -> List[Image.Image]:
    """
    Render essay report using PyMuPDF with built-in fonts (like annotations).
    This ensures consistent font sizes in all environments without external font files.
    Returns a list of PIL Images for compatibility with merge function.
    """
    # PyMuPDF works in points (72 DPI), input page_size is in pixels at 200 DPI
    dpi_ratio = 72.0 / 200.0
    W_pt = page_size[0] * dpi_ratio
    H_pt = page_size[1] * dpi_ratio
    
    print(f"Essay report rendering with PyMuPDF: page_size=({int(W_pt)}x{int(H_pt)})pt")
    
    # Create PDF document
    doc = fitz.open()
    page = doc.new_page(width=W_pt, height=H_pt)
    
    # Font sizes in points (balanced for readability and page filling)
    title_size = 28
    header_size = 16
    cell_size = 13
    
    print(f"Essay report font sizes: title={title_size}pt, header={header_size}pt, cell={cell_size}pt")
    
    # Margins and layout - increased right padding
    margin = W_pt * 0.055
    right_margin = W_pt * 0.08  # More padding from right
    y = margin
    
    # Column widths (proportional to page width) - only Criterion and Key Comments
    table_width = W_pt - margin - right_margin
    col_criterion = table_width * 0.35
    col_comments = table_width - col_criterion
    
    # Get grading data
    topic = grading.get("topic", "")
    total_range = grading.get("total_awarded_range", "0-0")
    criteria_list = grading.get("criteria", [])
    
    # Title
    page.insert_text(
        (margin, y),
        "Essay Evaluation Report",
        fontname="hebo",  # Helvetica Bold
        fontsize=title_size,
        color=(0, 0, 0)
    )
    y += title_size * 1.5
    
    # Topic (may wrap)
    topic_text = f"Topic: {topic}"
    # Simple text wrapping for PyMuPDF
    topic_words = topic_text.split()
    topic_line = ""
    for word in topic_words:
        test_line = topic_line + word + " "
        text_width = fitz.get_text_length(test_line, fontname="hebo", fontsize=header_size)
        if text_width > W_pt - 2 * margin:
            if topic_line:
                page.insert_text((margin, y), topic_line.strip(), fontname="hebo", fontsize=header_size, color=(0, 0, 0))
                y += header_size * 1.4
            topic_line = word + " "
        else:
            topic_line = test_line
    if topic_line:
        page.insert_text((margin, y), topic_line.strip(), fontname="hebo", fontsize=header_size, color=(0, 0, 0))
        y += header_size * 1.4
    
    # Add proper gap between Topic and Total Marks
    y += 15  # Extra spacing
    
    # Total marks - bigger font and red color
    total_marks_size = header_size * 1.5  # 50% bigger
    page.insert_text(
        (margin, y),
        f"Total Marks (Range): {total_range}/100",
        fontname="hebo",
        fontsize=total_marks_size,
        color=(1, 0, 0)  # Red color
    )
    y += total_marks_size * 1.8  # More spacing after total marks
    
    # Table header - only Criterion and Key Comments
    table_x = margin
    table_w = table_width
    row_h = 42  # Increased row height for better readability
    
    headers = ["Criterion", "Key Comments"]
    header_rect = fitz.Rect(table_x, y, table_x + table_w, y + row_h)
    page.draw_rect(header_rect, color=(0, 0, 0), fill=(0.4, 0.4, 0.4), width=2)
    
    x = table_x
    splits = [col_criterion, col_comments]
    for i, htxt in enumerate(headers):
        page.insert_text((x + 5, y + 23), htxt, fontname="hebo", fontsize=header_size, color=(0, 0, 0))
        x += splits[i]
        if i < len(headers) - 1:
            page.draw_line((x, y), (x, y + row_h), color=(0, 0, 0), width=2)
    y += row_h
    
    # Table rows - only Criterion and Key Comments columns
    for idx, c in enumerate(criteria_list):
        crit = c.get("criterion", "")
        comments = str(c.get("key_comments", ""))
        
        # Estimate row height based on text wrapping - increased for better spacing
        # Simple approximation: count characters and estimate lines
        comment_chars_per_line = int((col_comments - 10) / (cell_size * 0.5))
        comment_lines = max(1, (len(comments) + comment_chars_per_line - 1) // comment_chars_per_line)
        
        crit_chars_per_line = int((col_criterion - 10) / (cell_size * 0.5))
        crit_lines = max(1, (len(crit) + crit_chars_per_line - 1) // crit_chars_per_line)
        
        row_h = max(45, max(comment_lines, crit_lines) * cell_size * 1.7)  # Taller rows with more spacing
        
        # Alternating row color
        fill_color = (0.8, 0.8, 0.8) if idx % 2 == 0 else (1, 1, 1)
        row_rect = fitz.Rect(table_x, y, table_x + table_w, y + row_h)
        page.draw_rect(row_rect, color=(0, 0, 0), fill=fill_color, width=1)
        
        # Draw cell content
        x = table_x
        
        # Criterion (with wrapping)
        crit_y = y + 18
        crit_words = crit.split()
        crit_line = ""
        for word in crit_words:
            test_line = crit_line + word + " "
            text_width = fitz.get_text_length(test_line, fontname="helv", fontsize=cell_size)
            if text_width > col_criterion - 10:
                if crit_line:
                    page.insert_text((x + 5, crit_y), crit_line.strip(), fontname="helv", fontsize=cell_size, color=(0, 0, 0))
                    crit_y += cell_size * 1.35
                crit_line = word + " "
            else:
                crit_line = test_line
        if crit_line:
            page.insert_text((x + 5, crit_y), crit_line.strip(), fontname="helv", fontsize=cell_size, color=(0, 0, 0))
        
        x += col_criterion
        page.draw_line((x, y), (x, y + row_h), color=(0, 0, 0), width=1)
        
        # Key Comments (with wrapping)
        comment_y = y + 18
        comment_words = comments.split()
        comment_line = ""
        for word in comment_words:
            test_line = comment_line + word + " "
            text_width = fitz.get_text_length(test_line, fontname="helv", fontsize=cell_size)
            if text_width > col_comments - 10:
                if comment_line:
                    page.insert_text((x + 5, comment_y), comment_line.strip(), fontname="helv", fontsize=cell_size, color=(0, 0, 0))
                    comment_y += cell_size * 1.35
                comment_line = word + " "
            else:
                comment_line = test_line
        if comment_line:
            page.insert_text((x + 5, comment_y), comment_line.strip(), fontname="helv", fontsize=cell_size, color=(0, 0, 0))
        
        y += row_h
    
    # Add bullet sections - fit within remaining page space
    y += 20
    
    # Calculate available space for bullet sections
    bottom_margin = W_pt * 0.06  # Bottom margin padding
    available_height = H_pt - y - bottom_margin
    
    # Collect all bullet section data
    reasons = grading.get("reasons_for_low_score", []) or ["(Not provided)"]
    improvements = grading.get("suggested_improvements_for_higher_score_70_plus", []) or ["(Not provided)"]
    
    # Estimate total content lines to determine if we need to shrink fonts
    def _count_bullet_lines(bullets: List[str], font_name: str, font_sz: float, max_w: float) -> int:
        total = 0
        for b in bullets:
            words = b.split()
            line = ""
            lines = 0
            for word in words:
                test = line + word + " "
                tw = fitz.get_text_length(test, fontname=font_name, fontsize=font_sz)
                if tw > max_w:
                    if line:
                        lines += 1
                    line = word + " "
                else:
                    line = test
            if line:
                lines += 1
            total += max(1, lines)
        return total
    
    # Try progressively smaller font sizes until content fits
    # Start with ideal sizes and shrink if needed
    section_title_size = min(22, title_size - 4)
    bullet_font_size = min(13, header_size - 2)
    bullet_line_spacing = 1.35
    bullet_gap = 8  # Between bullets
    section_gap = 16  # Between sections
    
    text_max_w = W_pt - margin - right_margin - 45  # Account for indent
    
    for shrink_attempt in range(8):
        # Calculate estimated height for both sections
        reasons_lines = _count_bullet_lines(reasons, "helv", bullet_font_size, text_max_w)
        improvements_lines = _count_bullet_lines(improvements, "helv", bullet_font_size, text_max_w)
        
        est_height = (
            (section_title_size * 1.5) +  # First section title
            reasons_lines * (bullet_font_size * bullet_line_spacing) +
            len(reasons) * bullet_gap +
            section_gap +
            (section_title_size * 1.5) +  # Second section title
            improvements_lines * (bullet_font_size * bullet_line_spacing) +
            len(improvements) * bullet_gap +
            section_gap
        )
        
        if est_height <= available_height:
            break
        # Shrink fonts
        section_title_size = max(12, section_title_size - 1.5)
        bullet_font_size = max(8.5, bullet_font_size - 0.8)
        bullet_line_spacing = max(1.2, bullet_line_spacing - 0.02)
        bullet_gap = max(4, bullet_gap - 1)
        section_gap = max(8, section_gap - 2)
        text_max_w = W_pt - margin - right_margin - 40
    
    print(f"  Report bullet sections: title={section_title_size:.1f}pt, bullet={bullet_font_size:.1f}pt, available={available_height:.0f}pt, est={est_height:.0f}pt")
    
    def draw_bullet_section(title: str, bullets: List[str]) -> None:
        nonlocal y
        # Check if we're too close to bottom
        if y + section_title_size * 2 > H_pt - bottom_margin:
            return
        
        page.insert_text((margin, y), title, fontname="hebo", fontsize=section_title_size, color=(0.2, 0.2, 0.2))
        y += section_title_size * 1.5
        
        if not bullets:
            bullets = ["(Not provided)"]
        
        bullet_indent = margin + 12
        text_indent = bullet_indent + 16
        
        for bullet in bullets:
            # Stop if we're running out of page space
            if y + bullet_font_size * 2 > H_pt - bottom_margin:
                break
            
            # Draw bullet point (circle)
            bullet_y = y - 3
            page.draw_circle((bullet_indent + 3, bullet_y), 2.5, color=(0, 0, 0), fill=(0, 0, 0))
            
            # Word wrap the bullet text
            bullet_words = bullet.split()
            bullet_line = ""
            first_line = True
            line_y = y
            
            for word in bullet_words:
                test_line = bullet_line + word + " "
                tw = fitz.get_text_length(test_line, fontname="helv", fontsize=bullet_font_size)
                
                if tw > text_max_w:
                    if bullet_line:
                        # Stop if next line would overflow page
                        if line_y + bullet_font_size * bullet_line_spacing > H_pt - bottom_margin:
                            break
                        x_pos = text_indent
                        page.insert_text((x_pos, line_y), bullet_line.strip(), fontname="helv", fontsize=bullet_font_size, color=(0.1, 0.1, 0.1))
                        line_y += bullet_font_size * bullet_line_spacing
                        first_line = False
                    bullet_line = word + " "
                else:
                    bullet_line = test_line
            
            if bullet_line and line_y + bullet_font_size < H_pt - bottom_margin:
                x_pos = text_indent
                page.insert_text((x_pos, line_y), bullet_line.strip(), fontname="helv", fontsize=bullet_font_size, color=(0.1, 0.1, 0.1))
                line_y += bullet_font_size * bullet_line_spacing
            
            y = line_y + bullet_gap
        
        y += section_gap
    
    draw_bullet_section("Reasons for Low Score", reasons)
    draw_bullet_section("Suggested Improvements for Higher Score", improvements)
    
    # Convert PDF page to PIL Image
    # Render at original pixel dimensions
    mat = fitz.Matrix(200.0 / 72.0, 200.0 / 72.0)  # Scale back to 200 DPI
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    doc.close()
    
    print(f"✓ Essay report rendered successfully with PyMuPDF built-in fonts")
    print(f"  Final page dimensions: {img.width}x{img.height}px")
    
    return [img]



# -----------------------------
# Merge pages into final PDF
# -----------------------------

def pil_images_to_pdf_bytes(pages: List[Image.Image]) -> bytes:
    out = io.BytesIO()
    if not pages:
        return b""
    pages_rgb = [p.convert("RGB") for p in pages]
    pages_rgb[0].save(out, format="PDF", save_all=True, append_images=pages_rgb[1:])
    return out.getvalue()


def add_spelling_annotations_to_pdf(
    input_pdf_path: str,
    output_pdf_path: str,
    ocr_data: Dict[str, Any],
    spelling_errors: List[Dict[str, Any]],
) -> None:
    """
    Add PyMuPDF-based spelling annotations to the original PDF.
    This should be called BEFORE essay grading annotations.
    """
    try:
        from .annotate_pdf_with_essay_rubric import (
            _word_rects_in_page_coords_fitz,
            _find_error_word_span_fitz,
        )
    except (ImportError, ModuleNotFoundError):
        from annotate_pdf_with_essay_rubric import (
            _word_rects_in_page_coords_fitz,
            _find_error_word_span_fitz,
        )
    
    src_doc = fitz.open(input_pdf_path)
    pages_data = ocr_data.get("pages", [])
    matched_count = 0
    total_count = len(spelling_errors)
    
    for error in spelling_errors:
        page_num = error.get("page", 1) - 1  # Convert to 0-indexed
        
        if page_num < 0 or page_num >= len(src_doc):
            continue
        
        page = src_doc[page_num]
        page_info = pages_data[page_num] if page_num < len(pages_data) else {}
        
        error_text = error.get("error_text", "")
        correction = error.get("correction", "")
        anchor_quote = error.get("anchor_quote")
        
        if not error_text or not correction:
            continue
        
        # Get word rectangles from OCR
        wordrects = _word_rects_in_page_coords_fitz(page_info)
        if not wordrects:
            print(f"⚠ No wordrects for page {page_num + 1}")
            continue
        
        # Find the error location
        rect = _find_error_word_span_fitz(wordrects, error_text, anchor_quote)
        if not rect:
            # Debug: show first 5 words to help diagnose
            sample_words = [w[2] for w in wordrects[:5]]
            print(f"⚠ Could not locate error '{error_text}' on page {page_num + 1} (sample words: {sample_words})")
            continue
        
        # Scale from Azure OCR coordinates to actual PDF page coordinates
        page_w = float(page_info.get("page_width") or page.rect.width)
        page_h = float(page_info.get("page_height") or page.rect.height)
        unit = page_info.get("unit", "pixel")
        azure_scale = 72.0 if unit.lower() == "inch" else 1.0
        
        if page_w > 0 and page_h > 0:
            sx = page.rect.width / (page_w * azure_scale)
            sy = page.rect.height / (page_h * azure_scale)
            rect = rect * fitz.Matrix(sx, sy)
        
        matched_count += 1
        
        # Draw red rectangle around error with thicker border
        page.draw_rect(rect, color=(0.8, 0, 0), width=2.5)
        
        # Prepare correction text without prefix
        correction_text = correction
        
        # Use fixed font size for consistency
        font_size = 11
        text_width = fitz.get_text_length(correction_text, fontname="hebo", fontsize=font_size)
        
        text_height = 14  # Approximate height for this font size
        padding_x = 8  # Increased horizontal padding
        padding_y = 4  # Vertical padding
        margin_from_edge = 8
        
        # Calculate required box width with extra padding
        box_width = text_width + padding_x * 2 + 4
        box_height = text_height + padding_y * 2
        
        # Determine best position: try above first, then below - with overlap detection
        # Check if placing above would overlap with another word/correction
        above_candidate = fitz.Rect(
            rect.x0, rect.y0 - box_height - 5,
            rect.x0 + box_width, rect.y0 - 5
        )
        above_overlaps = False
        for wr in wordrects:
            wr_rect = wr[0]  # index 0 = fitz.Rect, index 1 = confidence float, index 2 = word text
            # Scale wr_rect the same way as error rect for proper comparison
            if page_w > 0 and page_h > 0:
                wr_rect_scaled = wr_rect * fitz.Matrix(sx, sy)
            else:
                wr_rect_scaled = wr_rect
            # Check intersection
            if (wr_rect_scaled.x0 < above_candidate.x1 and wr_rect_scaled.x1 > above_candidate.x0 and
                wr_rect_scaled.y0 < above_candidate.y1 and wr_rect_scaled.y1 > above_candidate.y0):
                above_overlaps = True
                break
        
        positions = []
        
        if not above_overlaps and rect.y0 >= box_height + 8:
            # Above is clear - use it (priority 1)
            positions.append({
                'x': rect.x0,
                'y': rect.y0 - box_height - 5,
                'width': box_width,
                'height': box_height,
                'priority': 1
            })
        
        # Try below (priority 2 if above has no overlap, priority 1 if above overlaps)
        if rect.y1 + box_height + 8 < page.rect.height:
            positions.append({
                'x': rect.x0,
                'y': rect.y1 + 5,
                'width': box_width,
                'height': box_height,
                'priority': 1 if above_overlaps else 2
            })
        
        if not positions:
            # Fallback: place above with clipping
            text_x = max(margin_from_edge, min(rect.x0, page.rect.width - box_width - margin_from_edge))
            text_y = max(box_height + padding_y, rect.y0 - box_height - 5)
        else:
            # Use highest priority position
            positions.sort(key=lambda p: p['priority'])
            best_pos = positions[0]
            text_x = best_pos['x']
            text_y = best_pos['y']
        
        # Constrain to page bounds - shift box if it goes off right edge
        if text_x + box_width > page.rect.width - margin_from_edge:
            text_x = page.rect.width - box_width - margin_from_edge
        
        # Ensure not off left edge
        text_x = max(margin_from_edge, text_x)
        
        # Constrain vertically
        text_y = max(padding_y, min(text_y, page.rect.height - box_height - padding_y))
        
        # Draw white background with red border for correction text - make it wider
        bg_rect = fitz.Rect(
            text_x - padding_x,
            text_y - padding_y,
            text_x + text_width + padding_x + 2,
            text_y + text_height + padding_y
        )
        
        # Ensure bg_rect stays in bounds
        if bg_rect.x1 > page.rect.width:
            bg_rect = fitz.Rect(page.rect.width - box_width, bg_rect.y0, page.rect.width - margin_from_edge, bg_rect.y1)
        if bg_rect.x0 < 0:
            bg_rect = fitz.Rect(margin_from_edge, bg_rect.y0, margin_from_edge + box_width, bg_rect.y1)
        
        page.draw_rect(bg_rect, color=(0.8, 0, 0), fill=(1, 1, 1), width=1.5)
        
        # Insert correction text in bold dark red
        text_point = fitz.Point(text_x, text_y + text_height - 3)
        page.insert_text(
            text_point,
            correction_text,
            fontsize=font_size,
            color=(0.8, 0, 0),
            fontname="hebo"
        )
    
    src_doc.save(output_pdf_path)
    src_doc.close()
    
    if spelling_errors:
        print(f"✓ Rendered {matched_count}/{total_count} spelling/grammar annotations using PyMuPDF")


def merge_report_and_annotated_answer(
    report_pages: List[Image.Image],
    annotated_pages: List[Image.Image],
    output_pdf_path: str,
) -> None:
    """
    Merge report pages and annotated answer pages into a final PDF.
    """
    report_pdf = pil_images_to_pdf_bytes(report_pages)
    answer_pdf = pil_images_to_pdf_bytes(annotated_pages)

    out_doc = fitz.open()
    if report_pdf:
        rdoc = fitz.open("pdf", report_pdf)
        out_doc.insert_pdf(rdoc)
        rdoc.close()
    
    if answer_pdf:
        adoc = fitz.open("pdf", answer_pdf)
        out_doc.insert_pdf(adoc)
        adoc.close()

    out_doc.save(output_pdf_path)
    out_doc.close()


# -----------------------------
# Main
# -----------------------------

def run_essay_grading(
    pdf_path: str,
    output_json_path: str,
    output_pdf_path: str,
    essay_rubric_docx: str = "CSS English Essay Evaluation Rubric Based on FPSC Examiners.docx",
    annotations_rubric_docx: str = "ANNOTATIONS RUBRIC FOR ESSAY.docx",
    report_format_docx: str = "Report Format.docx",
    ocr_workers: int = 3,
    debug_ocr_pages_dir: str = "",
    debug_structure_json: str = "",
    debug_ocr_json: str = "",
    progress_callback: Optional[callable] = None,
    job_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Programmatic entry point for essay grading pipeline.
    Called from API routes to grade PDFs with optional progress tracking.
    
    Args:
        pdf_path: Path to input PDF
        output_json_path: Path to save grading JSON
        output_pdf_path: Path to save annotated PDF
        essay_rubric_docx: Path to essay rubric document
        annotations_rubric_docx: Path to annotations rubric document
        report_format_docx: Path to report format document
        ocr_workers: Number of parallel OCR workers
        debug_ocr_pages_dir: Directory to save OCR debug pages (empty to disable)
        debug_structure_json: Path to save structure debug JSON (empty to disable)
        debug_ocr_json: Path to save OCR debug JSON (empty to disable)
        progress_callback: Optional callback function(percentage: float, message: str)
    
    Returns:
        Dict with status, paths, and grading results
    """
    validate_input_paths(pdf_path, output_json_path, output_pdf_path)
    grok_key, doc_client = load_environment()
    
    essay_rubric_text = load_essay_rubric_text(essay_rubric_docx)
    annotations_rubric_text = load_annotations_rubric_text(annotations_rubric_docx)
    report_format_text = load_report_format_text(report_format_docx)

    total_start = time.perf_counter()
    timings: Dict[str, float] = {}
    
    # Create unique timestamped folders for this job to prevent mixing between concurrent requests
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_suffix = f"{job_id}_{timestamp}" if job_id else timestamp
    eng_essay_dir = _get_eng_essay_dir()
    debug_folder_name = f"debug_llm_{job_suffix}"
    grok_folder_name = f"grok_images_essay_{job_suffix}"
    debug_dir = os.path.join(eng_essay_dir, debug_folder_name)
    grok_images_dir = os.path.join(eng_essay_dir, grok_folder_name)
    
    print(f"Using unique folders: {debug_folder_name}, {grok_folder_name}")

    if progress_callback:
        progress_callback(10, "Running OCR on PDF...")
    
    print("Running OCR (Azure Document Intelligence)...")
    t0 = time.perf_counter()
    ocr_data_raw = run_ocr_on_pdf(
        doc_client,
        pdf_path,
        workers=ocr_workers,
        debug_pages_dir=debug_ocr_pages_dir or None,
    )
    ocr_data, extra_things = filter_essay_extra_text(ocr_data_raw)
    timings["OCR extraction"] = time.perf_counter() - t0
    print(f"OCR done. Time: {_format_duration(timings['OCR extraction'])}")
    
    if debug_ocr_json:
        os.makedirs(os.path.dirname(debug_ocr_json), exist_ok=True)
        with open(debug_ocr_json, "w", encoding="utf-8") as f:
            json.dump(ocr_data, f, indent=2, ensure_ascii=False)

    if progress_callback:
        progress_callback(20, "Preparing page images...")
    
    page_images = pdf_to_page_images_for_grok(pdf_path, output_dir=grok_images_dir)

    if progress_callback:
        progress_callback(30, "Analyzing essay structure...")
    
    print("Analyzing essay structure...")
    t0 = time.perf_counter()
    structure = call_grok_for_essay_structure_paragraphs_only(
        grok_key,
        ocr_data,
        page_images,
    )
    timings["Structure analysis"] = time.perf_counter() - t0
    print(f"Structure analysis done. Time: {_format_duration(timings['Structure analysis'])}")
    
    if debug_structure_json:
        os.makedirs(os.path.dirname(debug_structure_json), exist_ok=True)
        with open(debug_structure_json, "w", encoding="utf-8") as f:
            json.dump(structure, f, ensure_ascii=False, indent=2)

    if progress_callback:
        progress_callback(50, "Grading essay content...")
    
    print("Grading essay content (STRICT range marking)...")
    t0 = time.perf_counter()
    grading = call_grok_for_essay_grading_strict_range(
        grok_key,
        essay_rubric_text=essay_rubric_text,
        report_format_text=report_format_text,
        ocr_data=ocr_data,
        structure=structure,
        page_images=page_images,
    )
    timings["Content grading"] = time.perf_counter() - t0
    print(f"Content grading done. Time: {_format_duration(timings['Content grading'])}")

    if progress_callback:
        progress_callback(65, "Checking spelling and grammar...")
    
    print("Detecting spelling/grammar errors...")
    t0 = time.perf_counter()
    spelling_errors = detect_spelling_grammar_errors(grok_key, ocr_data)
    spelling_errors = _filter_errors(spelling_errors)
    timings["Spelling detection"] = time.perf_counter() - t0
    print(f"Spelling detection done. Time: {_format_duration(timings['Spelling detection'])}")
    print(f"Found {len(spelling_errors)} spelling/grammar errors.")

    if progress_callback:
        progress_callback(70, "Generating annotations...")
    
    print("Generating annotations...")
    t0 = time.perf_counter()
    ann_pack = call_grok_for_essay_annotations(
        grok_key,
        annotations_rubric_text=annotations_rubric_text,
        ocr_data=ocr_data,
        structure=structure,
        grading=grading,
        page_images=page_images,
        debug_dir=debug_dir,
    )
    timings["Annotations"] = time.perf_counter() - t0
    print(f"Annotations done. Time: {_format_duration(timings['Annotations'])}")

    print("Generating page suggestions (separate call)...")
    t0 = time.perf_counter()
    sugg_pack = call_grok_for_essay_page_suggestions(
        grok_key,
        ocr_data=ocr_data,
        structure=structure,
        grading=grading,
        page_images=page_images,
        debug_dir=debug_dir,
    )
    timings["Page suggestions"] = time.perf_counter() - t0
    print(f"Page suggestions done. Time: {_format_duration(timings['Page suggestions'])}")

    annotations = ann_pack.get("annotations") or []
    page_suggestions = sugg_pack.get("page_suggestions") or []
    ann_errors = ann_pack.get("errors") or []
    suggestion_errors = sugg_pack.get("errors") or []
    print(f"Annotations: {len(annotations)}")
    print(f"Spelling/Grammar errors: {len(spelling_errors)}")

    # Save JSON
    output_data = {
        "structure": structure,
        "grading": grading,
        "annotations": annotations,
        "page_suggestions": page_suggestions,
        "annotation_errors": ann_errors,
        "suggestion_errors": suggestion_errors,
        "spelling_grammar_errors": spelling_errors,
        "extra_things": extra_things,
    }
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"Saved grading JSON → {output_json_path}")

    temp_spelling_pdf = None
    pdf_for_grading = pdf_path
    # Separate grammar_language annotations for inline display
    # grammar_language -> inline display (like spelling errors)
    # outline_quality, introduction_quality, and other types -> right-side margin display
    grammar_language_annotations = [a for a in annotations if a.get("type") == "grammar_language"]
    other_annotations = [a for a in annotations if a.get("type") != "grammar_language"]
    
    print(f"Annotation breakdown: {len(grammar_language_annotations)} grammar (inline), {len(other_annotations)} other (right-side)")
    
    # Convert grammar_language annotations to spelling error format for inline display
    grammar_errors_inline = []
    for ann in grammar_language_annotations:
        error_item = {
            "page": ann.get("page", 1),
            "error_text": ann.get("target_word_or_sentence", "") or ann.get("anchor_quote", ""),
            "error_type": ann.get("rubric_point", "Grammar/Language"),
            "suggestion": ann.get("correction", "") or ann.get("comment", ""),
        }
        grammar_errors_inline.append(error_item)
    
    # Combine grammar errors with spelling errors for inline display
    all_inline_errors = spelling_errors + grammar_errors_inline
    print(f"Total inline errors: {len(spelling_errors)} spelling + {len(grammar_errors_inline)} grammar = {len(all_inline_errors)}")
    
    if all_inline_errors:
        if progress_callback:
            progress_callback(80, "Marking spelling and grammar errors on PDF...")
        
        print("Creating spelling+grammar marked PDF...")
        t0 = time.perf_counter()
        import tempfile
        temp_spelling_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
        add_spelling_annotations_to_pdf(
            input_pdf_path=pdf_path,
            output_pdf_path=temp_spelling_pdf,
            ocr_data=ocr_data,
            spelling_errors=all_inline_errors,  # Include both spelling and grammar errors
        )
        pdf_for_grading = temp_spelling_pdf
        timings["Spelling+Grammar marking"] = time.perf_counter() - t0
        print(f"Spelling+Grammar marking done. Time: {_format_duration(timings['Spelling+Grammar marking'])}")

    if progress_callback:
        progress_callback(85, "Rendering report and annotations...")
    
    print("Rendering report + annotations...")
    t0 = time.perf_counter()
    calculated_size = get_report_page_size(pdf_for_grading)
    print(f"Calculated report page size: {calculated_size[0]}x{calculated_size[1]}")
    
    # CRITICAL FIX: Ensure minimum report page size to prevent tiny fonts in deployed version
    # Even if PDF is rendered at low DPI, report must maintain readable font sizes
    MIN_REPORT_WIDTH = 2977  # Standard A4 width at 200 DPI
    MIN_REPORT_HEIGHT = 4211  # Standard A4 height at 200 DPI
    page_size = (max(calculated_size[0], MIN_REPORT_WIDTH), max(calculated_size[1], MIN_REPORT_HEIGHT))
    
    if page_size != calculated_size:
        print(f"✓ Enforced minimum page size: {page_size[0]}x{page_size[1]} (was {calculated_size[0]}x{calculated_size[1]})")
        print(f"  This ensures consistent font sizes between local and deployed environments.")
    
    report_pages = render_essay_report_pages_range(grading, page_size=page_size)

    annotated_pages = annotate_pdf_essay_pages(
        pdf_path=pdf_for_grading,
        ocr_data=ocr_data,
        structure=structure,
        grading=grading,
        annotations=other_annotations,  # Only non-grammar_language annotations for right side
        page_suggestions=page_suggestions,
        spelling_errors=None,  # Already added inline in step 1
    )

    merge_report_and_annotated_answer(
        report_pages,
        annotated_pages,
        output_pdf_path,
    )

    if temp_spelling_pdf:
        try:
            os.unlink(temp_spelling_pdf)
        except Exception:
            pass

    timings["Rendering"] = time.perf_counter() - t0
    print(f"Rendering done. Time: {_format_duration(timings['Rendering'])}")
    print(f"Saved annotated PDF → {output_pdf_path}")

    # Check PDF file size and compress if needed
    print("Checking PDF file size for compression...")
    t_compress = time.perf_counter()
    compression_performed = compress_pdf_if_needed(
        pdf_path=output_pdf_path,
        target_size_mb=10.0,
        max_quality=75,
        max_dimension=2000,
    )
    timings["PDF Compression"] = time.perf_counter() - t_compress
    if compression_performed:
        print(f"PDF compression completed. Time: {_format_duration(timings['PDF Compression'])}")
    else:
        print(f"PDF compression skipped (not needed). Time: {_format_duration(timings['PDF Compression'])}")

    total_elapsed = time.perf_counter() - total_start
    print("")
    print("=" * 60)
    print("ESSAY GRADING TIMING SUMMARY")
    print("=" * 60)
    for phase, elapsed in timings.items():
        print(f"  {phase}: {_format_duration(elapsed)}")
    print("-" * 60)
    print(f"  Total essay grading time: {_format_duration(total_elapsed)}")
    print("=" * 60)

    if progress_callback:
        progress_callback(100, "Essay grading complete!")

    return {
        "status": "success",
        "json_path": output_json_path,
        "pdf_path": output_pdf_path,
        "grading": grading,
        "timings": timings,
        "total_time": total_elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="Input essay PDF path")
    parser.add_argument("--output-json", default="essay_result.json")
    parser.add_argument("--output-pdf", default="essay_annotated.pdf")
    parser.add_argument("--essay-rubric-docx", default="CSS English Essay Evaluation Rubric Based on FPSC Examiners.docx")
    parser.add_argument("--annotations-rubric-docx", default="ANNOTATIONS RUBRIC FOR ESSAY.docx")
    parser.add_argument("--report-format-docx", default="Report Format.docx")
    parser.add_argument("--ocr-workers", type=int, default=3, help="Parallel Azure OCR workers (pages in flight)")
    parser.add_argument(
        "--debug-ocr-pages-dir",
        default="debug_llm/ocr_pages",
        help="Directory to save per-page OCR debug JSON with bounding boxes (set empty to disable)",
    )
    parser.add_argument(
        "--debug-structure-json",
        default="debug_llm/structure_raw.json",
        help="Optional path to save raw structure result",
    )
    parser.add_argument(
        "--debug-ocr-json",
        default="debug_llm/ocr_full.json",
        help="Optional path to save full OCR output for debugging",
    )
    args = parser.parse_args()
    
    import os
    import tempfile

    validate_input_paths(args.pdf, args.output_json, args.output_pdf)

    grok_key, doc_client = load_environment()

    essay_rubric_text = load_essay_rubric_text(args.essay_rubric_docx)
    annotations_rubric_text = load_annotations_rubric_text(args.annotations_rubric_docx)
    report_format_text = load_report_format_text(args.report_format_docx)

    total_start = time.perf_counter()
    timings: Dict[str, float] = {}

    print("Running OCR (Azure Document Intelligence)...")
    t0 = time.perf_counter()
    ocr_data_raw = run_ocr_on_pdf(
        doc_client,
        args.pdf,
        workers=args.ocr_workers,
        debug_pages_dir=args.debug_ocr_pages_dir or None,
    )
    ocr_data, extra_things = filter_essay_extra_text(ocr_data_raw)
    timings["OCR extraction"] = time.perf_counter() - t0
    print(f"OCR done. Time: {_format_duration(timings['OCR extraction'])}")
    if args.debug_ocr_json:
        os.makedirs(os.path.dirname(args.debug_ocr_json), exist_ok=True)
        with open(args.debug_ocr_json, "w", encoding="utf-8") as f:
            f.write(ocr_data.get("full_text", ""))
        print(f"OCR full text saved to {args.debug_ocr_json}")

    page_images = pdf_to_page_images_for_grok(args.pdf)

    print("Calling Grok for structure detection (outline first)...")
    t0 = time.perf_counter()
    structure = call_grok_for_essay_structure_paragraphs_only(grok_key, ocr_data, page_images)
    timings["Grok structure detection"] = time.perf_counter() - t0
    print(f"Structure detected. Time: {_format_duration(timings['Grok structure detection'])}")
    if args.debug_structure_json:
        os.makedirs(os.path.dirname(args.debug_structure_json), exist_ok=True)
        with open(args.debug_structure_json, "w", encoding="utf-8") as f:
            json.dump(structure, f, ensure_ascii=False, indent=2)
        print(f"Structure saved to {args.debug_structure_json}")

    print("Calling Grok for STRICT range grading...")
    t0 = time.perf_counter()
    grading = call_grok_for_essay_grading_strict_range(
        grok_key,
        essay_rubric_text=essay_rubric_text,
        report_format_text=report_format_text,
        ocr_data=ocr_data,
        structure=structure,
        page_images=page_images,
    )
    timings["Strict range grading"] = time.perf_counter() - t0
    print(f"Grading done. Time: {_format_duration(timings['Strict range grading'])}")
    
    
    print("Detecting spelling and grammar errors...")
    t0 = time.perf_counter()
    spelling_errors = detect_spelling_grammar_errors(grok_key, ocr_data)
    spelling_errors = _filter_errors(spelling_errors)
    timings["Spelling and grammar detection"] = time.perf_counter() - t0
    print(f"Found {len(spelling_errors)} spelling/grammar errors. Time: {_format_duration(timings['Spelling and grammar detection'])}")
    
    print("Calling Grok for annotations...")
    t0 = time.perf_counter()
    ann_pack = call_grok_for_essay_annotations(
        grok_key,
        annotations_rubric_text=annotations_rubric_text,
        ocr_data=ocr_data,
        structure=structure,
        grading=grading,
        page_images=page_images,
    )
    timings["Annotations"] = time.perf_counter() - t0
    print(f"Annotations done. Time: {_format_duration(timings['Annotations'])}")

    print("Calling Grok for page suggestions (separate call)...")
    t0 = time.perf_counter()
    sugg_pack = call_grok_for_essay_page_suggestions(
        grok_key,
        ocr_data=ocr_data,
        structure=structure,
        grading=grading,
        page_images=page_images,
    )
    timings["Page suggestions"] = time.perf_counter() - t0
    print(f"Page suggestions done. Time: {_format_duration(timings['Page suggestions'])}")

    annotations = ann_pack.get("annotations") or []
    page_suggestions = sugg_pack.get("page_suggestions") or []
    ann_errors = ann_pack.get("errors") or []
    suggestion_errors = sugg_pack.get("errors") or []
    print(f"Annotations: {len(annotations)}")
    print(f"Spelling/Grammar errors: {len(spelling_errors)}")

    output = {
        "structure": structure,
        "grading": grading,
        "annotations": annotations,
        "page_suggestions": page_suggestions,
        "annotation_errors": ann_errors,
        "suggestion_errors": suggestion_errors,
        "spelling_grammar_errors": spelling_errors,
        "extra_things": extra_things,
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Saved JSON → {args.output_json}")

    # Separate grammar_language annotations for inline display
    # grammar_language -> inline display (like spelling errors)
    # outline_quality and other types -> right-side margin display
    grammar_language_annotations = [a for a in annotations if a.get("type") == "grammar_language"]
    other_annotations = [a for a in annotations if a.get("type") != "grammar_language"]
    
    print(f"Annotation breakdown: {len(grammar_language_annotations)} grammar (inline), {len(other_annotations)} other (right-side)")
    
    # Convert grammar_language annotations to spelling error format for inline display
    grammar_errors_inline = []
    for ann in grammar_language_annotations:
        error_item = {
            "page": ann.get("page", 1),
            "error_text": ann.get("target_word_or_sentence", "") or ann.get("anchor_quote", ""),
            "error_type": ann.get("rubric_point", "Grammar/Language"),
            "suggestion": ann.get("correction", "") or ann.get("comment", ""),
        }
        grammar_errors_inline.append(error_item)
    
    # Combine grammar errors with spelling errors for inline display
    all_inline_errors = spelling_errors + grammar_errors_inline
    print(f"Total inline errors: {len(spelling_errors)} spelling + {len(grammar_errors_inline)} grammar = {len(all_inline_errors)}")
    
    # STEP 1–3: Rendering (spelling+grammar annotations, report + annotated pages, merge)
    print("Rendering report and annotated PDF...")
    t0 = time.perf_counter()
    temp_spelling_pdf = None
    pdf_for_grading = args.pdf

    if all_inline_errors:
        temp_spelling_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
        add_spelling_annotations_to_pdf(
            input_pdf_path=args.pdf,
            output_pdf_path=temp_spelling_pdf,
            ocr_data=ocr_data,
            spelling_errors=all_inline_errors,  # Include both spelling and grammar errors
        )
        pdf_for_grading = temp_spelling_pdf

    calculated_size = get_report_page_size(pdf_for_grading)
    print(f"Calculated report page size: {calculated_size[0]}x{calculated_size[1]}")
    
    # CRITICAL FIX: Ensure minimum report page size to prevent tiny fonts in deployed version
    # Even if PDF is rendered at low DPI, report must maintain readable font sizes
    MIN_REPORT_WIDTH = 2977  # Standard A4 width at 200 DPI
    MIN_REPORT_HEIGHT = 4211  # Standard A4 height at 200 DPI
    page_size = (max(calculated_size[0], MIN_REPORT_WIDTH), max(calculated_size[1], MIN_REPORT_HEIGHT))
    
    if page_size != calculated_size:
        print(f"✓ Enforced minimum page size: {page_size[0]}x{page_size[1]} (was {calculated_size[0]}x{calculated_size[1]})")
        print(f"  This ensures consistent font sizes between local and deployed environments.")
    
    report_pages = render_essay_report_pages_range(grading, page_size=page_size)

    annotated_pages = annotate_pdf_essay_pages(
        pdf_path=pdf_for_grading,
        ocr_data=ocr_data,
        structure=structure,
        grading=grading,
        annotations=other_annotations,  # Only non-grammar_language annotations for right side
        page_suggestions=page_suggestions,
        spelling_errors=None,  # Already added in step 1
    )

    merge_report_and_annotated_answer(
        report_pages,
        annotated_pages,
        args.output_pdf,
    )

    if temp_spelling_pdf:
        try:
            os.unlink(temp_spelling_pdf)
        except Exception:
            pass

    timings["Rendering"] = time.perf_counter() - t0
    print(f"Rendering done. Time: {_format_duration(timings['Rendering'])}")
    print(f"Saved annotated PDF  {args.output_pdf}")

    # Check PDF file size and compress if needed
    print("Checking PDF file size for compression...")
    t_compress = time.perf_counter()
    compression_performed = compress_pdf_if_needed(
        pdf_path=args.output_pdf,
        target_size_mb=10.0,
        max_quality=75,
        max_dimension=2000,
    )
    timings["PDF Compression"] = time.perf_counter() - t_compress
    if compression_performed:
        print(f"PDF compression completed. Time: {_format_duration(timings['PDF Compression'])}")
    else:
        print(f"PDF compression skipped (not needed). Time: {_format_duration(timings['PDF Compression'])}")

    total_elapsed = time.perf_counter() - total_start
    print("")
    print("=" * 60)
    print("ESSAY GRADING TIMING SUMMARY")
    print("=" * 60)
    for phase, elapsed in timings.items():
        print(f"  {phase}: {_format_duration(elapsed)}")
    print("-" * 60)
    print(f"  Total essay grading time: {_format_duration(total_elapsed)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
