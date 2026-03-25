#!/usr/bin/env python3
"""
Service wrapper for regular (non-backend) OCR files.
This is a temporary service to test the regular files.
"""

from __future__ import annotations

import logging
import tempfile
import os
import json
import time
import uuid
from typing import Any, Dict, List, Tuple, Optional

# Import regular files directly (not backend versions)
import sys
from pathlib import Path
import importlib.util

# Get project root
project_root = Path(__file__).parent.parent.parent
backend_root = project_root / "backend"

# Temporarily remove backend.ocr from sys.modules to force regular files
# This ensures grade_pdf_answer.py uses the regular annotate_pdf_with_rubric.py
original_backend_annotate = sys.modules.pop("backend.ocr.annotate_pdf_with_rubric", None)

try:
    # Add project root to path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Import annotate_pdf_with_rubric from project root if present, otherwise fall back
    # to backend/ocr version (root-level file was removed).
    regular_annotate_path = project_root / "annotate_pdf_with_rubric.py"
    backend_annotate_path = backend_root / "ocr" / "annotate_pdf_with_rubric.py"
    annotate_path = regular_annotate_path if regular_annotate_path.exists() else backend_annotate_path
    spec_annotate = importlib.util.spec_from_file_location("annotate_pdf_with_rubric", annotate_path)
    annotate_module = importlib.util.module_from_spec(spec_annotate)
    sys.modules["annotate_pdf_with_rubric"] = annotate_module
    spec_annotate.loader.exec_module(annotate_module)
    annotate_pdf_answer_pages = annotate_module.annotate_pdf_answer_pages
    
    # Now import grade_pdf_answer (prefer root if present, else backend/ocr version).
    regular_grade_path = project_root / "grade_pdf_answer.py"
    backend_grade_path = backend_root / "ocr" / "grade_pdf_answer.py"
    grade_path = regular_grade_path if regular_grade_path.exists() else backend_grade_path
    spec_grade = importlib.util.spec_from_file_location("grade_pdf_answer", grade_path)
    grade_module = importlib.util.module_from_spec(spec_grade)
    sys.modules["grade_pdf_answer"] = grade_module
    spec_grade.loader.exec_module(grade_module)
    grade_pdf_answer = grade_module.grade_pdf_answer
    
finally:
    # Restore backend module if it existed (for other code that might need it)
    if original_backend_annotate:
        sys.modules["backend.ocr.annotate_pdf_with_rubric"] = original_backend_annotate

logger = logging.getLogger(__name__)


class OCRAnnotatorRegular:
    """
    Service wrapper that uses the regular (non-backend) OCR files.
    This is for temporary testing purposes.
    """

    def annotate_pdf(
        self,
        *,
        pdf_bytes: bytes,
        original_filename: str,
        subject: str,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> Tuple[bytes, Dict[str, Any], str]:
        """
        Annotate PDF using regular files.
        
        Args:
            pdf_bytes: PDF file bytes
            original_filename: Original filename
            subject: Subject name
            user_id: Optional user ID
            request_id: Optional request ID
            
        Returns:
            Tuple of (annotated_pdf_bytes, metadata, request_id)
        """
        logger.info(
            "Running regular OCR evaluation for '%s' (%s bytes) subject=%s",
            original_filename,
            len(pdf_bytes),
            subject,
        )
        request_id = request_id or uuid.uuid4().hex[:8]
        
        # Create temporary files for input PDF, output JSON, and output PDF
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as input_pdf, \
             tempfile.NamedTemporaryFile(suffix=".json", delete=False) as output_json, \
             tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as output_pdf:
            
            input_pdf_path = input_pdf.name
            output_json_path = output_json.name
            output_pdf_path = output_pdf.name
            
            # Write input bytes
            input_pdf.write(pdf_bytes)
            input_pdf.flush()

        try:
            # Call the regular grading function (simpler signature)
            grade_pdf_answer(
                pdf_path=input_pdf_path,
                subject=subject,
                output_json_path=output_json_path,
                output_pdf_path=output_pdf_path,
                user_id=user_id,
            )

            # Read back the results
            if os.path.exists(output_pdf_path):
                with open(output_pdf_path, "rb") as f:
                    annotated_pdf_bytes = f.read()
            else:
                raise RuntimeError("Output PDF was not generated.")

            if os.path.exists(output_json_path):
                with open(output_json_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            else:
                metadata = {}
            
            logger.info(
                "Regular OCR evaluation completed for '%s' request=%s",
                original_filename,
                request_id,
            )
            
            return annotated_pdf_bytes, metadata, request_id

        except Exception as exc:
            logger.error(
                "Regular OCR evaluation failed for '%s' request=%s: %s",
                original_filename,
                request_id,
                exc,
            )
            raise

        finally:
            # Cleanup temp files
            for path in [input_pdf_path, output_json_path, output_pdf_path]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception as e:
                        logger.warning(f"Failed to remove temp file {path}: {e}")


def get_all_available_subjects() -> List[Dict[str, str]]:
    """
    Get all available subjects from rubric directory.
    Uses the same logic as regular files.
    """
    try:
        from backend.utils.rubric_loader import list_available_subjects
        return list_available_subjects()
    except ImportError:
        # Fallback: try to find subjects from 20marks_Rubrics directory
        project_root = Path(__file__).parent.parent.parent
        rubrics_dir = project_root / "20marks_Rubrics"
        if not rubrics_dir.exists():
            # Try backend location
            rubrics_dir = project_root / "backend" / "20marks_Rubrics"
        
        subjects = []
        if rubrics_dir.exists():
            for entry in sorted(rubrics_dir.iterdir()):
                if entry.is_dir():
                    subjects.append({
                        "id": entry.name.lower().replace(" ", "-"),
                        "name": entry.name,
                    })
        return subjects

