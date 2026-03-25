#!/usr/bin/env python3
"""
Grok-powered OCR evaluation service.
"""

from __future__ import annotations

import logging
import tempfile
import os
import json
import datetime
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

# Import the restored function
from .grade_pdf_answer import grade_pdf_answer
from .progress_tracker import OCRProgressTracker
from .job_manager import OCRJobManager, OCRJob, JobStatus
from backend.utils.rubric_loader import list_available_subjects as get_subjects_for_dropdown

logger = logging.getLogger(__name__)


def _append_log(log_path: str, level: str, message: str) -> None:
    try:
        timestamp = datetime.datetime.utcnow().isoformat() + "Z"
        line = f"{timestamp} [{level}] {message}\n"
        
        # Determine log directory
        log_dir = os.path.dirname(log_path)
        
        # Check if this is an OCR-related log message
        # OCR logs include: upload, steps, timing reports, completion, OCR events
        is_ocr_log = (
            "upload_start" in message or
            "start pdf=" in message or
            " step=" in message or  # Note: space before step= to avoid false matches
            "TIMING_REPORT" in message or
            ("completed" in message and "request=" in message) or
            "report_generated" in message or
            "ocr_" in message or
            "Step " in message  # Timing report steps (e.g., "Step 1: Convert PDF")
        )
        
        # Write to main log file
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line)
        
        # Write OCR logs to separate OCR log file
        if is_ocr_log:
            ocr_log_path = os.path.join(log_dir, "ocr_log.txt")
            with open(ocr_log_path, "a", encoding="utf-8") as f:
                f.write(line)
        
        # Write errors to separate error log file
        if level == "ERROR":
            error_log_path = os.path.join(log_dir, "errors_log.txt")
            with open(error_log_path, "a", encoding="utf-8") as f:
                f.write(line)
    except Exception:
        # Never fail the pipeline due to logging issues.
        pass


@dataclass
class OCRAnnotator:
    """
    Thin wrapper that invokes the Grok pipeline and exposes the old interface.
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
        logger.info(
            "Running Grok evaluation for '%s' (%s bytes) subject=%s",
            original_filename,
            len(pdf_bytes),
            subject,
        )
        request_id = request_id or uuid.uuid4().hex[:8]
        upload_start_ts = time.perf_counter()
        # Ensure logs directory exists
        logs_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "logs")
        )
        os.makedirs(logs_dir, exist_ok=True)
        log_path = os.path.join(logs_dir, "log.txt")
        _append_log(
            log_path,
            "INFO",
            f"request={request_id} upload_start filename={original_filename} bytes={len(pdf_bytes)} subject={subject}",
        )
        
        # Initialize progress tracker
        progress_tracker = OCRProgressTracker(logs_dir=logs_dir)
        progress_tracker.update_progress(
            request_id=request_id,
            step="Starting",
            step_number=0,
            total_steps=11,
            progress_percent=0.0,
            message="Starting evaluation...",
        )

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
            # Lazy import to avoid loading heavy OCR/vision libs unless needed
            from .grade_pdf_answer import grade_pdf_answer  # type: ignore
            # Call the restored grading function
            # grade_pdf_answer(pdf_path, subject, output_json_path, output_pdf_path)
            grade_pdf_answer(
                pdf_path=input_pdf_path,
                subject=subject,
                output_json_path=output_json_path,
                output_pdf_path=output_pdf_path,
                user_id=user_id,
                log_path=log_path,
                request_id=request_id,
                progress_tracker=progress_tracker,
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

            _append_log(
                log_path,
                "INFO",
                f"request={request_id} report_generated filename={original_filename} total_duration_ms={int((time.perf_counter() - upload_start_ts) * 1000)}",
            )
            
            # Clear progress after completion (with small delay to allow final poll)
            import threading
            def clear_progress_delayed():
                time.sleep(5)  # Keep progress available for 5 seconds after completion
                progress_tracker.clear_progress(request_id)
            threading.Thread(target=clear_progress_delayed, daemon=True).start()
            
            return annotated_pdf_bytes, metadata, request_id

        except Exception as exc:
            _append_log(
                log_path,
                "ERROR",
                f"request={request_id} annotate_failed error={exc}",
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
                        _append_log(
                            log_path,
                            "WARNING",
                            f"request={request_id} temp_file_cleanup_failed path={path} error={e}",
                        )


def get_all_available_subjects() -> List[Dict[str, str]]:
    return get_subjects_for_dropdown()


def _get_logs_dir() -> str:
    """
    Get the logs directory path consistently.
    This ensures both routes/ocr.py and service.py use the same path.
    """
    # Calculate from this file: backend/ocr/service.py
    # Go up 2 levels: ocr -> backend -> project root
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_file_dir, "..", ".."))
    logs_dir = os.path.join(project_root, "logs")
    return os.path.abspath(logs_dir)


def process_ocr_job(job: OCRJob, job_manager: OCRJobManager) -> None:
    """
    Process an OCR job in the background.
    
    Args:
        job: OCR job to process
        job_manager: Job manager instance (for cancellation checks)
    """
    import tempfile
    
    # Check if cancelled before starting
    if job_manager.is_job_cancelled(job.job_id):
        return
    
    # Ensure logs directory exists
    # Use helper function to ensure consistent path calculation
    logs_dir = _get_logs_dir()
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, "log.txt")
    
    # Log the paths being used for debugging
    _append_log(
        log_path,
        "INFO",
        f"request={job.request_id} job={job.job_id} service_logs_dir={logs_dir}",
    )
    
    # Initialize progress tracker
    progress_tracker = OCRProgressTracker(logs_dir=logs_dir)
    progress_tracker.update_progress(
        request_id=job.request_id,
        step="Starting",
        step_number=0,
        total_steps=11,
        progress_percent=0.0,
        message="Starting evaluation...",
    )
    
    # Create temporary files for input PDF, output JSON, and output PDF
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as input_pdf, \
         tempfile.NamedTemporaryFile(suffix=".json", delete=False) as output_json, \
         tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as output_pdf:
        
        input_pdf_path = input_pdf.name
        output_json_path = output_json.name
        output_pdf_path = output_pdf.name
        
        # Read PDF bytes from result directory (stored during job submission)
        input_pdf_path_stored = os.path.join(logs_dir, "results", f"input_{job.job_id}.pdf")
        
        # Log the expected path for debugging
        _append_log(
            log_path,
            "INFO",
            f"request={job.request_id} job={job.job_id} looking_for_input_pdf path={input_pdf_path_stored}",
        )
        
        # Check if results directory exists
        results_dir = os.path.join(logs_dir, "results")
        if not os.path.exists(results_dir):
            _append_log(
                log_path,
                "ERROR",
                f"request={job.request_id} job={job.job_id} results_dir_not_found path={results_dir}",
            )
            raise FileNotFoundError(
                f"Results directory not found: {results_dir}. "
                f"Expected input PDF at: {input_pdf_path_stored}"
            )
        
        # List files in results directory for debugging
        try:
            files_in_results = os.listdir(results_dir)
            _append_log(
                log_path,
                "INFO",
                f"request={job.request_id} job={job.job_id} files_in_results_dir count={len(files_in_results)} files={files_in_results[:10]}",
            )
        except Exception as e:
            _append_log(
                log_path,
                "WARNING",
                f"request={job.request_id} job={job.job_id} failed_to_list_results_dir error={str(e)}",
            )
        
        # Wait a bit and retry if file doesn't exist (handles race condition)
        max_retries = 5
        retry_delay = 0.2  # 200ms
        for attempt in range(max_retries):
            if os.path.exists(input_pdf_path_stored):
                file_size = os.path.getsize(input_pdf_path_stored)
                _append_log(
                    log_path,
                    "INFO",
                    f"request={job.request_id} job={job.job_id} input_pdf_found attempt={attempt+1} size={file_size}",
                )
                break
            if attempt < max_retries - 1:
                _append_log(
                    log_path,
                    "WARNING",
                    f"request={job.request_id} job={job.job_id} input_pdf_not_found attempt={attempt+1}/{max_retries} retrying_in={retry_delay}s",
                )
                time.sleep(retry_delay)
            else:
                # Final attempt failed - log detailed error
                _append_log(
                    log_path,
                    "ERROR",
                    f"request={job.request_id} job={job.job_id} input_pdf_not_found_after_retries path={input_pdf_path_stored} results_dir={results_dir}",
                )
                raise FileNotFoundError(
                    f"Input PDF not found for job {job.job_id} after {max_retries} attempts. "
                    f"Expected path: {input_pdf_path_stored}. "
                    f"Results directory exists: {os.path.exists(results_dir)}. "
                    f"Please check if the file was written correctly during job submission."
                )
        
        # Copy input PDF to temp location
        import shutil
        shutil.copy2(input_pdf_path_stored, input_pdf_path)
    
    try:
        # Check cancellation before processing
        if job_manager.is_job_cancelled(job.job_id):
            return
        
        _append_log(
            log_path,
            "INFO",
            f"request={job.request_id} job={job.job_id} job_start filename={job.filename} subject={job.subject}",
        )
        
        # Call the grading function
        grade_pdf_answer(
            pdf_path=input_pdf_path,
            subject=job.subject,
            output_json_path=output_json_path,
            output_pdf_path=output_pdf_path,
            user_id=job.user_id,
            log_path=log_path,
            request_id=job.request_id,
            progress_tracker=progress_tracker,
        )
        
        # Check cancellation after processing
        if job_manager.is_job_cancelled(job.job_id):
            return
        
        # Move results to result directory
        # Use the same logs_dir from above
        results_dir = os.path.join(logs_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        
        result_pdf_path = os.path.join(results_dir, f"result_{job.job_id}.pdf")
        result_json_path = os.path.join(results_dir, f"result_{job.job_id}.json")
        
        if os.path.exists(output_pdf_path):
            import shutil
            shutil.move(output_pdf_path, result_pdf_path)
            job.result_pdf_path = result_pdf_path
        else:
            raise RuntimeError("Output PDF was not generated.")
        
        if os.path.exists(output_json_path):
            import shutil
            shutil.move(output_json_path, result_json_path)
            job.result_json_path = result_json_path
        
        _append_log(
            log_path,
            "INFO",
            f"request={job.request_id} job={job.job_id} job_complete filename={job.filename}",
        )
        
        # Update job with result paths
        job_manager._save_job(job)
        
        # Clear progress after completion (with delay)
        import threading
        def clear_progress_delayed():
            time.sleep(5)
            progress_tracker.clear_progress(job.request_id)
        threading.Thread(target=clear_progress_delayed, daemon=True).start()
        
    except Exception as exc:
        _append_log(
            log_path,
            "ERROR",
            f"request={job.request_id} job={job.job_id} job_failed error={exc}",
        )
        raise
    finally:
        # Cleanup temp files
        for path in [input_pdf_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass
