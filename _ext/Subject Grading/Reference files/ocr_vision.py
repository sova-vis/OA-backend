# ocr_vision.py
#
# Google Cloud Vision OCR: run_ocr_on_pdf and helpers.
# Used by grade_pdf_answer. load_environment and the run_ocr_on_pdf call stay in grade_pdf_answer.

from __future__ import annotations

import io
import random
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any, Callable, Dict, List, Optional, Tuple

import fitz
from google.cloud import vision
from PIL import Image

# -----------------------------
# OCR WITH GOOGLE VISION
# -----------------------------


def _noop_append_log(_path, _level, _msg):  # noqa: ARG001
    pass


def _bbox_to_tuples(bbox) -> List[Tuple[int, int]]:
    return [(v.x, v.y) for v in bbox.vertices]


def _paragraph_text(paragraph) -> str:
    words = []
    for word in paragraph.words:
        symbols = "".join(symbol.text for symbol in word.symbols)
        words.append(symbols)
    return " ".join(words).strip()


def _is_noise_text(text: str, bbox: List[Tuple[int, int]], page_w: int, page_h: int) -> bool:
    """
    Filter out background noise from OCR results.
    Returns True if the text is likely noise.
    """
    if not text or not bbox:
        return True

    # Filter very short text (1-2 chars) that's likely noise
    if len(text.strip()) <= 2:
        return True

    # Calculate bbox dimensions
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    if not xs or not ys:
        return True

    width = max(xs) - min(xs)
    height = max(ys) - min(ys)

    # Filter extremely small text (likely artifacts)
    if width < 10 or height < 10:
        return True

    # Filter text at extreme edges (often page numbers or noise)
    center_x = (min(xs) + max(xs)) / 2
    center_y = (min(ys) + max(ys)) / 2

    margin = 30  # pixels
    if center_x < margin or center_x > page_w - margin:
        if center_y < margin or center_y > page_h - margin:
            return True

    return False


def _is_retryable_error(
    error: Exception,
    response: Optional[vision.AnnotateImageResponse] = None,
) -> Tuple[bool, str]:
    """
    Determine if an error should be retried based on error type and message.
    
    Args:
        error: The exception that was raised
        response: Optional Vision API response object (for checking response.error)
    
    Returns:
        Tuple of (is_retryable: bool, error_category: str)
        error_category: One of: 'network_error', 'rate_limit', 'server_error', 
                       'timeout', 'auth_error', 'invalid_input', 'not_found', 'unknown'
    """
    error_type = type(error).__name__
    error_msg = str(error).lower()
    
    # Check Google Vision API response errors first (if available)
    if response and hasattr(response, 'error') and response.error.message:
        api_error_msg = response.error.message.lower()
        
        # Rate limit errors
        if any(keyword in api_error_msg for keyword in [
            'resource_exhausted', 'rate limit', 'quota', '429', 'rate limit'
        ]):
            return True, 'rate_limit'
        
        # Server errors
        if any(keyword in api_error_msg for keyword in [
            'unavailable', 'deadline_exceeded', 'internal error', '500', '502', '503', '504'
        ]):
            return True, 'server_error'
        
        # Non-retryable API errors
        if any(keyword in api_error_msg for keyword in [
            'permission_denied', 'invalid_argument', 'invalid_image', 
            'not_found', '401', '403', '400', '404', '422'
        ]):
            if 'permission_denied' in api_error_msg or '401' in api_error_msg or '403' in api_error_msg:
                return False, 'auth_error'
            elif 'invalid' in api_error_msg or '400' in api_error_msg or '422' in api_error_msg:
                return False, 'invalid_input'
            elif 'not_found' in api_error_msg or '404' in api_error_msg:
                return False, 'not_found'
    
    # Check exception types
    # Network/Connection errors (retryable)
    if error_type in ['ConnectionError', 'ConnectionResetError', 'ConnectionAbortedError']:
        return True, 'network_error'
    
    if 'connection' in error_msg or 'network' in error_msg or 'dns' in error_msg or 'socket' in error_msg:
        return True, 'network_error'
    
    # Timeout errors (retryable, but conditional)
    if error_type == 'TimeoutError' or 'timeout' in error_msg or 'deadline' in error_msg:
        return True, 'timeout'
    
    # Rate limit errors (retryable)
    if '429' in error_msg or 'rate limit' in error_msg or 'quota' in error_msg or 'resource_exhausted' in error_msg:
        return True, 'rate_limit'
    
    # Server errors (retryable)
    if any(code in error_msg for code in ['500', '502', '503', '504']):
        return True, 'server_error'
    if 'unavailable' in error_msg or 'internal error' in error_msg or 'gateway' in error_msg:
        return True, 'server_error'
    
    # Authentication/Permission errors (non-retryable)
    if error_type in ['PermissionError']:
        return False, 'auth_error'
    if any(keyword in error_msg for keyword in [
        'auth', 'permission', 'unauthorized', 'forbidden', '401', '403'
    ]):
        return False, 'auth_error'
    
    # Invalid request errors (non-retryable)
    if error_type == 'ValueError':
        return False, 'invalid_input'
    if any(keyword in error_msg for keyword in [
        'invalid', 'bad request', '400', '422', 'invalid_argument', 'invalid_image'
    ]):
        return False, 'invalid_input'
    
    # Not found errors (non-retryable)
    if '404' in error_msg or 'not found' in error_msg:
        return False, 'not_found'
    
    # Format/corruption errors (non-retryable)
    if any(keyword in error_msg for keyword in [
        'format', 'corrupt', 'unsupported', 'too large', 'size limit'
    ]):
        return False, 'invalid_input'
    
    # Default: For unknown errors, be conservative and don't retry
    # This can be adjusted based on observed error patterns
    return False, 'unknown'


def _calculate_backoff_delay(
    attempt_number: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter_range: float = 0.2,
    is_rate_limit: bool = False,
    rate_limit_base_delay: float = 5.0,
    rate_limit_max_delay: float = 300.0,
    retry_after: Optional[float] = None,
) -> float:
    """
    Calculate exponential backoff delay with jitter.
    
    Formula: delay = base_delay * (2 ^ (attempt_number - 1)) + jitter
    - Exponential growth per attempt
    - Capped at max_delay
    - Jitter added to prevent thundering herd
    
    Args:
        attempt_number: Current attempt number (1-indexed, attempt 1 = no wait)
        base_delay: Base delay in seconds (default: 1.0)
        max_delay: Maximum delay in seconds (default: 60.0)
        jitter_range: Jitter range (0.0-1.0) for randomization (default: 0.2 = ±20%)
        is_rate_limit: If True, use rate limit specific delays
        rate_limit_base_delay: Base delay for rate limits (default: 5.0)
        rate_limit_max_delay: Max delay for rate limits (default: 300.0)
        retry_after: Optional Retry-After header value in seconds (takes precedence)
    
    Returns:
        Delay in seconds (float)
    
    Examples:
        # Standard backoff (attempt 2, base=1.0, max=60.0, jitter=0.2)
        # delay = 1.0 * 2^1 = 2.0s, jitter = ±0.4s → 1.6-2.4s
        
        # Rate limit backoff (attempt 3, base=5.0, max=300.0, jitter=0.2)
        # delay = 5.0 * 2^2 = 20.0s, jitter = ±4.0s → 16.0-24.0s
        
        # With Retry-After header
        # delay = retry_after (e.g., 30.0s) - no jitter applied
    """
    # If Retry-After header is provided, use it directly (no jitter)
    if retry_after is not None and retry_after > 0:
        return float(retry_after)
    
    # Use rate limit parameters if this is a rate limit error
    if is_rate_limit:
        effective_base = rate_limit_base_delay
        effective_max = rate_limit_max_delay
    else:
        effective_base = base_delay
        effective_max = max_delay
    
    # Calculate exponential delay: base * (2 ^ (attempt - 1))
    # Attempt 1 = no wait (2^0 = 1, but we don't wait before first attempt)
    # Attempt 2 = base * 2^1 = base * 2
    # Attempt 3 = base * 2^2 = base * 4
    # etc.
    if attempt_number <= 1:
        # First attempt has no backoff (immediate)
        return 0.0
    
    exponential_delay = effective_base * (2 ** (attempt_number - 1))
    
    # Cap at maximum delay
    capped_delay = min(exponential_delay, effective_max)
    
    # Add jitter: random variation of ±jitter_range percentage
    # Example: jitter_range=0.2 means ±20% variation
    # jitter = random.uniform(-0.2, 0.2) * capped_delay
    jitter = random.uniform(-jitter_range, jitter_range) * capped_delay
    final_delay = capped_delay + jitter
    
    # Ensure delay is non-negative
    return max(0.0, final_delay)


def _check_retry_budget(
    elapsed_time: float,
    overall_timeout: Optional[float],
    backoff_delay: float,
    estimated_attempt_time: float,
    safety_margin: float = 5.0,
) -> bool:
    """
    Check if retry budget allows another retry attempt.
    
    Determines if there's enough time remaining in the overall timeout budget
    to perform another retry attempt (including backoff delay and estimated attempt time).
    
    Args:
        elapsed_time: Time already spent on processing (seconds)
        overall_timeout: Overall timeout budget (None = no limit)
        backoff_delay: Calculated backoff delay before retry (seconds)
        estimated_attempt_time: Estimated time for the retry attempt (seconds)
        safety_margin: Safety margin to avoid cutting it too close (default: 5.0 seconds)
    
    Returns:
        True if retry budget allows another attempt, False otherwise
    
    Examples:
        # Overall timeout = 600s, elapsed = 580s, backoff = 2s, attempt = 10s
        # Remaining = 20s, needed = 12s → True (within budget)
        
        # Overall timeout = 600s, elapsed = 590s, backoff = 5s, attempt = 10s
        # Remaining = 10s, needed = 15s → False (exceeds budget)
    """
    # If no overall timeout, always allow retry
    if overall_timeout is None:
        return True
    
    # Calculate remaining budget
    remaining_budget = overall_timeout - elapsed_time
    
    # Calculate total retry cost (backoff + attempt + safety margin)
    retry_cost = backoff_delay + estimated_attempt_time + safety_margin
    
    # Check if retry would exceed budget
    return retry_cost <= remaining_budget


def _estimate_attempt_time(
    per_page_timeout: float,
    previous_attempt_duration: Optional[float] = None,
) -> float:
    """
    Estimate time for a retry attempt.
    
    Uses conservative estimate based on per-page timeout or previous attempt duration.
    
    Args:
        per_page_timeout: Per-page timeout (used as max estimate)
        previous_attempt_duration: Duration of previous attempt (if available)
    
    Returns:
        Estimated attempt time in seconds
    """
    # If we have previous attempt duration, use it as estimate (with small buffer)
    if previous_attempt_duration is not None:
        # Use previous duration + 10% buffer as estimate
        return previous_attempt_duration * 1.1
    
    # Otherwise, use per-page timeout as conservative estimate
    # This assumes worst case: attempt takes full timeout
    return per_page_timeout


def _call_vision_with_retry(
    vision_client: vision.ImageAnnotatorClient,
    vision_image: vision.Image,
    timeout_seconds: float,
    page_number: int,
    max_retries: int = 3,
    retry_base_delay: float = 1.0,
    retry_max_delay: float = 60.0,
    retry_jitter_range: float = 0.2,
    rate_limit_base_delay: float = 5.0,
    rate_limit_max_delay: float = 300.0,
    overall_timeout: Optional[float] = None,
    overall_start_time: Optional[float] = None,
    log_path: Optional[str] = None,
    request_id: Optional[str] = None,
    retry_stats: Optional[Dict[str, Any]] = None,
    append_log: Optional[Callable[[Optional[str], str, str], None]] = None,
) -> vision.AnnotateImageResponse:
    """
    Call Google Vision API with timeout protection and retry logic.
    
    Wraps `_call_vision_with_timeout()` with retry logic, exponential backoff,
    and budget management. Retries transient failures while failing fast on
    permanent errors.
    
    Args:
        vision_client: Google Vision client
        vision_image: Image to process
        timeout_seconds: Maximum time per attempt (seconds)
        page_number: Page number for error messages
        max_retries: Maximum retry attempts (default: 3)
        retry_base_delay: Base delay for exponential backoff (default: 1.0)
        retry_max_delay: Maximum delay between retries (default: 60.0)
        retry_jitter_range: Jitter range for randomization (default: 0.2)
        rate_limit_base_delay: Base delay for rate limit errors (default: 5.0)
        rate_limit_max_delay: Max delay for rate limit errors (default: 300.0)
        overall_timeout: Overall timeout budget (None = no limit)
        overall_start_time: Start time for overall timeout tracking (None = current time)
        log_path: Optional path to log file
        request_id: Optional request ID for logging
    
    Returns:
        Vision API response
    
    Raises:
        TimeoutError: If timeout occurs and retries exhausted or budget exceeded
        RuntimeError: If API call fails with non-retryable error or all retries exhausted
    """
    if overall_start_time is None:
        overall_start_time = time.perf_counter()
    _log = append_log if append_log is not None else _noop_append_log
    
    last_error: Optional[Exception] = None
    last_response: Optional[vision.AnnotateImageResponse] = None
    previous_attempt_duration: Optional[float] = None
    
    # Initialize retry statistics if not provided
    if retry_stats is None:
        retry_stats = {
            "total_attempts": 0,
            "successful_retries": 0,
            "exhausted_retries": 0,
            "rate_limit_events": 0,
            "non_retryable_errors": 0,
            "budget_exceeded": 0,
            "retry_attempts_by_category": {},
        }
    
    for attempt in range(1, max_retries + 1):
        attempt_start_time = time.perf_counter()
        retry_stats["total_attempts"] += 1
        
        try:
            # Make API call with timeout
            response = _call_vision_with_timeout(
                vision_client=vision_client,
                vision_image=vision_image,
                timeout_seconds=timeout_seconds,
                page_number=page_number,
            )
            
            # Store response for potential error classification
            last_response = response
            
            # Success on first attempt
            if attempt == 1:
                return response
            
            # Success after retry
            attempt_duration = time.perf_counter() - attempt_start_time
            retry_stats["successful_retries"] += 1
            _log(
                log_path,
                "INFO",
                f"request={request_id} ocr_retry_success page={page_number} "
                f"attempt={attempt}/{max_retries} total_attempts={attempt} "
                f"duration_ms={int(attempt_duration * 1000)}",
            )
            # Return response with retry stats (for potential future use)
            return response
            
        except Exception as e:
            last_error = e
            attempt_duration = time.perf_counter() - attempt_start_time
            previous_attempt_duration = attempt_duration
            
            # Note: last_response may not be available if exception occurred before API call
            # Error classification will work with exception type and message patterns
            
            # Classify error
            is_retryable, error_category = _is_retryable_error(e, last_response)
            
            # Non-retryable error: fail fast
            if not is_retryable:
                retry_stats["non_retryable_errors"] += 1
                _log(
                    log_path,
                    "ERROR",
                    f"request={request_id} ocr_non_retryable page={page_number} "
                    f"attempt={attempt} error_category={error_category} error={str(e)}",
                )
                raise RuntimeError(
                    f"OCR failed on page {page_number} (non-retryable {error_category}): {str(e)}"
                ) from e
            
            # Check if we have retries remaining
            if attempt >= max_retries:
                # All retries exhausted
                retry_stats["exhausted_retries"] += 1
                # Track retry attempts by category
                if error_category not in retry_stats["retry_attempts_by_category"]:
                    retry_stats["retry_attempts_by_category"][error_category] = 0
                retry_stats["retry_attempts_by_category"][error_category] += 1
                _log(
                    log_path,
                    "ERROR",
                    f"request={request_id} ocr_retry_exhausted page={page_number} "
                    f"attempts={max_retries} error_category={error_category} error={str(e)}",
                )
                raise RuntimeError(
                    f"OCR failed on page {page_number} after {max_retries} attempts "
                    f"({error_category}): {str(e)}"
                ) from e
            
            # Track retry attempts by category
            if error_category not in retry_stats["retry_attempts_by_category"]:
                retry_stats["retry_attempts_by_category"][error_category] = 0
            retry_stats["retry_attempts_by_category"][error_category] += 1
            
            # Calculate backoff delay
            is_rate_limit = (error_category == 'rate_limit')
            if is_rate_limit:
                retry_stats["rate_limit_events"] += 1
            backoff_delay = _calculate_backoff_delay(
                attempt_number=attempt + 1,  # Next attempt number
                base_delay=retry_base_delay,
                max_delay=retry_max_delay,
                jitter_range=retry_jitter_range,
                is_rate_limit=is_rate_limit,
                rate_limit_base_delay=rate_limit_base_delay,
                rate_limit_max_delay=rate_limit_max_delay,
                retry_after=None,  # TODO: Extract from response headers if available
            )
            
            # Estimate attempt time for budget check
            estimated_attempt_time = _estimate_attempt_time(
                per_page_timeout=timeout_seconds,
                previous_attempt_duration=previous_attempt_duration,
            )
            
            # Check retry budget
            elapsed_time = time.perf_counter() - overall_start_time
            if not _check_retry_budget(
                elapsed_time=elapsed_time,
                overall_timeout=overall_timeout,
                backoff_delay=backoff_delay,
                estimated_attempt_time=estimated_attempt_time,
            ):
                # Budget exceeded, stop retrying
                retry_stats["budget_exceeded"] += 1
                _log(
                    log_path,
                    "WARNING",
                    f"request={request_id} ocr_retry_budget_exceeded page={page_number} "
                    f"attempt={attempt} elapsed_s={elapsed_time:.1f} "
                    f"overall_timeout_s={overall_timeout} backoff_s={backoff_delay:.1f} "
                    f"estimated_s={estimated_attempt_time:.1f}",
                )
                raise TimeoutError(
                    f"OCR retry budget exceeded on page {page_number} at attempt {attempt}: "
                    f"elapsed {elapsed_time:.1f}s, would need {backoff_delay + estimated_attempt_time:.1f}s"
                )
            
            # Log retry attempt
            _log(
                log_path,
                "INFO",
                f"request={request_id} ocr_retry_attempt page={page_number} "
                f"attempt={attempt + 1}/{max_retries} error_category={error_category} "
                f"wait_s={backoff_delay:.2f} previous_duration_ms={int(previous_attempt_duration * 1000)}",
            )
            
            # Special logging for rate limits
            if is_rate_limit:
                _log(
                    log_path,
                    "WARNING",
                    f"request={request_id} ocr_rate_limit page={page_number} "
                    f"attempt={attempt + 1} wait_s={backoff_delay:.2f}",
                )
            
            # Wait before retry
            time.sleep(backoff_delay)
            
            # Continue to next retry attempt
            continue
    
    # Should never reach here, but handle just in case
    if last_error:
        raise RuntimeError(
            f"OCR failed on page {page_number} after {max_retries} attempts: {str(last_error)}"
        ) from last_error
    
    raise RuntimeError(f"OCR failed on page {page_number}: unexpected error")


def _call_vision_with_timeout(
    vision_client: vision.ImageAnnotatorClient,
    vision_image: vision.Image,
    timeout_seconds: float,
    page_number: int,
) -> vision.AnnotateImageResponse:
    """
    Call Google Vision API with timeout protection.
    
    Args:
        vision_client: Google Vision client
        vision_image: Image to process
        timeout_seconds: Maximum time to wait for response
        page_number: Page number for error messages
    
    Returns:
        Vision API response
    
    Raises:
        TimeoutError: If the API call exceeds timeout_seconds
        RuntimeError: If the API call fails with an error
    """
    def _make_api_call():
        """Make the actual API call in a separate thread."""
        return vision_client.document_text_detection(image=vision_image)
    
    # Use ThreadPoolExecutor to enforce timeout
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_make_api_call)
        try:
            response = future.result(timeout=timeout_seconds)
            
            # Check for API errors
            if response.error.message:
                raise RuntimeError(
                    f"OCR failed on page {page_number}: {response.error.message}"
                )
            
            return response
            
        except FutureTimeoutError:
            # Cancel the future if possible
            future.cancel()
            raise TimeoutError(
                f"OCR timeout on page {page_number}: "
                f"exceeded {timeout_seconds} seconds"
            )
        except Exception as e:
            # Re-raise other exceptions with context
            if isinstance(e, (TimeoutError, RuntimeError)):
                raise
            raise RuntimeError(
                f"OCR error on page {page_number}: {str(e)}"
            ) from e


def _optimize_image_for_ocr(
    img: Image.Image,
    max_dimension: int,
    min_dimension_for_optimization: int,
    enabled: bool = True,
) -> Tuple[Image.Image, float, float]:
    """
    Optimize image for OCR by downscaling if it exceeds maximum dimensions.
    Preserves aspect ratio and returns scale factors for bounding box adjustment.
    
    Args:
        img: PIL Image to optimize
        max_dimension: Maximum width or height (downscale if larger)
        min_dimension_for_optimization: Only optimize if dimension exceeds this
        enabled: Whether optimization is enabled
    
    Returns:
        Tuple of (optimized_image, scale_x, scale_y)
        - optimized_image: Optimized PIL Image (or original if not optimized)
        - scale_x: X-axis scale factor (original_width / optimized_width, 1.0 if not scaled)
        - scale_y: Y-axis scale factor (original_height / optimized_height, 1.0 if not scaled)
    """
    if not enabled:
        return img, 1.0, 1.0
    
    original_w, original_h = img.size
    max_dim = max(original_w, original_h)
    
    # Only optimize if image exceeds minimum dimension threshold
    if max_dim <= min_dimension_for_optimization:
        return img, 1.0, 1.0
    
    # Calculate new dimensions preserving aspect ratio
    if original_w > original_h:
        # Landscape: limit width
        if original_w > max_dimension:
            new_w = max_dimension
            new_h = int(original_h * (max_dimension / original_w))
        else:
            return img, 1.0, 1.0
    else:
        # Portrait or square: limit height
        if original_h > max_dimension:
            new_h = max_dimension
            new_w = int(original_w * (max_dimension / original_h))
        else:
            return img, 1.0, 1.0
    
    # Downscale using high-quality resampling
    optimized_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Calculate scale factors for bounding box adjustment
    scale_x = original_w / new_w
    scale_y = original_h / new_h
    
    return optimized_img, scale_x, scale_y


def _process_single_page_ocr(
    vision_client: vision.ImageAnnotatorClient,
    img: Image.Image,
    page_num: int,
    per_page_timeout: float,
    overall_timeout: Optional[float],
    overall_start_time: float,
    max_retries: int,
    retry_base_delay: float,
    retry_max_delay: float,
    retry_jitter_range: float,
    rate_limit_base_delay: float,
    rate_limit_max_delay: float,
    log_path: Optional[str],
    request_id: Optional[str],
    image_optimization_enabled: bool = True,
    image_max_dimension: int = 2048,
    image_min_dimension_for_optimization: int = 1500,
    append_log: Optional[Callable[[Optional[str], str, str], None]] = None,
) -> Tuple[int, Dict[str, Any], Dict[str, Any], Optional[str]]:
    """
    Process a single page OCR. This function is designed to be called in parallel.
    
    Returns:
        Tuple of (page_number, page_output_dict, retry_stats_dict, full_text, error_message)
        - page_output_dict: Page OCR data or error info
        - retry_stats_dict: Retry statistics for this page
        - full_text: Extracted text from page (empty string if error)
        - error_message: None if success, error message if failed
    """
    page_retry_stats = {
        "total_attempts": 0,
        "successful_retries": 0,
        "exhausted_retries": 0,
        "rate_limit_events": 0,
        "non_retryable_errors": 0,
        "budget_exceeded": 0,
        "retry_attempts_by_category": {},
    }
    _log = append_log if append_log is not None else _noop_append_log
    
    try:
        # Check overall timeout before processing
        if overall_timeout is not None:
            elapsed = time.perf_counter() - overall_start_time
            if elapsed >= overall_timeout:
                error_msg = f"OCR overall timeout exceeded at page {page_num}: {elapsed:.1f}s >= {overall_timeout}s"
                _log(
                    log_path,
                    "WARNING",
                    f"request={request_id} ocr_overall_timeout page={page_num} "
                    f"elapsed={elapsed:.1f}s limit={overall_timeout}s",
                )
                return (
                    page_num,
                    {
                        "page_number": page_num,
                        "lines": [],
                        "error": "timeout",
                        "error_message": error_msg,
                    },
                    page_retry_stats,
                    "",  # No text on error
                    error_msg,
                )
        
        # Store original dimensions for bounding box adjustment and noise filtering
        original_page_w, original_page_h = img.size
        
        # Optimize image for OCR (downscale if too large)
        optimized_img, scale_x, scale_y = _optimize_image_for_ocr(
            img=img,
            max_dimension=image_max_dimension,
            min_dimension_for_optimization=image_min_dimension_for_optimization,
            enabled=image_optimization_enabled,
        )
        
        # Log optimization if applied
        if scale_x != 1.0 or scale_y != 1.0:
            _log(
                log_path,
                "INFO",
                f"request={request_id} ocr_image_optimized page={page_num} "
                f"original_size={original_page_w}x{original_page_h} "
                f"optimized_size={optimized_img.size[0]}x{optimized_img.size[1]} "
                f"scale_x={scale_x:.2f} scale_y={scale_y:.2f}",
            )
        
        # Use optimized image for OCR
        buffer = io.BytesIO()
        optimized_img.save(buffer, format="PNG")
        vision_image = vision.Image(content=buffer.getvalue())
        
        # Use original dimensions for noise filtering (bounding boxes will be adjusted)
        page_w, page_h = original_page_w, original_page_h
        
        # Call Vision API with timeout protection and retry logic
        page_start_time = time.perf_counter()
        response = _call_vision_with_retry(
            vision_client=vision_client,
            vision_image=vision_image,
            timeout_seconds=per_page_timeout,
            page_number=page_num,
            max_retries=max_retries,
            retry_base_delay=retry_base_delay,
            retry_max_delay=retry_max_delay,
            retry_jitter_range=retry_jitter_range,
            rate_limit_base_delay=rate_limit_base_delay,
            rate_limit_max_delay=rate_limit_max_delay,
            overall_timeout=overall_timeout,
            overall_start_time=overall_start_time,
            log_path=log_path,
            request_id=request_id,
            retry_stats=page_retry_stats,
            append_log=_log,
        )
        page_duration = time.perf_counter() - page_start_time
        
        _log(
            log_path,
            "INFO",
            f"request={request_id} ocr_page_complete page={page_num} "
            f"duration_ms={int(page_duration * 1000)}",
        )
        
        # Process the response
        page_lines: List[Dict[str, Any]] = []
        annotation = response.full_text_annotation
        full_text = ""
        
        if annotation:
            full_text = annotation.text.strip()
            for page in annotation.pages:
                for block in page.blocks:
                    for paragraph in block.paragraphs:
                        text = _paragraph_text(paragraph)
                        para_bbox = _bbox_to_tuples(paragraph.bounding_box)
                        
                        # Adjust bounding box for image optimization (scale up to original dimensions)
                        if scale_x != 1.0 or scale_y != 1.0:
                            para_bbox = [
                                (int(x * scale_x), int(y * scale_y)) for x, y in para_bbox
                            ]
                        
                        # Filter noise (using original page dimensions)
                        if _is_noise_text(text, para_bbox, page_w, page_h):
                            continue
                        
                        word_entries: List[Dict[str, Any]] = []
                        for word in paragraph.words:
                            w_text = "".join(
                                symbol.text for symbol in word.symbols
                            ).strip()
                            if not w_text:
                                continue
                            word_bbox = _bbox_to_tuples(word.bounding_box)
                            
                            # Adjust bounding box for image optimization (scale up to original dimensions)
                            if scale_x != 1.0 or scale_y != 1.0:
                                word_bbox = [
                                    (int(x * scale_x), int(y * scale_y)) for x, y in word_bbox
                                ]
                            
                            # Filter noise words (using original page dimensions)
                            if _is_noise_text(w_text, word_bbox, page_w, page_h):
                                continue
                            
                            word_entries.append({
                                "text": w_text,
                                "bbox": word_bbox,
                            })
                        
                        if word_entries:  # Only add paragraph if it has valid words
                            page_lines.append({
                                "text": text,
                                "bbox": para_bbox,
                                "words": word_entries,
                            })
        else:
            text_annotations = response.text_annotations
            if text_annotations:
                full_text = text_annotations[0].description.strip()
                for ta in text_annotations[1:]:
                    ta_bbox = _bbox_to_tuples(ta.bounding_poly)
                    
                    # Adjust bounding box for image optimization (scale up to original dimensions)
                    if scale_x != 1.0 or scale_y != 1.0:
                        ta_bbox = [
                            (int(x * scale_x), int(y * scale_y)) for x, y in ta_bbox
                        ]
                    
                    # Filter noise (using original page dimensions)
                    if not _is_noise_text(ta.description, ta_bbox, page_w, page_h):
                        page_lines.append({
                            "text": ta.description,
                            "bbox": ta_bbox,
                            "words": [],
                        })
        
        return (
            page_num,
            {"page_number": page_num, "lines": page_lines},
            page_retry_stats,
            full_text,
            None,  # No error
        )
        
    except TimeoutError as e:
        # Handle per-page timeout
        error_msg = str(e)
        _log(
            log_path,
            "ERROR",
            f"request={request_id} ocr_page_timeout page={page_num} "
            f"timeout={per_page_timeout}s error={error_msg}",
        )
        return (
            page_num,
            {
                "page_number": page_num,
                "lines": [],
                "error": "timeout",
                "error_message": error_msg,
            },
            page_retry_stats,
            "",  # No text on error
            error_msg,
        )
        
    except RuntimeError as e:
        # Handle API errors
        error_msg = str(e)
        _log(
            log_path,
            "ERROR",
            f"request={request_id} ocr_page_error page={page_num} error={error_msg}",
        )
        return (
            page_num,
            {
                "page_number": page_num,
                "lines": [],
                "error": "api_error",
                "error_message": error_msg,
            },
            page_retry_stats,
            "",  # No text on error
            error_msg,
        )
        
    except Exception as e:
        # Handle unexpected errors
        error_msg = str(e)
        _log(
            log_path,
            "ERROR",
            f"request={request_id} ocr_page_unexpected_error page={page_num} "
            f"error={error_msg}",
        )
        return (
            page_num,
            {
                "page_number": page_num,
                "lines": [],
                "error": "unexpected_error",
                "error_message": error_msg,
            },
            page_retry_stats,
            "",  # No text on error
            error_msg,
        )


def run_ocr_on_pdf(
    vision_client: vision.ImageAnnotatorClient,
    pdf_path: str,
    per_page_timeout: float = 120.0,
    overall_timeout: Optional[float] = None,
    log_path: Optional[str] = None,
    request_id: Optional[str] = None,
    max_retries: int = 3,
    progress_tracker: Optional[Any] = None,
    retry_base_delay: float = 1.0,
    retry_max_delay: float = 60.0,
    retry_jitter_range: float = 0.2,
    rate_limit_base_delay: float = 5.0,
    rate_limit_max_delay: float = 300.0,
    concurrent_pages: int = 2,
    batch_size: int = 5,
    batch_failure_threshold: float = 0.5,
    adaptive_concurrency_enabled: bool = True,
    adaptive_min_concurrency: int = 1,
    adaptive_max_concurrency: int = 4,
    adaptive_latency_threshold_ms: float = 90000.0,
    adaptive_stable_batches: int = 2,
    image_optimization_enabled: bool = True,
    image_max_dimension: int = 2048,
    image_min_dimension_for_optimization: int = 1500,
    append_log: Optional[Callable[[Optional[str], str, str], None]] = None,
) -> Dict[str, Any]:
    """
    Run Google Cloud Vision DOCUMENT_TEXT_DETECTION on each page of the PDF.
    Filters out background noise and artifacts.
    
    Args:
        vision_client: Google Vision client
        pdf_path: Path to PDF file
        per_page_timeout: Maximum seconds per page OCR call (default: 120)
        overall_timeout: Maximum seconds for entire OCR process (default: None = no limit)
        log_path: Optional path to log file for timeout logging
        request_id: Optional request ID for logging
        max_retries: Maximum retry attempts per page (default: 3)
        retry_base_delay: Base delay in seconds for exponential backoff (default: 1.0)
        retry_max_delay: Maximum delay in seconds between retries (default: 60.0)
        retry_jitter_range: Jitter range (0.0-1.0) for backoff randomization (default: 0.2)
        rate_limit_base_delay: Base delay in seconds for rate limit backoff (default: 5.0)
        rate_limit_max_delay: Maximum delay in seconds for rate limit backoff (default: 300.0)
        concurrent_pages: Number of pages to process in parallel (default: 2)
        batch_size: Number of pages to process per batch (default: 5)
        batch_failure_threshold: Stop processing if batch failure rate exceeds this (0.0-1.0, default: 0.5)
    
    Returns:
        Dictionary with 'pages' (list of page OCR data) and 'full_text' (concatenated text)
    
    Raises:
        TimeoutError: If overall timeout is exceeded or per-page timeout occurs
        RuntimeError: If OCR processing fails
    """
    _log = append_log if append_log is not None else _noop_append_log
    # Track overall processing time for timeout enforcement
    overall_start_time = time.perf_counter()
    
    doc = fitz.open(pdf_path)
    try:
        images = []
        for page in doc:
            # Check overall timeout before processing each page
            if overall_timeout is not None:
                elapsed = time.perf_counter() - overall_start_time
                if elapsed >= overall_timeout:
                    raise TimeoutError(
                        f"OCR overall timeout exceeded: {elapsed:.1f}s >= {overall_timeout}s"
                    )
            
            pix = page.get_pixmap(dpi=200)
            img_bytes = pix.tobytes("png")
            pil_img = Image.open(io.BytesIO(img_bytes))
            images.append(pil_img)

        pages_output: List[Dict[str, Any]] = []
        full_text_parts: List[str] = []
        failed_pages: List[Dict[str, Any]] = []  # Track failed pages for partial success
        
        # Track aggregate retry statistics across all pages
        aggregate_retry_stats = {
            "total_retry_attempts": 0,
            "successful_retries": 0,
            "exhausted_retries": 0,
            "rate_limit_events": 0,
            "non_retryable_errors": 0,
            "budget_exceeded": 0,
            "retry_attempts_by_category": {},
        }

        # CONDITIONAL PARALLEL OCR: Only parallelize when beneficial
        # Small files (≤5 pages): Sequential processing (no parallel overhead)
        # Medium files (6-15 pages): Low concurrency (2 pages)
        # Large files (16+ pages): Higher concurrency (up to 4 pages)
        total_pages = len(images)
        if total_pages <= 5:
            # Small files: Use sequential processing to avoid parallel overhead
            concurrent_pages = 1
            _log(
                log_path,
                "INFO",
                f"request={request_id} ocr_conditional_parallel pages={total_pages} "
                f"mode=sequential reason=small_file",
            )
        elif total_pages <= 15:
            # Medium files: Use low concurrency (2 pages)
            concurrent_pages = min(2, total_pages)
            _log(
                log_path,
                "INFO",
                f"request={request_id} ocr_conditional_parallel pages={total_pages} "
                f"mode=low_concurrency concurrent_pages={concurrent_pages}",
            )
        else:
            # Large files: Use higher concurrency (up to 4 pages)
            concurrent_pages = min(4, total_pages)
            _log(
                log_path,
                "INFO",
                f"request={request_id} ocr_conditional_parallel pages={total_pages} "
                f"mode=high_concurrency concurrent_pages={concurrent_pages}",
            )
        
        # Ensure concurrent_pages is at least 1 and not more than total pages
        concurrent_pages = max(1, min(concurrent_pages, total_pages))
        
        # Create list to store results (maintains order by page number)
        results: List[Tuple[int, Dict[str, Any], Dict[str, Any], str, Optional[str]]] = []
        
        # WARM-UP PHASE: Process page 1 sequentially first to warm up API connections
        # This prevents cold-start overhead when parallel processing begins
        if len(images) > 0:
            warmup_start_time = time.perf_counter()
            _log(
                log_path,
                "INFO",
                f"request={request_id} ocr_warmup_start page=1",
            )
            
            # Process page 1 sequentially (outside ThreadPoolExecutor)
            page1_result = _process_single_page_ocr(
                vision_client=vision_client,
                img=images[0],
                page_num=1,
                per_page_timeout=per_page_timeout,
                overall_timeout=overall_timeout,
                overall_start_time=overall_start_time,
                max_retries=max_retries,
                retry_base_delay=retry_base_delay,
                retry_max_delay=retry_max_delay,
                retry_jitter_range=retry_jitter_range,
                rate_limit_base_delay=rate_limit_base_delay,
                rate_limit_max_delay=rate_limit_max_delay,
                log_path=log_path,
                request_id=request_id,
                image_optimization_enabled=image_optimization_enabled,
                image_max_dimension=image_max_dimension,
                image_min_dimension_for_optimization=image_min_dimension_for_optimization,
                append_log=_log,
            )
            results.append(page1_result)
            
            # Update progress after page 1
            if progress_tracker:
                total_pages = len(images)
                progress_tracker.update_progress(
                    request_id=request_id,
                    step="OCR Processing",
                    step_number=2,
                    total_steps=11,  # Will be updated dynamically
                    progress_percent=15.0 + (1 / total_pages) * 30.0,  # 15-45% for OCR
                    message=f"Processing page 1 of {total_pages}...",
                    details={"pages_completed": 1, "total_pages": total_pages},
                )
            
            warmup_duration = time.perf_counter() - warmup_start_time
            _log(
                log_path,
                "INFO",
                f"request={request_id} ocr_warmup_complete page=1 duration_ms={int(warmup_duration * 1000)}",
            )
        
        # BATCH ORCHESTRATION PHASE: Process remaining pages (2+) in batches
        remaining_pages = len(images) - 1  # Exclude page 1 which was already processed
        
        if remaining_pages > 0:
            # Adjust concurrent_pages for remaining pages
            effective_concurrency = max(1, min(concurrent_pages, remaining_pages))
            
            # Initialize adaptive concurrency tracking
            if adaptive_concurrency_enabled:
                # Ensure concurrency is within adaptive bounds
                effective_concurrency = max(adaptive_min_concurrency, min(effective_concurrency, adaptive_max_concurrency))
                stable_batch_count = 0  # Count consecutive stable batches
                previous_concurrency = effective_concurrency
            
            # Calculate number of batches
            num_batches = (remaining_pages + batch_size - 1) // batch_size  # Ceiling division
            
            _log(
                log_path,
                "INFO",
                f"request={request_id} ocr_batch_orchestration_start total_pages={len(images)} "
                f"remaining_pages={remaining_pages} batch_size={batch_size} num_batches={num_batches} "
                f"concurrent_pages={effective_concurrency} "
                f"adaptive_enabled={adaptive_concurrency_enabled}",
            )
            
            # Process pages in batches
            remaining_indices = list(range(1, len(images)))  # Indices for pages 2+
            batch_num = 0
            should_continue = True
            
            while remaining_indices and should_continue:
                batch_num += 1
                
                # Check overall timeout before starting batch
                if overall_timeout is not None:
                    elapsed = time.perf_counter() - overall_start_time
                    if elapsed >= overall_timeout:
                        _log(
                            log_path,
                            "WARNING",
                            f"request={request_id} ocr_batch_timeout_check batch={batch_num} "
                            f"elapsed={elapsed:.1f}s limit={overall_timeout}s stopping",
                        )
                        break
                
                # Get next batch of pages
                batch_indices = remaining_indices[:batch_size]
                batch_pages = [idx + 1 for idx in batch_indices]  # Convert to page numbers (1-indexed)
                batch_start_time = time.perf_counter()
                
                _log(
                    log_path,
                    "INFO",
                    f"request={request_id} ocr_batch_start batch={batch_num}/{num_batches} "
                    f"pages={batch_pages} size={len(batch_pages)}",
                )
                
                # Process batch in parallel
                batch_results: List[Tuple[int, Dict[str, Any], Dict[str, Any], str, Optional[str]]] = []
                batch_futures = {}
                
                with ThreadPoolExecutor(max_workers=min(effective_concurrency, len(batch_indices))) as executor:
                    # Submit batch tasks
                    for idx in batch_indices:
                        page_num = idx + 1
                        future = executor.submit(
                            _process_single_page_ocr,
                            vision_client,
                            images[idx],
                            page_num,
                            per_page_timeout,
                            overall_timeout,
                            overall_start_time,
                            max_retries,
                            retry_base_delay,
                            retry_max_delay,
                            retry_jitter_range,
                            rate_limit_base_delay,
                            rate_limit_max_delay,
                            log_path,
                            request_id,
                            image_optimization_enabled,
                            image_max_dimension,
                            image_min_dimension_for_optimization,
                            _log,
                        )
                        batch_futures[future] = page_num
                    
                    # Collect batch results
                    for future in batch_futures:
                        try:
                            result = future.result()
                            batch_results.append(result)
                        except Exception as e:
                            # Handle unexpected executor errors
                            page_num = batch_futures[future]
                            error_msg = f"Executor error on page {page_num}: {str(e)}"
                            _log(
                                log_path,
                                "ERROR",
                                f"request={request_id} ocr_executor_error page={page_num} error={error_msg}",
                            )
                            batch_results.append((
                                page_num,
                                {
                                    "page_number": page_num,
                                    "lines": [],
                                    "error": "executor_error",
                                    "error_message": error_msg,
                                },
                                {
                                    "total_attempts": 0,
                                    "successful_retries": 0,
                                    "exhausted_retries": 0,
                                    "rate_limit_events": 0,
                                    "non_retryable_errors": 0,
                                    "budget_exceeded": 0,
                                    "retry_attempts_by_category": {},
                                },
                                "",  # No text on error
                                error_msg,
                            ))
                
                # Analyze batch results
                batch_duration = time.perf_counter() - batch_start_time
                batch_success = sum(1 for r in batch_results if r[4] is None)  # Count successes (error_msg is None)
                batch_failures = len(batch_results) - batch_success
                batch_failure_rate = batch_failures / len(batch_results) if batch_results else 0.0
                
                # Calculate batch metrics for adaptive concurrency
                batch_rate_limit_events = sum(r[2].get("rate_limit_events", 0) for r in batch_results)
                batch_total_attempts = sum(r[2].get("total_attempts", 0) for r in batch_results)
                
                # Calculate average latency per page in batch (for successful pages)
                successful_results = [r for r in batch_results if r[4] is None]
                if successful_results:
                    # Estimate average latency from retry stats (rough approximation)
                    # In practice, we'd track actual page durations, but for now use batch duration / pages
                    avg_latency_ms = (batch_duration / len(successful_results)) * 1000 if successful_results else 0
                else:
                    avg_latency_ms = 0
                
                # Add batch results to overall results
                results.extend(batch_results)
                
                # Update progress after batch completes
                if progress_tracker:
                    total_pages = len(images)
                    pages_completed = len(results)  # Total pages completed so far
                    progress_percent = 15.0 + (pages_completed / total_pages) * 30.0  # 15-45% for OCR
                    progress_tracker.update_progress(
                        request_id=request_id,
                        step="OCR Processing",
                        step_number=2,
                        total_steps=11,
                        progress_percent=progress_percent,
                        message=f"Processing page {pages_completed} of {total_pages}...",
                        details={"pages_completed": pages_completed, "total_pages": total_pages},
                    )
                
                # Remove processed indices
                remaining_indices = remaining_indices[len(batch_indices):]
                
                _log(
                    log_path,
                    "INFO",
                    f"request={request_id} ocr_batch_complete batch={batch_num}/{num_batches} "
                    f"pages={batch_pages} duration_ms={int(batch_duration * 1000)} "
                    f"success={batch_success} failures={batch_failures} "
                    f"failure_rate={batch_failure_rate:.2%} "
                    f"rate_limits={batch_rate_limit_events} avg_latency_ms={int(avg_latency_ms)}",
                )
                
                # ADAPTIVE CONCURRENCY: Adjust concurrency based on batch performance
                if adaptive_concurrency_enabled and batch_num > 1:  # Start adapting after first batch
                    concurrency_changed = False
                    new_concurrency = effective_concurrency
                    
                    # Reduce concurrency if rate limits occurred
                    if batch_rate_limit_events > 0:
                        new_concurrency = max(adaptive_min_concurrency, effective_concurrency - 1)
                        if new_concurrency < effective_concurrency:
                            concurrency_changed = True
                            stable_batch_count = 0  # Reset stable count on reduction
                            _log(
                                log_path,
                                "WARNING",
                                f"request={request_id} ocr_adaptive_concurrency_reduce batch={batch_num} "
                                f"rate_limits={batch_rate_limit_events} "
                                f"concurrency={effective_concurrency}->{new_concurrency} reason=rate_limits",
                            )
                    
                    # Reduce concurrency if average latency exceeds threshold
                    elif avg_latency_ms > adaptive_latency_threshold_ms and effective_concurrency > adaptive_min_concurrency:
                        new_concurrency = max(adaptive_min_concurrency, effective_concurrency - 1)
                        if new_concurrency < effective_concurrency:
                            concurrency_changed = True
                            stable_batch_count = 0  # Reset stable count on reduction
                            _log(
                                log_path,
                                "WARNING",
                                f"request={request_id} ocr_adaptive_concurrency_reduce batch={batch_num} "
                                f"avg_latency_ms={int(avg_latency_ms)} threshold={int(adaptive_latency_threshold_ms)} "
                                f"concurrency={effective_concurrency}->{new_concurrency} reason=high_latency",
                            )
                    
                    # Increase concurrency if stable (no rate limits, low latency)
                    elif (batch_rate_limit_events == 0 and 
                          avg_latency_ms < adaptive_latency_threshold_ms and 
                          effective_concurrency < adaptive_max_concurrency):
                        stable_batch_count += 1
                        if stable_batch_count >= adaptive_stable_batches:
                            new_concurrency = min(adaptive_max_concurrency, effective_concurrency + 1)
                            if new_concurrency > effective_concurrency:
                                concurrency_changed = True
                                stable_count_before_reset = stable_batch_count
                                stable_batch_count = 0  # Reset after increase
                                _log(
                                    log_path,
                                    "INFO",
                                    f"request={request_id} ocr_adaptive_concurrency_increase batch={batch_num} "
                                    f"stable_batches={stable_count_before_reset} "
                                    f"concurrency={effective_concurrency}->{new_concurrency} reason=stable_performance",
                                )
                    else:
                        # Stable but not ready to increase yet
                        stable_batch_count += 1
                    
                    # Update concurrency if changed
                    if concurrency_changed:
                        previous_concurrency = effective_concurrency
                        effective_concurrency = new_concurrency
                        # Ensure concurrency doesn't exceed remaining pages
                        effective_concurrency = min(effective_concurrency, len(remaining_indices))
                
                # Check if we should continue based on failure rate
                if batch_failure_rate > batch_failure_threshold:
                    _log(
                        log_path,
                        "WARNING",
                        f"request={request_id} ocr_batch_high_failure_rate batch={batch_num} "
                        f"failure_rate={batch_failure_rate:.2%} threshold={batch_failure_threshold:.2%} "
                        f"stopping_processing",
                    )
                    should_continue = False
                    break
                
                # Check overall timeout after batch
                if overall_timeout is not None:
                    elapsed = time.perf_counter() - overall_start_time
                    remaining_time = overall_timeout - elapsed
                    if remaining_time < (per_page_timeout * 2):  # Not enough time for another batch
                        _log(
                            log_path,
                            "INFO",
                            f"request={request_id} ocr_batch_timeout_check batch={batch_num} "
                            f"elapsed={elapsed:.1f}s remaining={remaining_time:.1f}s "
                            f"insufficient_time_for_next_batch stopping",
                        )
                        break
            
            _log(
                log_path,
                "INFO",
                f"request={request_id} ocr_batch_orchestration_complete batches_processed={batch_num} "
                f"remaining_pages={len(remaining_indices)}",
            )
        elif len(images) == 1:
            # Only one page, already processed in warm-up
            _log(
                log_path,
                "INFO",
                f"request={request_id} ocr_single_page_complete page=1",
            )
        
        # Sort results by page number to maintain order
        results.sort(key=lambda x: x[0])
        
        # Process results and aggregate statistics
        for page_num, page_output, page_retry_stats, page_full_text, error_msg in results:
            # Aggregate retry statistics
            aggregate_retry_stats["total_retry_attempts"] += page_retry_stats["total_attempts"]
            aggregate_retry_stats["successful_retries"] += page_retry_stats["successful_retries"]
            aggregate_retry_stats["exhausted_retries"] += page_retry_stats["exhausted_retries"]
            aggregate_retry_stats["rate_limit_events"] += page_retry_stats["rate_limit_events"]
            aggregate_retry_stats["non_retryable_errors"] += page_retry_stats["non_retryable_errors"]
            aggregate_retry_stats["budget_exceeded"] += page_retry_stats["budget_exceeded"]
            # Merge category stats
            for category, count in page_retry_stats["retry_attempts_by_category"].items():
                if category not in aggregate_retry_stats["retry_attempts_by_category"]:
                    aggregate_retry_stats["retry_attempts_by_category"][category] = 0
                aggregate_retry_stats["retry_attempts_by_category"][category] += count
            
            # Add page output
            pages_output.append(page_output)
            
            # Add full text if available
            if page_full_text:
                full_text_parts.append(page_full_text)
            
            # Track failed pages
            if error_msg:
                failed_pages.append({
                    "page_number": page_num,
                    "error": page_output.get("error", "unknown_error"),
                    "error_message": error_msg,
                })

        # Log summary
        total_duration = time.perf_counter() - overall_start_time
        success_count = len(pages_output) - len(failed_pages)
        total_pages = len(images)
        
        # Calculate retry success rate
        retry_success_rate = 0.0
        if aggregate_retry_stats["total_retry_attempts"] > 0:
            retry_success_rate = (
                aggregate_retry_stats["successful_retries"] / 
                aggregate_retry_stats["total_retry_attempts"]
            ) * 100.0
        
        _log(
            log_path,
            "INFO",
            f"request={request_id} ocr_complete total_pages={total_pages} "
            f"success={success_count} failed={len(failed_pages)} "
            f"duration_ms={int(total_duration * 1000)}",
        )
        
        # Log retry statistics summary
        if aggregate_retry_stats["total_retry_attempts"] > 0:
            category_summary = ", ".join([
                f"{cat}={count}" 
                for cat, count in aggregate_retry_stats["retry_attempts_by_category"].items()
            ])
            _log(
                log_path,
                "INFO",
                f"request={request_id} ocr_retry_stats "
                f"total_attempts={aggregate_retry_stats['total_retry_attempts']} "
                f"successful_retries={aggregate_retry_stats['successful_retries']} "
                f"retry_success_rate_pct={retry_success_rate:.1f} "
                f"exhausted={aggregate_retry_stats['exhausted_retries']} "
                f"rate_limits={aggregate_retry_stats['rate_limit_events']} "
                f"non_retryable={aggregate_retry_stats['non_retryable_errors']} "
                f"budget_exceeded={aggregate_retry_stats['budget_exceeded']} "
                f"categories=[{category_summary}]",
            )
        
        # If all pages failed, raise an error
        if len(failed_pages) == total_pages:
            raise RuntimeError(
                f"OCR failed on all {total_pages} pages. "
                f"Errors: {[p['error'] for p in failed_pages]}"
            )
        
        # If some pages failed, log warning but return partial results
        if failed_pages:
            failed_page_nums = [p["page_number"] for p in failed_pages]
            _log(
                log_path,
                "WARNING",
                f"request={request_id} ocr_partial_success "
                f"failed_pages={failed_page_nums}",
            )

        return {
            "pages": pages_output,
            "full_text": "\n".join(full_text_parts).strip(),
            "metadata": {
                "total_pages": total_pages,
                "successful_pages": success_count,
                "failed_pages": len(failed_pages),
                "failed_page_numbers": [p["page_number"] for p in failed_pages],
                "processing_duration_seconds": total_duration,
                "retry_statistics": {
                    "total_retry_attempts": aggregate_retry_stats["total_retry_attempts"],
                    "successful_retries": aggregate_retry_stats["successful_retries"],
                    "retry_success_rate_percent": round(retry_success_rate, 2),
                    "exhausted_retries": aggregate_retry_stats["exhausted_retries"],
                    "rate_limit_events": aggregate_retry_stats["rate_limit_events"],
                    "non_retryable_errors": aggregate_retry_stats["non_retryable_errors"],
                    "budget_exceeded": aggregate_retry_stats["budget_exceeded"],
                    "retry_attempts_by_category": aggregate_retry_stats["retry_attempts_by_category"],
                },
            }
        }
    finally:
        doc.close()  # Always close the document to release file handle
