"""
Quick validation script for annotation matching system
Verifies that all necessary functions exist and have correct signatures
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

try:
    from annotate_pdf_with_essay_rubric import (
        _normalize_compact,
        _build_strict_annotation_candidates,
        _find_exact_rect_in_pdf_text,
        _find_exact_rect_from_ocr,
        _clip_rect,
        _draw_pointer_line,
        annotate_pdf_essay_pages
    )
    print("✅ All required functions imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Test _normalize_compact
test_text = "Hello World, this is a TEST!"
normalized = _normalize_compact(test_text)
expected = "hello world this is a test"  # Stopwords are kept for exact matching
assert normalized == expected, f"Expected '{expected}', got '{normalized}'"
print(f"✅ _normalize_compact: '{test_text}' → '{normalized}'")

# Test _build_strict_annotation_candidates
test_ann1 = {
    "section_id": "2) 1947-1956: A State Without a stable Political Compass",
    "page": 1
}
candidates1 = _build_strict_annotation_candidates(test_ann1)
assert len(candidates1) >= 1, "Should have at least 1 candidate"
assert "1947-1956" in candidates1[0], "Should contain section_id"
print(f"✅ _build_strict_annotation_candidates (heading): {len(candidates1)} candidates")

test_ann2 = {
    "type": "grammar_language",
    "target_word_or_sentence": "Dutline",
    "correction": "Outline"
}
candidates2 = _build_strict_annotation_candidates(test_ann2)
assert "Dutline" in candidates2, "Should contain target_word_or_sentence"
print(f"✅ _build_strict_annotation_candidates (grammar): {len(candidates2)} candidates")

test_ann3 = {
    "target_sentence": "Ex: 14 prime ministers have been changed",
    "target_sentence_start": "Ex: 14 prime ministers"
}
candidates3 = _build_strict_annotation_candidates(test_ann3)
assert "14 prime ministers" in candidates3[0], "Should contain target_sentence"
print(f"✅ _build_strict_annotation_candidates (factual): {len(candidates3)} candidates")

# Test _clip_rect
test_rect = (10, 20, 1000, 500)
clipped = _clip_rect(test_rect, 800, 600)
assert clipped[0] >= 0 and clipped[1] >= 0, "Coordinates should be non-negative"
assert clipped[2] < 800 and clipped[3] < 600, "Coordinates should be within bounds"
print(f"✅ _clip_rect: {test_rect} → {clipped}")

# Test bounds checking
test_rect_overflow = (-5, -10, 1500, 2000)
clipped_overflow = _clip_rect(test_rect_overflow, 800, 600)
assert clipped_overflow == (0, 0, 799, 599), f"Should clip to bounds, got {clipped_overflow}"
print(f"✅ _clip_rect (overflow): {test_rect_overflow} → {clipped_overflow}")

print("\n" + "="*60)
print("✅ ALL VALIDATION CHECKS PASSED")
print("="*60)
print("\nThe annotation matching system is ready to use.")
print("\nTo test with an actual essay PDF:")
print("  cd D:\\css_proj\\insightLLM_backend")
print("  python backend/eng_essay/grade_pdf_essay.py --pdf essay.pdf --output-json result.json --output-pdf annotated.pdf")
