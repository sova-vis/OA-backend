# OA Extraction

Small Python package and CLI for extracting one handwritten question-answer pair from an image or PDF using Grok.

## What It Does

- Accepts one image (`.png`, `.jpg`, `.jpeg`) or one PDF per run.
- Builds OCR candidates from three image variants: original, grayscale autocontrast, and sharpened binary.
- Uses Grok as the primary OCR/split/classification model, then selectively falls back to Azure Document Intelligence for low-confidence or ambiguous cases.
- Applies candidate scoring, disagreement detection, targeted repair, split retries, and validation heuristics for extraction quality.
- Returns schema-valid JSON suitable for a larger pipeline.

## Installation

```bash
pip install -e ".[dev]"
```

## Configuration

Create a `.env` file or export environment variables:

```env
Grok_API=your_xai_api_key
OA_GROK_MODEL=grok-4.20-reasoning
OA_GROK_OCR_SPLIT_TIMEOUT_SECONDS=180
OA_GROK_OCR_SPLIT_MAX_RETRIES=2
OA_ENABLE_AZURE_FALLBACK=true
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://your-resource.cognitiveservices.azure.com
AZURE_DOCUMENT_INTELLIGENCE_API_KEY=your_azure_key

# Optional: second Grok pass for math answers that look fraction/chain-linearized (default off).
# OA_MATH_ANSWER_REFINE_MIN_CONFIDENCE=0.85
# OA_ENABLE_MATH_ANSWER_REFINE=true
```

`Grok_API` is the primary key variable. `XAI_API_KEY` is also supported.

When `OA_ENABLE_MATH_ANSWER_REFINE` is true, the pipeline may call Grok once more after the usual OCR/split pass for **Mathematics** only, when the answer matches an internal “fraction or equality-chain” ambiguity heuristic. The model re-transcribes **answer text only** into explicit `\\frac` / `( )/( )` form; it does not re-solve the problem. This adds latency and cost but leaves `whole_text_raw` aligned with the selected OCR candidate while `answer_raw` may be replaced if refine **confidence** meets `OA_MATH_ANSWER_REFINE_MIN_CONFIDENCE` (default `0.85`). If refine fails or is below threshold, the original `answer_raw` is kept.

## CLI Usage

```bash
oa-extract OA-images/oa_1.png --pretty
oa-extract path/to/file.pdf --json-out result.json
```

## Python Usage

```python
from oa_extraction import extract_qa

result = extract_qa("OA-images/oa_1.png")
print(result.model_dump(mode="json"))
```

## Output Shape

The result JSON contains:

- `input_type`
- `page_count`
- `whole_text_raw`
- `question_raw`
- `answer_raw`
- `question_normalized`
- `answer_normalized`
- `subject`
- `confidence`
- `flags`
- `needs_review`
- `diagnostics`

## Notes

- v1 assumes exactly one QA pair per input.
- PDFs are rendered page-by-page at 300 DPI before model calls.
- Normalization is formatting-only. The pipeline does not silently correct math semantics.
- `diagnostics` includes the selected OCR engine and variant, candidate OCR outputs, disagreement spans, repair actions, selection reasons, split-retry state, and optional `math_answer_refine` metadata when the refine step runs.
