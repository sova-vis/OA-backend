# Subject Grading

## O/A Levels Evaluator

### Install

```powershell
python -m pip install -r requirements.txt
```

`requirements.txt` installs the repo-local `OA-Extraction` package in editable mode, which is the only supported Mode A extraction backend.

### Run API

```powershell
python -m uvicorn oa_main_pipeline.api:app --host 0.0.0.0 --port 8001 --reload
```

### Run Frontend

```powershell
python -m http.server 5173 --directory frontend
```

Open `http://localhost:5173/`.

### Endpoints

- `POST /oa-level/evaluate`: typed question flow (Mode B)
- `POST /oa-level/evaluate-from-image/preview`: Mode A extraction preview
- `POST /oa-level/evaluate-from-image/confirm`: confirm/edit extracted text and grade
- `POST /oa-level/evaluate-from-image`: legacy Mode A auto-confirm path
- `GET /oa-level/health`: liveness
- `GET /oa-level/ready`: readiness

### Mode A

- Mode A uses the repo-local OA-Extraction adapter in [oa_main_pipeline/mode_a_oa_extraction.py](oa_main_pipeline/mode_a_oa_extraction.py).
- The underlying extraction pipeline lives in [OA-Extraction/src/oa_extraction/pipeline.py](OA-Extraction/src/oa_extraction/pipeline.py).
- Preview keeps the existing API contract, but extraction/debug data now comes from OA-Extraction diagnostics instead of the retired local Grok recovery pipeline.

See [PIPELINE_FLOW.md](PIPELINE_FLOW.md) for a higher-level flow summary.
