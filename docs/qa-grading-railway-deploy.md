# Q/A Grading Railway Deployment Guide

This guide deploys Q/A grading in production using two services:

- Backend service (Node/Express): OA-backend
- Grading service (Python/FastAPI): _ext/Subject-Grading

## 1) Deploy the grading service on Railway

Create a new Railway service from this repository with:

- Root directory: _ext/Subject-Grading
- Build command: pip install -r requirements.txt
- Start command: python -m uvicorn oa_main_pipeline.api:create_app --factory --host 0.0.0.0 --port $PORT

Set these environment variables in the grading service:

- Grok_API
- AZURE_ENDPOINT
- AZURE_KEY
- OA_EMBED_BACKEND=hash
- OA_GROK_OCR_SPLIT_TIMEOUT_SECONDS=90
- OA_GROK_OCR_SPLIT_MAX_RETRIES=2
- OA_SOURCE_PRIORITY=o_level_main_first
- OA_MAIN_JSON_ROOT=O_LEVEL_MAIN_JSON

Optional but recommended:

- HF_TOKEN (if you switch to sentence_transformers embed backend)
- LOG_LEVEL=INFO

Health endpoints after deploy:

- /oa-level/health
- /oa-level/ready

## 2) Wire backend to grading service

In OA-backend Railway service, set:

- QA_GRADING_SERVICE_URL=https://<your-grading-service-domain>
- AUTO_START_QA_GRADING_SERVICE=false

Important:

- In production, do not use http://127.0.0.1:8001 unless grading service is truly in the same container.

## 3) Verify end-to-end

From backend deployment URL:

- GET /qa-grading/health
- GET /qa-grading/ready

Expected:

- health shows qa_service_url pointing to deployed grading URL
- ready returns ready=true once grading warmup completes

## 4) Frontend check

Open student Q/A grading page and test:

- Typed mode submit
- Upload mode preview
- Confirm and final grading

If you get connection errors, re-check QA_GRADING_SERVICE_URL in backend env and grading service logs.
