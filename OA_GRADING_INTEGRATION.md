# OA Grading Integration

`OA-backend` already contains the transport layer for the Python `Subject Grading` service. The backend feature is exposed as `OA grading` and mounted on both:

- `/oa-grading` (preferred)
- `/qa-grading` (legacy alias)

## What is wired

- `POST /oa-grading/evaluate`
  Proxies typed question grading to `POST /oa-level/evaluate`.
- `POST /oa-grading/evaluate-from-image/preview`
  Proxies image/PDF preview extraction to `POST /oa-level/evaluate-from-image/preview`.
- `POST /oa-grading/evaluate-from-image/confirm`
  Proxies confirm grading to `POST /oa-level/evaluate-from-image/confirm`.
- `GET /oa-grading/health`
  Backend-facing health for the OA grading feature.
- `GET /oa-grading/ready`
  Readiness passthrough from the Python service warmup state.

## Expected Python service

The Node backend expects the Python sidecar to expose:

- `GET /oa-level/health`
- `GET /oa-level/ready`
- `POST /oa-level/evaluate`
- `POST /oa-level/evaluate-from-image/preview`
- `POST /oa-level/evaluate-from-image/confirm`

The current `Subject Grading` repo already provides these via `oa_main_pipeline.api:create_app`.

## Local startup

From `package.json`:

```powershell
npm run oa-grading:service
```

The launcher will look for the Python repo in this order:

1. `OA_GRADING_SERVICE_ROOT`
2. `QA_GRADING_SERVICE_ROOT`
3. `_ext/Subject-Grading`
4. `_ext/Subject Grading`
5. `Subject-Grading`
6. `Subject Grading`

## Environment variables

Set these in backend `.env`:

```env
OA_GRADING_SERVICE_URL=http://127.0.0.1:8001
AUTO_START_OA_GRADING_SERVICE=true
OA_GRADING_SERVICE_ROOT=
```

`QA_GRADING_*` aliases still work, but new integration should use `OA_GRADING_*`.
