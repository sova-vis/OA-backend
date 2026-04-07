FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

CMD ["sh", "-c", "python -m uvicorn oa_main_pipeline.api:create_app --factory --host 0.0.0.0 --port ${PORT:-8001}"]
