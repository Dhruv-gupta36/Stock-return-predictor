FROM python:3.11-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

FROM base AS dependencies

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

FROM dependencies AS application

COPY . .

RUN mkdir -p \
    data/raw \
    data/processed \
    data/external \
    models \
    reports/figures \
    logs && \
    touch \
    data/raw/.gitkeep \
    data/processed/.gitkeep \
    data/external/.gitkeep \
    models/.gitkeep \
    reports/figures/.gitkeep \
    logs/.gitkeep

ENV PYTHONPATH=/app
ENV MODEL_DIR=/app/models
ENV CONFIG_PATH=/app/configs/config.yaml

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
