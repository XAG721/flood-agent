FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLOOD_DB_PATH=/app/data/flood_warning_system.db \
    FLOOD_ENVIRONMENT=development

WORKDIR /app
COPY pyproject.toml README.md ./
COPY flood_system ./flood_system
COPY scripts ./scripts
COPY infra ./infra
RUN pip install --no-cache-dir ".[postgres]"

RUN useradd --create-home --uid 10001 flood && mkdir -p /app/data && chown -R flood:flood /app
USER flood

EXPOSE 8000
HEALTHCHECK --interval=20s --timeout=3s --retries=5 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=2)"
CMD ["uvicorn", "flood_system.api:app", "--host", "0.0.0.0", "--port", "8000"]
