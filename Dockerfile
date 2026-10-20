FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY alembic.ini ./
COPY alembic ./alembic

RUN python -m pip install --upgrade pip \
    && python -m pip install .

EXPOSE 8000

CMD ["uvicorn", "rag_agent.api:app", "--host", "0.0.0.0", "--port", "8000"]