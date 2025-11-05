FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates git \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel \
 && pip config set global.default-timeout 120 \
 && pip config set global.disable-pip-version-check true

COPY services/backend/requirements.txt /app/backend_requirements.txt
COPY services/agent/requirements.txt /app/agent_requirements.txt

RUN pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r /app/agent_requirements.txt \
 && pip install --no-cache-dir -r /app/backend_requirements.txt

ARG EMBEDDING_MODEL
RUN python -c "from haystack.components.embedders import SentenceTransformersTextEmbedder; SentenceTransformersTextEmbedder(model='${EMBEDDING_MODEL}')"

COPY services/backend/app /app/backend/app
COPY services/agent /app/agent

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]