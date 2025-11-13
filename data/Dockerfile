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

RUN pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ARG EMBEDDING_MODEL
ENV EMBEDDING_MODEL=${EMBEDDING_MODEL}

RUN python - << 'PY'
from langchain_huggingface import HuggingFaceEmbeddings
import os

model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
print(f"Pre-descargando modelo de embeddings: {model_name}")
HuggingFaceEmbeddings(model_name=model_name)
PY

COPY . .
