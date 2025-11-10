"""
Central configuration module for the data pipeline.

Loads all necessary environment variables for S3, OpenSearch,
EDA, and Ingestion processes.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- General ---
PDF_DIR = os.getenv("PDF_DIR", "./data/raw_policies")

# --- S3 Download ---
S3_BUCKET = os.getenv("S3_BUCKET", "anyoneai-datasets")
S3_PREFIX = os.getenv("S3_PREFIX", "queplan_insurance/")
S3_AWS_ACCESS_KEY_ID = os.getenv("S3_AWS_ACCESS_KEY_ID")
S3_AWS_SECRET_ACCESS_KEY = os.getenv("S3_AWS_SECRET_ACCESS_KEY")

# --- EDA ---
EDA_OUT_DIR = "./eda_out"
EDA_RECS_FILE = os.path.join(EDA_OUT_DIR, "eda_recommendations.json")
CHUNK_SIZE_ENV = os.getenv("CHUNK_SIZE")
CHUNK_OVERLAP_ENV = os.getenv("CHUNK_OVERLAP")

# --- OpenSearch ---
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", "9200"))
OPENSEARCH_USER = os.getenv("OPENSEARCH_USER", "admin")
OPENSEARCH_PASSWORD = os.getenv("OPENSEARCH_PASSWORD", "admin")
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "policies")
OPENSEARCH_EMBED_DIM = int(os.getenv("OPENSEARCH_EMBED_DIM", "384"))
OPENSEARCH_SHARDS = int(os.getenv("OPENSEARCH_SHARDS", "1"))
OPENSEARCH_REPLICAS = int(os.getenv("OPENSEARCH_REPLICAS", "0"))

# --- Ingestion ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "128"))