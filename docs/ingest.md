## Intake Flow — recreate the index from scratch

1) Prepare environment
   - Docker up: `docker compose up -d`
   - Variables: use `.env` (OPENSEARCH_HOST, OPENSEARCH_PORT, OPENSEARCH_USER, OPENSEARCH_PASSWORD, OPENSEARCH_INDEX, PDF_DIR, CHUNK_SIZE, CHUNK_OVERLAP).

2) Create index
   - Run: `python -m pipeline.setup_opensearch`
   - The script must **read EVERYTHING from environment variables** and create/delete/create the index (if it exists).

3) EDA (optional but recommended)
   - Run: `python -m pipeline.eda_policies`
   - Generates chunking metrics and recommendations based on `PDF_DIR`.

4) Ingestion
   - Run: `python -m pipeline.ingest`
   - Reads PDFs from `PDF_DIR`, splits them into chunks (CHUNK_SIZE/CHUNK_OVERLAP), generates embeddings, and performs `bulk` to `OPENSEARCH_INDEX`.

5) Fast validation
   - `curl -u admin:admin "http://localhost:9200/$env:OPENSEARCH_INDEX/_count?pretty"`
   - Run tests: `pytest -q`
