## Flujo de Ingesta — recrear el índice desde cero

1) Preparar entorno
   - Docker arriba: `docker compose up -d`
   - Variables: usa `.env` (OPENSEARCH_HOST, OPENSEARCH_PORT, OPENSEARCH_USER, OPENSEARCH_PASSWORD, OPENSEARCH_INDEX, PDF_DIR, CHUNK_SIZE, CHUNK_OVERLAP).

2) Crear índice
   - Ejecuta: `python -m pipeline.setup_opensearch`
   - El script debe **leer TODO desde variables de entorno** y crear/eliminar/crear el índice (si existe).

3) EDA (opcional pero recomendado)
   - Ejecuta: `python -m pipeline.eda_policies`
   - Genera métricas y recomendaciones de chunking basadas en `PDF_DIR`.

4) Ingesta
   - Ejecuta: `python -m pipeline.ingest`
   - Lee PDFs desde `PDF_DIR`, parte en chunks (CHUNK_SIZE/CHUNK_OVERLAP), genera embeddings y hace `bulk` a `OPENSEARCH_INDEX`.

5) Validación rápida
   - `curl -u admin:admin "http://localhost:9200/$env:OPENSEARCH_INDEX/_count?pretty"`
   - Correr pruebas: `pytest -q`
