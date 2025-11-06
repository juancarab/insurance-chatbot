# ingest.py — versión parametrizada por entorno
import os, glob, argparse
from typing import List, Dict

# --- Carga .env si está disponible ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from opensearchpy import OpenSearch, helpers
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# --- Config desde entorno ---
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", "9200"))
OPENSEARCH_USER = os.getenv("OPENSEARCH_USER", "admin")
OPENSEARCH_PASSWORD = os.getenv("OPENSEARCH_PASSWORD", "admin")

INDEX_NAME = os.getenv("OPENSEARCH_INDEX", "policies")
PDF_DIR = os.getenv("PDF_DIR", "./data/raw_policies")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

# Acepta EMBEDDING_MODEL (como en tu .env) o MODEL_NAME (fallback)
MODEL_NAME = os.getenv("EMBEDDING_MODEL", os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"))
EMBED_DIM = int(os.getenv("OPENSEARCH_EMBED_DIM", "384"))
BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "128"))

# --- Cliente OpenSearch ---
client = OpenSearch(
    hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
    http_auth=(OPENSEARCH_USER, OPENSEARCH_PASSWORD),
    use_ssl=False, verify_certs=False, ssl_show_warn=False, ssl_assert_hostname=False,
)

# --- Cargador de embeddings ---
embedder = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    encode_kwargs={"normalize_embeddings": True, "batch_size": BATCH_SIZE},
)

# --- Mapeo índice (texto + vector) ---
MAPPING = {
    "settings": {
        "index": {
            "knn": True,
            "number_of_shards": 1,
            "number_of_replicas": 0
        }
    },
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "metadata": {
                "type": "object",
                "properties": {
                    "file_name": {"type": "keyword"},
                    "page": {"type": "integer"},
                    "chunk_id": {"type": "integer"},
                },
            },
            "embedding": {
                "type": "knn_vector",
                "dimension": EMBED_DIM,
                "method": {
                    "name": "hnsw",
                    "engine": "nmslib",
                    "space_type": "cosinesimil",
                    "parameters": {"ef_construction": 128, "m": 24},
                },
            },
        }
    },
}

def ensure_index(recreate: bool) -> None:
    exists = client.indices.exists(index=INDEX_NAME)
    if exists and recreate:
        client.indices.delete(index=INDEX_NAME, ignore=[404])
        exists = False
    if not exists:
        client.indices.create(index=INDEX_NAME, body=MAPPING)
        print(f"Índice creado: {INDEX_NAME}")
    else:
        client.delete_by_query(index=INDEX_NAME, body={"query": {"match_all": {}}}, refresh=True)
        print(f"Índice limpiado: {INDEX_NAME}")

# --- Cargar páginas de PDF ---
def load_pages(folder: str) -> List[Dict]:
    pdfs = sorted(glob.glob(os.path.join(folder, "*.pdf")))
    if not pdfs:
        raise SystemExit(f"No se encontraron PDFs en: {folder}")
    pages = []
    for f in pdfs:
        for i, d in enumerate(PyPDFLoader(f).load(), start=1):
            t = (d.page_content or "").strip()
            if t:
                pages.append({"file_name": os.path.basename(f), "page": i, "text": t})
    return pages

# --- Partir en chunks ---
def make_chunks(pages: List[Dict]):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    texts, metas = [], []
    for p in pages:
        for j, doc in enumerate(splitter.create_documents([p["text"]])):
            texts.append(doc.page_content)
            metas.append({"file_name": p["file_name"], "page": p["page"], "chunk_id": j})
    return texts, metas

# --- Indexar en bloque ---
def bulk_index(texts, metas) -> int:
    vecs = embedder.embed_documents(texts)  # batch
    actions = [{
        "_op_type": "index",
        "_index": INDEX_NAME,
        "_source": {"text": t, "metadata": m, "embedding": v},
    } for t, m, v in zip(texts, metas, vecs)]
    if actions:
        helpers.bulk(client, actions, chunk_size=500)
        client.indices.refresh(index=INDEX_NAME)
    return len(actions)

# --- Consulta de prueba (opcional) ---
def knn_test(q: str, k: int = 5) -> None:
    qv = embedder.embed_query(q)
    body = {"size": k, "query": {"knn": {"embedding": {"vector": qv, "k": k}}}, "_source": ["text", "metadata"]}
    res = client.search(index=INDEX_NAME, body=body)
    print("\nResultados k-NN:")
    for i, h in enumerate(res.get("hits", {}).get("hits", []), 1):
        s = h["_source"]; m = s["metadata"]; prev = s["text"].replace("\n", " ")
        print(f"[{i}] {m['file_name']} p.{m['page']}  score={h['_score']:.3f}")
        print(f"    {prev[:160]}{'...' if len(prev) > 160 else ''}")

# --- Main ---
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--recreate", action="store_true", help="Recrear el índice")
    ap.add_argument("--test_query", type=str, help="Pregunta de prueba k-NN")
    args = ap.parse_args()

    ensure_index(recreate=args.recreate)
    pages = load_pages(PDF_DIR)
    texts, metas = make_chunks(pages)
    n = bulk_index(texts, metas)

    num_pdfs = len({p["file_name"] for p in pages})
    print(f"Listo. PDFs={num_pdfs} | páginas={len(pages)} | chunks={n} | índice={INDEX_NAME} | chunk={CHUNK_SIZE}/{CHUNK_OVERLAP} | modelo={MODEL_NAME}")

    if args.test_query:
        knn_test(args.test_query)
