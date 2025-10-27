import os
import time
import random
from typing import List
from opensearchpy import OpenSearch, ConnectionError as OSConnectionError

INDEX_NAME = os.getenv("OPENSEARCH_INDEX", "policies")
EMBED_DIM = int(os.getenv("OPENSEARCH_EMBED_DIM", "384")) #Dimension del vector de embedding cuando lo tengamos
HOST = os.getenv("OPENSEARCH_HOST", "localhost")
PORT = int(os.getenv("OPENSEARCH_PORT", "9200"))

def client() -> OpenSearch:
    """Cliente para opensearch sin security plugin (HTTP plano)"""
    return OpenSearch(
        hosts=[{"host": HOST, "port": PORT}],
        http_compress=True,
        use_ssl=False,
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
        timeout=30,
    )

def wait_for_opensearch(os_client: OpenSearch, retries: int = 30, delay: float = 2.0):
    """Wait until opensearch is ready"""
    for i in range(retries):
        try:
            health = os_client.cluster.health()
            status = health.get("status")
            print(f"[i] OpenSearch health: {status}")
            if status in {"green", "yellow"}:
                return
        except OSConnectionError:
            pass
        time.sleep(delay)
    raise RuntimeError("OpenSearch timeout")

def ensure_index(os_client: OpenSearch, index_name: str, dim: int):
    """Create the index"""
    if os_client.indices.exists(index=index_name):
        print(f"[INFO] Index '{index_name}' already exists")
        return

    body = {
        "settings": {
            "index": {"knn": True}
        },
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "metadata": {"type": "object", "enabled": True},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": dim # debe coincidir con la dimension del embedding
                },
            }
        },
    }

    os_client.indices.create(index=index_name, body=body)
    print(f"[OK] Index '{index_name}' created")

#para la prueba de carga del pdf, crea un vector con 384 numeros aleatorios -EMBED_DIM = int(os.getenv("OPENSEARCH_EMBED_DIM", "384"))-
def random_unit_vector(dim: int) -> List[float]:
    v = [random.random() for _ in range(dim)]
    norm = sum(x*x for x in v) ** 0.5
    return [x / (norm + 1e-12) for x in v]

#Funcion simple para instertar un documento de prueba
def insert_sample_doc(os_client: OpenSearch, index_name: str, dim: int):
    doc = {
        "text": "he policy covers hospitalization and medical emergencies within the national territory",
        "metadata": {"source": "sample.pdf", "page": 1},
        "embedding": random_unit_vector(dim),
    }
    resp = os_client.index(index=index_name, body=doc, refresh=True)
    print(f"[OK] PDF de prueba insertado _id={resp.get('_id')}")

def main():
    os_client = client()

    wait_for_opensearch(os_client)

    ensure_index(os_client, INDEX_NAME, EMBED_DIM)
    insert_sample_doc(os_client, INDEX_NAME, EMBED_DIM)

if __name__ == "__main__":
    main()
