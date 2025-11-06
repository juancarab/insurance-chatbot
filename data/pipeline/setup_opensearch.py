# source/setup_opensearch.py — crea/recrea índice "policies"
import os, argparse
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from opensearchpy import OpenSearch

HOST       = os.getenv("OPENSEARCH_HOST", "opensearch")  # en host usa localhost (ver comando abajo)
PORT       = int(os.getenv("OPENSEARCH_PORT", "9200"))
USER       = os.getenv("OPENSEARCH_USER", "admin")
PASSWORD   = os.getenv("OPENSEARCH_PASSWORD", "admin")
INDEX      = os.getenv("OPENSEARCH_INDEX", "policies")
EMBED_DIM  = int(os.getenv("OPENSEARCH_EMBED_DIM", "384"))
SHARDS     = int(os.getenv("OPENSEARCH_SHARDS", "1"))
REPLICAS   = int(os.getenv("OPENSEARCH_REPLICAS", "0"))

MAPPING = {
    "settings": {
        "index": {
            "knn": True,
            "number_of_shards": SHARDS,
            "number_of_replicas": REPLICAS,
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

def get_client() -> OpenSearch:
    return OpenSearch(
        hosts=[{"host": HOST, "port": PORT}],
        http_auth=(USER, PASSWORD),
        use_ssl=False, verify_certs=False, ssl_show_warn=False, ssl_assert_hostname=False,
    )

def recreate_index(client: OpenSearch, index: str = INDEX, body: dict = MAPPING) -> None:
    if client.indices.exists(index=index):
        client.indices.delete(index=index, ignore=[404])
    client.indices.create(index=index, body=body)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recreate", action="store_true", help="Eliminar y crear el índice")
    ap.add_argument("--index", default=INDEX, help="Nombre del índice")
    args = ap.parse_args()

    client = get_client()
    if args.recreate:
        recreate_index(client, args.index, MAPPING)
        print(f"Índice recreado: {args.index}")
    else:
        if client.indices.exists(index=args.index):
            print(f"Índice ya existe: {args.index}")
        else:
            client.indices.create(index=args.index, body=MAPPING)
            print(f"Índice creado: {args.index}")

if __name__ == "__main__":
    main()
