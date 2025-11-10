"""
OpenSearch Index Setup Module.

Provides functions to connect to OpenSearch and create/recreate
the 'policies' index with the correct vector mapping.
"""
import os
import argparse
from opensearchpy import OpenSearch
import config

MAPPING = {
    "settings": {
        "index": {
            "knn": True,
            "number_of_shards": config.OPENSEARCH_SHARDS,
            "number_of_replicas": config.OPENSEARCH_REPLICAS,
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
                "dimension": config.OPENSEARCH_EMBED_DIM,
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
    """
    Initializes and returns an OpenSearch client.
    """
    return OpenSearch(
        hosts=[{"host": config.OPENSEARCH_HOST, "port": config.OPENSEARCH_PORT}],
        http_auth=(config.OPENSEARCH_USER, config.OPENSEARCH_PASSWORD),
        use_ssl=False, verify_certs=False, ssl_show_warn=False, ssl_assert_hostname=False,
    )

def recreate_index(client: OpenSearch, index: str = config.OPENSEARCH_INDEX, body: dict = MAPPING) -> None:
    """
    Deletes an index if it exists and creates it fresh.
    """
    if client.indices.exists(index=index):
        print(f"Deleting existing index: {index}")
        client.indices.delete(index=index, ignore=[404])
    print(f"Creating new index: {index}")
    client.indices.create(index=index, body=body)

def main():
    """
    Main function for standalone script execution.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--recreate", action="store_true", help="Delete and create the index")
    ap.add_argument("--index", default=config.OPENSEARCH_INDEX, help="Name of the index")
    args = ap.parse_args()

    client = get_client()
    if args.recreate:
        recreate_index(client, args.index, MAPPING)
        print(f"Index recreated: {args.index}")
    else:
        if client.indices.exists(index=args.index):
            print(f"Index already exists: {args.index}")
        else:
            client.indices.create(index=args.index, body=MAPPING)
            print(f"Index created: {args.index}")

if __name__ == "__main__":
    main()