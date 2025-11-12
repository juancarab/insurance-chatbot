import argparse, csv
from opensearchpy import OpenSearch, helpers
from sentence_transformers import SentenceTransformer

INDEX = "policy_summaries_index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DIM = 384

def get_client():
    return OpenSearch(
        [{"host": "localhost", "port": 9200}],
        http_auth=("admin", "admin"),
        use_ssl=False,
        verify_certs=False,
        ssl_show_warn=False,
        ssl_assert_hostname=False,
    )

def ensure_index(c):
    body = {
        "settings": {"index": {"knn": True}},
        "mappings": {
            "properties": {
                "file_name": {"type": "keyword"},
                "title": {"type": "text"},
                "carrier": {"type": "keyword"},
                "year": {"type": "integer"},
                "summary": {"type": "text"},
                "embedding": {"type": "knn_vector", "dimension": DIM}
            }
        }
    }
    if not c.indices.exists(index=INDEX):
        c.indices.create(index=INDEX, body=body)

def bulk_load(c, rows):
    model = SentenceTransformer(EMBED_MODEL)
    actions = []
    for r in rows:
        vec = model.encode(r["summary"]).tolist()
        actions.append({
            "_op_type": "index",
            "_index": INDEX,
            "_source": {
                "file_name": r["file_name"],
                "title": r.get("title", ""),
                "carrier": r.get("carrier", ""),
                "year": int(r.get("year", 0) or 0),
                "summary": r.get("summary", ""),
                "embedding": vec
            }
        })
    helpers.bulk(c, actions)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()
    client = get_client()
    ensure_index(client)
    with open(args.csv, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    bulk_load(client, rows)
    print(f"OK: {len(rows)} summaries indexed into {INDEX}")
