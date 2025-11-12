from typing import List
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer  # se deja por si luego reactivamos híbrido
from services.agent.app.config import get_settings

settings = get_settings()

EMBED_MODEL = settings.embedding_model
SUMMARIES_INDEX = settings.policy_summaries_index

def _make_client() -> OpenSearch:
    host_entry = str(settings.opensearch_host)
    hosts = [host_entry] if host_entry.startswith(("http://", "https://")) else [{"host": host_entry, "port": settings.opensearch_port}]
    return OpenSearch(
        hosts=hosts,
        http_auth=(settings.opensearch_user, settings.opensearch_password) if settings.opensearch_user else None,
        use_ssl=bool(settings.opensearch_use_ssl),
        verify_certs=False,
        ssl_show_warn=False,
        ssl_assert_hostname=False,
    )

class FindRelevantPoliciesTool:
    """Looks up the summaries index and returns a short list of candidate file_names (PDFs)."""

    def __init__(self):
        self.client = _make_client()
        # self.model = SentenceTransformer(EMBED_MODEL)  # no se usa en BM25

    def __call__(self, question: str, top_k: int = 5) -> List[str]:
        # BM25 puro (compatibilidad OpenSearch 2.12)
        body_text = {
            "size": top_k,
            "_source": ["file_name"],
            "query": {
                "multi_match": {
                    "query": question,
                    "fields": ["summary^2", "title"]
                }
            }
        }
        hits = self.client.search(index=SUMMARIES_INDEX, body=body_text)["hits"]["hits"]
        return [h["_source"]["file_name"] for h in hits]

def quick_find(query: str, top_k: int = 5) -> List[str]:
    return FindRelevantPoliciesTool()(query, top_k)

if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "coaseguro en el extranjero"
    print(quick_find(q))
