from typing import List
import sys
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer 
from agent.app.config import get_settings

settings = get_settings()

EMBED_MODEL = settings.embedding_model
SUMMARIES_INDEX = settings.policy_summaries_index

def _make_client() -> OpenSearch:
    host_entry = str(settings.opensearch_host)
    hosts = [host_entry] if host_entry.startswith(("http://", "https://")) else [{"host": host_entry, "port": settings.opensearch_port}]
    http_auth = (settings.opensearch_user, settings.opensearch_password) if settings.opensearch_user else None
    
    return OpenSearch(
        hosts=hosts,
        http_auth=http_auth,
        use_ssl=bool(settings.opensearch_use_ssl),
        verify_certs=False,
        ssl_show_warn=False,
        ssl_assert_hostname=False,
    )

class FindRelevantPoliciesTool:
    """Looks up the summaries index and returns a short list of candidate file_names (PDFs)."""

    def __init__(self):
        self.client = _make_client()

    def __call__(self, question: str, top_k: int = 5) -> List[str]:
        print(f"\n[ROUTER DEBUG] Buscando polizas candidatas para: '{question}' (top_k={top_k})", file=sys.stdout, flush=True)
        
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

        print("k desde el codigo del retriever : ", top_k)

        
        try:
            response = self.client.search(index=SUMMARIES_INDEX, body=body_text)
            hits = response["hits"]["hits"]
            
            files = [h["_source"]["file_name"] for h in hits]
            print(f"[ROUTER DEBUG] Archivos encontrados ({len(files)}): {files}", file=sys.stdout, flush=True)
            
            return files
            
        except Exception as e:
            print(f"[ROUTER DEBUG] ⚠️ ERROR en la búsqueda del router: {e}", file=sys.stderr, flush=True)
            return []

def quick_find(query: str, top_k: int = 5) -> List[str]:
    return FindRelevantPoliciesTool()(query, top_k)