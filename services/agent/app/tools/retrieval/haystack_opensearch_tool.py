import asyncio
import os
import logging
from typing import List, Optional, Type

from pydantic import BaseModel, Field, ConfigDict

from ...config import get_settings
from .reranker import CrossEncoderReranker

from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import (
    CallbackManagerForRetrieverRun,
    CallbackManagerForToolRun,
)
from langchain_core.documents import Document
from langchain_core.tools import BaseTool

try:
    from opensearch_haystack.document_store import OpenSearchDocumentStore 
    from opensearch_haystack.retriever import (
        OpenSearchHybridRetriever,
    )
    _OS_BACKEND = "opensearch_haystack"
except ModuleNotFoundError:
    from haystack_integrations.document_stores.opensearch import ( 
        OpenSearchDocumentStore,
    )
    from haystack_integrations.components.retrievers.opensearch import ( 
        OpenSearchHybridRetriever,
    )
    _OS_BACKEND = "haystack_integrations"

from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.dataclasses import Document as HaystackDocument

logger = logging.getLogger(__name__)

OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "http://opensearch")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", "9200"))
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "policies")
EMBED_DIM = int(os.getenv("OPENSEARCH_EMBED_DIM", "384"))
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

doc_store = OpenSearchDocumentStore(
    hosts=[OPENSEARCH_HOST],
    index=OPENSEARCH_INDEX,
    embedding_dim=EMBED_DIM,
    create_index=False,
    content_field="text",
    embedding_field="embedding",
    metadata_field="metadata",
    knn_space_type="cosinesimil",
    knn_engine="nmslib",
)

query_embedder = SentenceTransformersTextEmbedder(
    model=EMBED_MODEL,
    normalize_embeddings=True,
)

# Get settings for retrieval parameters
settings = get_settings()
retrieval_k = settings.retrieval_top_k

# Hybrid retriever: BM25 + embeddings + RRF
haystack_retriever_instance = OpenSearchHybridRetriever(
    document_store=doc_store,
    embedder=query_embedder,
    top_k_bm25=retrieval_k,
    top_k_embedding=retrieval_k,
    join_mode="reciprocal_rank_fusion",
)

def _convert_docs(haystack_docs: List[HaystackDocument]) -> List[Document]:
    out: List[Document] = []
    for h_doc in haystack_docs:
        if h_doc.meta and "metadata" in h_doc.meta:
            meta = dict(h_doc.meta["metadata"])
            for k, v in (h_doc.meta or {}).items():
                if k != "metadata":
                    meta[k] = v
        else:
            meta = dict(h_doc.meta or {})
        meta["score"] = getattr(h_doc, "score", None)
        meta["opensearch_backend"] = _OS_BACKEND
        page_content = h_doc.content or ""
        out.append(Document(page_content=page_content, metadata=meta))
    print(out)
    return out

class RetrieverInput(BaseModel):
    query: str = Field(description="User query to search insurance policy documents in OpenSearch")
    k: Optional[int] = Field(None, description="Número de documentos a recuperar (anula el defecto)")

class HybridOpenSearchTool(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "hybrid_opensearch_search"
    description: str = (
        "Searches the OpenSearch index (BM25 + embeddings with RRF) and returns "
        "a list of relevant policy documents with their metadata."
    )
    args_schema: Type[BaseModel] = RetrieverInput
    haystack_retriever: OpenSearchHybridRetriever

    def __init__(self, retriever: BaseRetriever):
        super().__init__()
        self.retriever = retriever
        self.reranker = CrossEncoderReranker()

    def _run(
        self,
        query: str,
        k: Optional[int] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> List[Document]:
        try:
            # Recuperar documentos usando el retriever con k aumentado
            k_value = k or settings.retrieval_top_k
            results = self.haystack_retriever.run(
                query=query,
                top_k_bm25=k_value,
                top_k_embedding=k_value
            )
            docs = _convert_docs(results.get("documents", []))
            
            # Aplicar reranking si hay documentos
            if docs:
                docs = self.reranker.rerank(query, docs)
            
            return docs
        except Exception as e:
            logger.exception("HybridOpenSearchTool._run failed: %s", e)
            return []

    async def _arun(
        self,
        query: str,
        k: Optional[int] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> List[Document]:
        try:
            # Recuperar documentos usando el retriever con k aumentado
            k_value = k or settings.retrieval_top_k
            results = await asyncio.to_thread(
                self.haystack_retriever.run,
                query=query,
                top_k_bm25=k_value,
                top_k_embedding=k_value
            )
            docs = _convert_docs(results.get("documents", []))
            
            # Aplicar reranking si hay documentos
            if docs:
                # El reranking se ejecuta en un thread separado para no bloquear
                docs = await asyncio.to_thread(self.reranker.rerank, query, docs)
            
            return docs
        except Exception as e:
            logger.exception("HybridOpenSearchTool._arun failed: %s", e)
            return []

retrieval_tool = HybridOpenSearchTool(
    haystack_retriever=haystack_retriever_instance
)