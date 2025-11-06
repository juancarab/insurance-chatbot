import asyncio
import os
import logging
from typing import List, Optional, Type

from pydantic import BaseModel, Field, ConfigDict

from langchain_core.documents import Document
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun

try:
    from opensearch_haystack.document_store import OpenSearchDocumentStore
    from opensearch_haystack.retriever import OpenSearchHybridRetriever

    _OS_BACKEND = "opensearch_haystack"
except ModuleNotFoundError:
    from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore
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

haystack_retriever_instance = OpenSearchHybridRetriever(
    document_store=doc_store,
    embedder=query_embedder,
    top_k_bm25=5,
    top_k_embedding=5,
    join_mode="reciprocal_rank_fusion",
)


def _convert_docs(haystack_docs: List[HaystackDocument]) -> List[Document]:
    """Convierte docs de Haystack al Document de LangChain asegurando que
    el texto real vaya en page_content.
    """
    out: List[Document] = []
    for h_doc in haystack_docs:
        raw_meta = dict(h_doc.meta or {})
        inner = raw_meta.pop("metadata", {}) if "metadata" in raw_meta else {}
        meta = {**inner, **raw_meta}

        meta["score"] = getattr(h_doc, "score", None)
        meta["opensearch_backend"] = _OS_BACKEND

        page_content = (
            h_doc.content
            or meta.get("text")
            or meta.get("content")
            or ""
        )

        meta.pop("text", None)
        meta.pop("content", None)

        out.append(Document(page_content=page_content, metadata=meta))

    logger.debug("OpenSearch devolvió %d documentos", len(out))
    return out


class RetrieverInput(BaseModel):
    query: str = Field(
        description="User query to search insurance policy documents in OpenSearch"
    )
    k: Optional[int] = Field(
        None,
        description="Número de documentos a recuperar (sobrescribe el defecto)",
    )


class HybridOpenSearchTool(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "hybrid_opensearch_search"
    description: str = (
        "Busca en el índice de OpenSearch (BM25 + embeddings con RRF) y devuelve "
        "una lista de documentos de pólizas de seguros chilenos con su metadata (archivo, página, score)."
    )
    args_schema: Type[BaseModel] = RetrieverInput
    haystack_retriever: OpenSearchHybridRetriever

    def _run(
        self,
        query: str,
        k: Optional[int] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> List[Document]:
        try:
            # el usuario quiere k docs finales
            k_user = k or 5
            # pero para fusionar es mejor traer un poco más
            bm25_k = max(k_user, 10)
            emb_k = max(k_user, 10)

            results = self.haystack_retriever.run(
                query=query,
                top_k_bm25=bm25_k,
                top_k_embedding=emb_k,
            )
            docs = _convert_docs(results.get("documents", []))
            return docs[:k_user]
        except Exception as e:
            logger.exception("HybridOpenSearchTool._run failed: %s", e)
            # devolvemos un doc de error para que el agente pueda decirlo
            return [
                Document(
                    page_content="No se pudo consultar OpenSearch.",
                    metadata={
                        "error": str(e),
                        "opensearch_backend": _OS_BACKEND,
                    },
                )
            ]

    async def _arun(
        self,
        query: str,
        k: Optional[int] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> List[Document]:
        try:
            k_user = k or 5
            bm25_k = max(k_user, 10)
            emb_k = max(k_user, 10)

            results = await asyncio.to_thread(
                self.haystack_retriever.run,
                query=query,
                top_k_bm25=bm25_k,
                top_k_embedding=emb_k,
            )
            docs = _convert_docs(results.get("documents", []))
            return docs[:k_user]
        except Exception as e:
            logger.exception("HybridOpenSearchTool._arun failed: %s", e)
            return [
                Document(
                    page_content="No se pudo consultar OpenSearch.",
                    metadata={
                        "error": str(e),
                        "opensearch_backend": _OS_BACKEND,
                    },
                )
            ]


retrieval_tool = HybridOpenSearchTool(
    haystack_retriever=haystack_retriever_instance
)