import asyncio
import os
import logging
from typing import List, Optional, Type

from pydantic import BaseModel, Field, ConfigDict

from langchain_core.documents import Document
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun

from .reranker import CrossEncoderReranker
from ...config import get_settings

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

try:
    from opensearchpy import (
        ConnectionError as OSConnectionError,
        TransportError,
        RequestError,
    )
except Exception:  
    OSConnectionError = TransportError = RequestError = Exception

from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.dataclasses import Document as HaystackDocument

logger = logging.getLogger(__name__)

settings = get_settings()

OPENSEARCH_HOST = settings.opensearch_host
OPENSEARCH_PORT = settings.opensearch_port
OPENSEARCH_INDEX = settings.opensearch_index
EMBED_DIM = settings.opensearch_embed_dim
EMBED_MODEL = settings.embedding_model

RETRIEVAL_K_NET = settings.retrieval_top_k

doc_store = OpenSearchDocumentStore(
    hosts=[OPENSEARCH_HOST],
    port=OPENSEARCH_PORT,
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
    top_k_bm25=RETRIEVAL_K_NET,
    top_k_embedding=RETRIEVAL_K_NET,
    join_mode="reciprocal_rank_fusion",
)

try:
    reranker_instance = CrossEncoderReranker()
except Exception as e:
    logger.error(f"Failed to initialize Reranker, retrieval will be degraded: {e}", exc_info=True)
    reranker_instance = None


def _convert_docs(haystack_docs: List[HaystackDocument]) -> List[Document]:
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
        description="Número de documentos a recuperar (ignorado, usa config de reranker)",
    )


class HybridOpenSearchTool(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "hybrid_opensearch_search"
    description: str = (
        "Busca en el índice de OpenSearch (BM25 + embeddings + RRF) y devuelve "
        "una lista de documentos de pólizas de seguros chilenos con su metadata (archivo, página, score). "
        "Aplica reranking para máxima precisión. "
        "En caso de error devuelve un documento con el error en metadata['error']."
    )
    args_schema: Type[BaseModel] = RetrieverInput
    haystack_retriever: OpenSearchHybridRetriever
    reranker: Optional[CrossEncoderReranker]

    def _return_error_doc(self, msg: str, error_type: str, exc: Exception | None = None) -> List[Document]:
        return [
            Document(
                page_content="No se pudo consultar OpenSearch.",
                metadata={
                    "error": msg,
                    "error_type": error_type,
                    "opensearch_backend": _OS_BACKEND,
                    "exception": str(exc) if exc else None,
                },
            )
        ]

    def _run(
        self,
        query: str,
        k: Optional[int] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> List[Document]:
        
        bm25_k = RETRIEVAL_K_NET
        emb_k = RETRIEVAL_K_NET

        try:
            results = self.haystack_retriever.run(
                query=query,
                top_k_bm25=bm25_k,
                top_k_embedding=emb_k,
            )
            docs = _convert_docs(results.get("documents", []))
            
            if docs and self.reranker:
                logger.debug(f"Reranking {len(docs)} docs for query: {query}")
                docs = self.reranker.rerank(query, docs)
                logger.debug(f"Reranking returned {len(docs)} docs")
            elif docs:
                docs = docs[:settings.rerank_top_k]
            
            return docs

        except OSConnectionError as e:
            logger.error("No se pudo conectar a OpenSearch en %s: %s", OPENSEARCH_HOST, e, exc_info=True)
            return self._return_error_doc(
                "Error: Unable to connect to the OpenSearch database.",
                "connection_error",
                e,
            )
        except RequestError as e:
            logger.warning("Consulta inválida a OpenSearch. query=%s error=%s", query, e, exc_info=True)
            return self._return_error_doc(
                f"Error: Invalid Query OpenSearch ({e.error}).",
                "request_error",
                e,
            )
        except TransportError as e:
            logger.error("Error de transporte con OpenSearch (status %s): %s", getattr(e, "status_code", "?"), e, exc_info=True)
            return self._return_error_doc(
                f"Error: Transport error with OpenSearch (status {getattr(e, 'status_code', 'unknown')}).",
                "transport_error",
                e,
            )
        except Exception as e:
            logger.exception("HybridOpenSearchTool._run failed inesperadamente")
            return self._return_error_doc(
                f"Error: Internal search unavailable ({type(e).__name__}).",
                "unexpected_error",
                e,
            )

    async def _arun(
        self,
        query: str,
        k: Optional[int] = None, 
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> List[Document]:
        
        bm25_k = RETRIEVAL_K_NET
        emb_k = RETRIEVAL_K_NET

        try:
            results = await asyncio.to_thread(
                self.haystack_retriever.run,
                query=query,
                top_k_bm25=bm25_k,
                top_k_embedding=emb_k,
            )
            docs = _convert_docs(results.get("documents", []))
            
            if docs and self.reranker:
                logger.debug(f"Async Reranking {len(docs)} docs for query: {query}")
                docs = await asyncio.to_thread(self.reranker.rerank, query, docs)
                logger.debug(f"Async Reranking returned {len(docs)} docs")
            elif docs:
                docs = docs[:settings.rerank_top_k]
                
            return docs

        except OSConnectionError as e:
            logger.error("ASYNC: no se pudo conectar a OpenSearch: %s", e, exc_info=True)
            return self._return_error_doc(
                "Error: Unable to connect to the OpenSearch database.",
                "connection_error",
                e,
            )
        except RequestError as e:
            logger.warning("ASYNC: consulta inválida. query=%s error=%s", query, e, exc_info=True)
            return self._return_error_doc(
                f"Error: Invalid Query OpenSearch ({e.error}).",
                "request_error",
                e,
            )
        except TransportError as e:
            logger.error("ASYNC: error de transporte: %s", e, exc_info=True)
            return self._return_error_doc(
                f"Error: Transport error with OpenSearch (status {getattr(e, 'status_code', 'unknown')}).",
                "transport_error",
                e,
            )
        except Exception as e:
            logger.exception("HybridOpenSearchTool._arun failed inesperadamente")
            return self._return_error_doc(
                f"Error: Internal search unavailable ({type(e).__name__}).",
                "unexpected_error",
                e,
            )

retrieval_tool = HybridOpenSearchTool(
    haystack_retriever=haystack_retriever_instance,
    reranker=reranker_instance
)