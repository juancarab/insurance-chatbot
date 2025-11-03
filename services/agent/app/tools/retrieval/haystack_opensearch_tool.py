import asyncio
import os
import logging
from typing import List, Optional, Type

from pydantic import BaseModel, Field, ConfigDict

from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import (
    CallbackManagerForRetrieverRun,
    CallbackManagerForToolRun,
)
from langchain_core.documents import Document
from langchain_core.tools import BaseTool

try:
    from opensearch_haystack.document_store import OpenSearchDocumentStore  # type: ignore
    from opensearch_haystack.retriever import (  # type: ignore
        OpenSearchHybridRetriever,
    )
    _OS_BACKEND = "opensearch_haystack"
except ModuleNotFoundError:
    from haystack_integrations.document_stores.opensearch import (  # type: ignore
        OpenSearchDocumentStore,
    )
    from haystack_integrations.components.retrievers.opensearch import (  # type: ignore
        OpenSearchHybridRetriever,
    )
    _OS_BACKEND = "haystack_integrations"

from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.dataclasses import Document as HaystackDocument

logger = logging.getLogger(__name__)

# Config
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "http://localhost:9200")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", "9200"))
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "policies")
EMBED_DIM = int(os.getenv("OPENSEARCH_EMBED_DIM", "384"))
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Keep the same schema your project expects
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

# Hybrid retriever: BM25 + embeddings + RRF
haystack_retriever_instance = OpenSearchHybridRetriever(
    document_store=doc_store,
    embedder=query_embedder,
    top_k_bm25=5,
    top_k_embedding=5,
    join_mode="reciprocal_rank_fusion",
)

class HaystackOpenSearchRetriever(BaseRetriever):
    """
    Wraps a Haystack OpenSearchHybridRetriever and exposes it as
    a LangChain BaseRetriever, returning langchain.documents.Document.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    haystack_retriever: OpenSearchHybridRetriever

    def _convert_docs(self, haystack_docs: List[HaystackDocument]) -> List[Document]:
        out: List[Document] = []
        for h_doc in haystack_docs:
            # normalize metadata (some ingestions store it under "metadata")
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
        return out

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        results = self.haystack_retriever.run(query=query)
        return self._convert_docs(results.get("documents", []))

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        results = await asyncio.to_thread(self.haystack_retriever.run, query=query)
        return self._convert_docs(results.get("documents", []))


lc_retriever = HaystackOpenSearchRetriever(
    haystack_retriever=haystack_retriever_instance
)

class RetrieverInput(BaseModel):
    """Input schema for the OpenSearch hybrid search tool."""
    query: str = Field(
        description="User query to search insurance policy documents in OpenSearch"
    )


class HybridOpenSearchTool(BaseTool):
    """
    LangChain tool that calls the hybrid OpenSearch retriever (BM25 + embeddings).
    Returns a List[Document] with metadata (including score).
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "hybrid_opensearch_search"
    description: str = (
        "Searches the OpenSearch index (BM25 + embeddings with RRF) and returns "
        "a list of relevant policy documents with their metadata."
    )
    args_schema: Type[BaseModel] = RetrieverInput

    retriever: BaseRetriever

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> List[Document]:
        try:
            return self.retriever.invoke(query)
        except Exception as e:
            logger.exception("HybridOpenSearchTool._run failed: %s", e)
            return []

    async def _arun(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> List[Document]:
        try:
            return await self.retriever.ainvoke(query)
        except Exception as e:
            logger.exception("HybridOpenSearchTool._arun failed: %s", e)
            return []

# this is the object your agent / runner will import
retrieval_tool = HybridOpenSearchTool(retriever=lc_retriever)
