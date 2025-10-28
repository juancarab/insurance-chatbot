import asyncio
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

from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore
from haystack_integrations.components.retrievers.opensearch import (
    OpenSearchHybridRetriever,
)
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.dataclasses import Document as HaystackDocument


logger = logging.getLogger(__name__)

OPENSEARCH_HOST = "http://localhost:9200"
OPENSEARCH_INDEX = "policies"
EMBED_DIM = 384
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

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


class HaystackOpenSearchRetriever(BaseRetriever):
    """
    Envuelve un OpenSearchHybridRetriever de Haystack para exponerlo como
    BaseRetriever de LangChain, devolviendo langchain.documents.Document.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    haystack_retriever: OpenSearchHybridRetriever

    def _convert_docs(self, haystack_docs: List[HaystackDocument]) -> List[Document]:
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
    """Entrada de la herramienta de búsqueda."""
    query: str = Field(description="Consulta de usuario para buscar en documentos de pólizas")


class HybridOpenSearchTool(BaseTool):
    """
    Herramienta LangChain que llama al retriever híbrido (BM25 + Embeddings).
    Devuelve List[Document] con metadatos (incluyendo 'score').
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "hybrid_opensearch_search"
    description: str = (
        "Busca en el índice OpenSearch (BM25 + embeddings con RRF) y devuelve una "
        "lista de documentos relevantes con sus metadatos."
    )
    args_schema: Type[BaseModel] = RetrieverInput

    retriever: BaseRetriever

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> List[Document]:
        try:
            return self.retriever.invoke(query)
        except Exception as e:
            logger.exception("Fallo en HybridOpenSearchTool._run: %s", e)
            return []

    async def _arun(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> List[Document]:
        try:
            return await self.retriever.ainvoke(query)
        except Exception as e:
            logger.exception("Fallo en HybridOpenSearchTool._arun: %s", e)
            return []

retrieval_tool = HybridOpenSearchTool(retriever=lc_retriever)