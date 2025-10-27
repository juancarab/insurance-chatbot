import asyncio
from typing import List

from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document

from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore
from haystack_integrations.components.retrievers.opensearch import OpenSearchHybridRetriever
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.dataclasses import Document as HaystackDocument
from langchain_core.tools.retriever import create_retriever_tool

doc_store = OpenSearchDocumentStore(
    hosts=["http://localhost:9200"],
    index="policies",                
    embedding_dim=384,              
    username="admin",
    password="admin",
    create_index=False,             
    content_field="text",            
    embedding_field="embedding",     
    metadata_field="metadata",      
    knn_space_type="cosinesimil",   
    knn_engine="nmslib",             
)

query_embedder = SentenceTransformersTextEmbedder(
    model="sentence-transformers/all-MiniLM-L6-v2",
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
    haystack_retriever: OpenSearchHybridRetriever

    def _convert_docs(self, haystack_docs: List[HaystackDocument]) -> List[Document]:
        out = []
        for h_doc in haystack_docs:
            meta = dict(h_doc.meta or {})
            meta["score"] = h_doc.score
            out.append(Document(page_content=h_doc.content or "", metadata=meta))
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

retrieval_tool = create_retriever_tool(
    lc_retriever,
    name="hybrid_opensearch_search",
    description=(
        "Busca en OpenSearch con híbrido BM25 + embeddings (RRF). "
        "Devuelve pasajes relevantes con metadatos."
    ),
)