from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
from sentence_transformers.cross_encoder import CrossEncoder
from langchain_core.documents import Document

from ...config import get_settings

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Reranker basado en Cross-Encoder para reordenar documentos recuperados.
    Utiliza el modelo ms-marco-MiniLM-L-6-v2 por defecto.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        batch_size: Optional[int] = None,
        device: Optional[str] = None,
    ):
        settings = get_settings()
        self.model_name = model_name or settings.rerank_model
        self.batch_size = batch_size or settings.rerank_batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(
            f"Initializing CrossEncoderReranker with model={self.model_name} "
            f"batch_size={self.batch_size} device={self.device}"
        )
        try:
            self.model = CrossEncoder(self.model_name, device=self.device)
        except Exception as e:
            logger.error(
                f"Failed to load CrossEncoder model {self.model_name}. "
                f"Ensure 'sentence-transformers', 'torch', and 'transformers' are installed. Error: {e}",
                exc_info=True
            )
            raise

    def _prepare_pairs(
        self, query: str, documents: List[Document]
    ) -> List[Tuple[str, str]]:
        """Prepara los pares query-documento para el cross-encoder."""
        return [(query, doc.page_content) for doc in documents]

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
    ) -> List[Document]:
        """
        Reordena los documentos usando el cross-encoder.

        Args:
            query: La consulta del usuario
            documents: Lista de documentos a reordenar
            top_k: Número de documentos a retornar (usa settings.rerank_top_k por defecto)

        Returns:
            Los top_k documentos reordenados por relevancia
        """
        if not documents:
            return []

        settings = get_settings()
        effective_top_k = min(
            top_k or settings.rerank_top_k,
            settings.rerank_top_k,
            len(documents)
        )

        pairs = self._prepare_pairs(query, documents)

        try:
            scores = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )

            doc_scores = list(zip(documents, scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)

            reranked_docs = []
            for doc, new_score in doc_scores[:effective_top_k]:
                if "score" in doc.metadata:
                    doc.metadata["initial_score"] = doc.metadata["score"]
                
                doc.metadata["rerank_score"] = float(new_score)
                doc.metadata["score"] = float(new_score)
                
                reranked_docs.append(doc)

            return reranked_docs

        except Exception as e:
            logger.exception("Error during reranking: %s", e)
            return documents[:effective_top_k]