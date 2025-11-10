import pytest
from unittest.mock import Mock, patch
from langchain_core.documents import Document

from ..reranker import CrossEncoderReranker

@pytest.fixture
def sample_documents():
    return [
        Document(page_content="Este es un documento sobre seguros de vida", metadata={"score": 0.8}),
        Document(page_content="Este documento habla de seguros de auto", metadata={"score": 0.7}),
        Document(page_content="Información sobre seguros de hogar", metadata={"score": 0.6}),
        Document(page_content="Póliza de seguro de viaje", metadata={"score": 0.5}),
    ]

@pytest.fixture
def mock_cross_encoder():
    with patch("sentence_transformers.cross_encoder.CrossEncoder") as mock:
        # Simular scores del reranker
        mock.return_value.predict.return_value = [0.95, 0.85, 0.75, 0.65]
        yield mock

def test_reranker_initialization():
    reranker = CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size=32
    )
    assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert reranker.batch_size == 32

def test_prepare_pairs():
    reranker = CrossEncoderReranker()
    query = "seguro de vida"
    docs = [
        Document(page_content="Documento 1"),
        Document(page_content="Documento 2"),
    ]
    pairs = reranker._prepare_pairs(query, docs)
    assert len(pairs) == 2
    assert pairs[0] == ("seguro de vida", "Documento 1")
    assert pairs[1] == ("seguro de vida", "Documento 2")

def test_rerank_empty_documents():
    reranker = CrossEncoderReranker()
    result = reranker.rerank("query", [])
    assert result == []

def test_rerank_documents(mock_cross_encoder, sample_documents):
    reranker = CrossEncoderReranker()
    query = "seguro de vida"
    
    # Rerank documentos
    reranked_docs = reranker.rerank(query, sample_documents, top_k=2)
    
    # Verificar que tenemos el número correcto de documentos
    assert len(reranked_docs) == 2
    
    # Verificar que los scores fueron actualizados
    assert reranked_docs[0].metadata["rerank_score"] == 0.95
    assert reranked_docs[0].metadata["initial_score"] == 0.8
    assert reranked_docs[0].metadata["score"] == 0.95
    
    # Verificar ordenamiento
    assert reranked_docs[0].metadata["rerank_score"] > reranked_docs[1].metadata["rerank_score"]

def test_rerank_handles_errors(mock_cross_encoder, sample_documents):
    mock_cross_encoder.return_value.predict.side_effect = Exception("Test error")
    
    reranker = CrossEncoderReranker()
    query = "seguro de vida"
    
    # Debería manejar el error y retornar los docs originales
    result = reranker.rerank(query, sample_documents, top_k=2)
    assert len(result) == 2
    assert result[0].metadata["score"] == 0.8  # Score original