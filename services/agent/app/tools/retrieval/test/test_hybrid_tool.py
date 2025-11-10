from typing import List, Optional, Type
import pytest
from unittest.mock import Mock, patch
from langchain_core.documents import Document

from ...tools.retrieval.haystack_opensearch_tool import HybridOpenSearchTool
from ...tools.retrieval.reranker import CrossEncoderReranker

@pytest.fixture
def mock_retriever():
    retriever = Mock()
    docs = [
        Document(page_content="Doc 1 sobre seguros", metadata={"score": 0.9}),
        Document(page_content="Doc 2 sobre pólizas", metadata={"score": 0.8}),
        Document(page_content="Doc 3 sobre coberturas", metadata={"score": 0.7}),
    ]
    retriever.invoke.return_value = docs
    retriever.ainvoke.return_value = docs
    return retriever

@pytest.fixture
def mock_reranker():
    with patch("sentence_transformers.cross_encoder.CrossEncoder") as mock:
        reranker = CrossEncoderReranker()
        # Simular reranking
        reranker.rerank = Mock(return_value=[
            Document(page_content="Doc rerankeado 1", metadata={"score": 0.95}),
            Document(page_content="Doc rerankeado 2", metadata={"score": 0.85}),
        ])
        yield reranker

def test_hybrid_tool_initialization(mock_retriever):
    tool = HybridOpenSearchTool(retriever=mock_retriever)
    assert tool.name == "hybrid_opensearch_search"
    assert tool.retriever == mock_retriever

def test_hybrid_tool_sync_run(mock_retriever, mock_reranker):
    tool = HybridOpenSearchTool(retriever=mock_retriever)
    tool.reranker = mock_reranker
    
    results = tool._run("seguro de vida")
    
    # Verificar que el retriever fue llamado
    mock_retriever.invoke.assert_called_once_with("seguro de vida")
    
    # Verificar que el reranker fue llamado con los docs del retriever
    mock_reranker.rerank.assert_called_once()
    
    # Verificar resultado final
    assert len(results) == 2
    assert results[0].metadata["score"] == 0.95
    assert results[1].metadata["score"] == 0.85

@pytest.mark.asyncio
async def test_hybrid_tool_async_run(mock_retriever, mock_reranker):
    tool = HybridOpenSearchTool(retriever=mock_retriever)
    tool.reranker = mock_reranker
    
    results = await tool._arun("seguro de vida")
    
    # Verificar que el retriever fue llamado
    mock_retriever.ainvoke.assert_called_once_with("seguro de vida")
    
    # Verificar que el reranker fue llamado
    mock_reranker.rerank.assert_called_once()
    
    # Verificar resultado final
    assert len(results) == 2
    assert results[0].metadata["score"] == 0.95
    assert results[1].metadata["score"] == 0.85

def test_hybrid_tool_handles_errors(mock_retriever):
    tool = HybridOpenSearchTool(retriever=mock_retriever)
    mock_retriever.invoke.side_effect = Exception("Test error")
    
    # Debería manejar el error y retornar lista vacía
    results = tool._run("query")
    assert results == []