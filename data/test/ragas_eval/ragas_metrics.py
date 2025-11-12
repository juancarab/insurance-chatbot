from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional

from datasets import Dataset
from ragas import evaluate
from ragas.llms.base import BaseRagasLLM
from ragas.metrics import (
    context_precision,
    faithfulness,
    answer_relevancy,
    context_recall,
)
from langchain_huggingface import HuggingFaceEmbeddings as LangChainHuggingFaceEmbeddings


def _extract_question(payload: Dict[str, Any]) -> str:
    messages = payload.get("messages") or []
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return str(msg.get("content", "")).strip()
    return ""


def _prepare_ragas_rows(
    entries: List[Dict[str, Any]], results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for entry, result in zip(entries, results):
        if result.get("error") or not result.get("answer"):
            continue
        question = _extract_question(entry.get("payload", {}))
        if not question:
            continue
        sources = result.get("sources") or []
        context_texts: List[str] = []
        for src in sources:
            if not isinstance(src, dict):
                continue
            snippet = src.get("snippet") or src.get("text")
            if not snippet:
                parts: List[str] = []
                if src.get("file_name"):
                    parts.append(str(src["file_name"]))
                if src.get("page") is not None:
                    parts.append(f"p.{src['page']}")
                if src.get("chunk_id") is not None:
                    parts.append(f"chunk {src['chunk_id']}")
                meta = " ".join(parts) if parts else "fuente desconocida"
                snippet = f"Sin extracto disponible ({meta})."
            context_texts.append(str(snippet))
        if not context_texts:
            context_texts = [""]
        ground_truth = (
            entry.get("reference_answer")
            or entry.get("expected_answer")
            or entry.get("ground_truth")
            or entry.get("reference")
        )
        row: Dict[str, Any] = {
            "question": question,
            "answer": result.get("answer"),
            "contexts": context_texts,
        }
        if ground_truth:
            row["ground_truth"] = ground_truth
        rows.append(row)
    return rows


class RagasHuggingFaceEmbeddings(BaseRagasLLM):
    """
    Wrapper para los embeddings de HuggingFace que satisface
    la interfaz completa que RAGAS 'evaluate' espera.
    
    1. Hereda de BaseRagasLLM: Para obtener '.run_config'.
    2. Implementa 'embed_text', 'embed_query' y 'embed_documents'
       (y sus versiones async) mapeándolos a los métodos correctos de LangChain.
    """
    def __init__(self, model_name: str):
        super().__init__()
        self.client = LangChainHuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True}
        )
        self.run_config.timeout = 60
        self.run_config.max_workers = 4
        self.run_config.max_retries = 2

    def embed_text(self, text: str) -> List[float]:
        return self.client.embed_query(text)
        
    async def aembed_text(self, text: str) -> List[float]:
        return await self.client.aembed_query(text)

    def embed_query(self, text: str) -> List[float]:
        return self.client.embed_query(text)
        
    async def aembed_query(self, text: str) -> List[float]:
        return await self.client.aembed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.client.embed_documents(texts)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return await self.client.aembed_documents(texts)
    
    def generate_text(self, prompt, **kwargs):
        raise NotImplementedError("Este wrapper es solo para embeddings")
        
    async def agenerate_text(self, prompt, **kwargs):
        raise NotImplementedError("Este wrapper es solo para embeddings")


def run_ragas_evaluation(
    entries: List[Dict[str, Any]],
    results: List[Dict[str, Any]],
    llm_wrapper: BaseRagasLLM, 
) -> Dict[str, Any]:
    """
    Punto de entrada que usa run_golden_set.py.
    """
    rows = _prepare_ragas_rows(entries, results)
    if not rows:
        return {
            "total_rows": 0,
            "metrics": {},
            "notes": "No hay respuestas exitosas para evaluar",
        }

    dataset = Dataset.from_list(rows)

    embed_model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    ragas_embed = RagasHuggingFaceEmbeddings(model_name=embed_model_name)

    faithfulness.llm = llm_wrapper
    context_precision.llm = llm_wrapper
    
    answer_relevancy.llm = llm_wrapper
    answer_relevancy.embeddings = ragas_embed

    context_recall.llm = llm_wrapper
    context_recall.embeddings = ragas_embed

    metrics_to_run = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]
    
    final_metrics_list = []
    has_ground_truth = any(r.get("ground_truth") for r in rows)

    for metric in metrics_to_run:
        if metric.name in ("context_precision", "context_recall", "answer_correctness"):
            if has_ground_truth:
                final_metrics_list.append(metric)
        else:
            final_metrics_list.append(metric)

    if not final_metrics_list:
        return {
            "total_rows": len(rows),
            "metrics": {},
            "errors": {"setup_error": "No metrics could be configured to run."},
        }

    print(f"\nEjecutando evaluate() con {len(final_metrics_list)} métricas sobre {len(rows)} filas...")
    
    final_metrics: Dict[str, Optional[float]] = {}
    errors: Dict[str, str] = {}
    
    try:
        result = evaluate(
            dataset,
            metrics=final_metrics_list
        )
        
        for k, v in result.items():
            if isinstance(v, (float, int)) and not math.isnan(v):
                final_metrics[k] = v
            
    except Exception as exc:
        print(f"[ERROR] RAGAS falló durante la evaluación: {exc}")
        errors["evaluation_error"] = str(exc)

    return {
        "total_rows": len(rows),
        "metrics": final_metrics,
        "errors": errors,
    }