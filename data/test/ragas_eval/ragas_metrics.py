from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, faithfulness
from ragas.llms.base import BaseRagasLLM


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
        # saltar los que fallaron o no devolvieron answer
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
                # construir algo legible
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


def _evaluate_metric(
    metric,
    rows: List[Dict[str, Any]],
    requires_ground_truth: bool,
    llm_wrapper: BaseRagasLLM,
) -> Optional[float]:
    data: List[Dict[str, Any]] = []
    for row in rows:
        if requires_ground_truth and not row.get("ground_truth"):
            continue
        item = {
            "question": row.get("question", ""),
            "answer": row.get("answer", ""),
            "contexts": row.get("contexts", []),
        }
        if row.get("ground_truth"):
            item["ground_truth"] = row["ground_truth"]
        data.append(item)

    if not data:
        return None

    dataset = Dataset.from_list(data)
    result = evaluate(dataset, metrics=[metric], llm=llm_wrapper)

    value = result.get(metric.name)
    if isinstance(value, dict):
        value = value.get("score")
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return numeric


def run_ragas_evaluation(
    entries: List[Dict[str, Any]],
    results: List[Dict[str, Any]],
    llm_wrapper: BaseRagasLLM,
) -> Dict[str, Any]:
    """
    Punto de entrada que usa run_golden_set.py
    """
    rows = _prepare_ragas_rows(entries, results)
    if not rows:
        return {
            "total_rows": 0,
            "metrics": {},
            "notes": "No hay respuestas exitosas para evaluar",
        }

    metrics: Dict[str, Optional[float]] = {}
    errors: Dict[str, str] = {}

    # context_precision: requiere ground truth
    try:
        cp = _evaluate_metric(
            context_precision,
            rows,
            requires_ground_truth=True,
            llm_wrapper=llm_wrapper,
        )
        if cp is not None:
            metrics["context_precision"] = cp
    except Exception as exc:  # noqa: BLE001
        errors["context_precision_error"] = str(exc)

    # faithfulness: no requiere ground truth
    try:
        faith = _evaluate_metric(
            faithfulness,
            rows,
            requires_ground_truth=False,
            llm_wrapper=llm_wrapper,
        )
        if faith is not None:
            metrics["faithfulness"] = faith
    except Exception as exc:  # noqa: BLE001
        errors["faithfulness_error"] = str(exc)

    return {
        "total_rows": len(rows),
        "metrics": metrics,
        "errors": errors,
    }