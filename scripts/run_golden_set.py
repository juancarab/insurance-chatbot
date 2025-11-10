"""Ejecuta el golden set contra la API del backend y guarda los resultados.

Uso:
    python scripts/run_golden_set.py \
        --base-url http://127.0.0.1:8001/chat \
        --golden-set data/golden_set/golden_set.json \
        --output results/golden_run.jsonl

El script carga las preguntas, realiza las llamadas al endpoint /chat y almacena
las respuestas del agente junto con metadatos (status HTTP, tiempo de respuesta,
conteo de fuentes, etc.)."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import os
import requests

from datasets import Dataset
from ragas import evaluate
from ragas.llms.base import BaseRagasLLM
from ragas.metrics import context_precision, faithfulness

import requests
from langchain_core.outputs import Generation, LLMResult
from langchain_core.prompt_values import PromptValue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ejecuta el golden set del chatbot")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8001/chat",
        help="Endpoint completo del backend FastAPI (/chat)",
    )
    parser.add_argument(
        "--golden-set",
        default="data/golden_set/golden_set.json",
        help="Ruta al archivo JSON con el golden set",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Archivo JSONL donde se escribirán los resultados (por defecto results/golden_<timestamp>.jsonl)",
    )
    parser.add_argument(
        "--baseline-metrics",
        default=None,
        help="Ruta a un archivo de métricas previo para comparar resultados RAGAS",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Timeout por petición en segundos",
    )
    parser.add_argument(
        "--ragas-model",
        default="gemini-2.5-flash",
        help="Modelo de Gemini que usará RAGAS para evaluar (default: gemini-2.5-flash)",
    )
    parser.add_argument(
        "--gemini-api-key",
        default=None,
        help="API key de Gemini (si no se setea, se usa GEMINI_API_KEY del entorno)",
    )
    return parser.parse_args()


def load_golden_set(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("El golden set debe ser una lista de escenarios")
    return data


def ensure_output_path(path: Path) -> Path:
    if path.is_dir():
        raise ValueError("--output debe ser un archivo, no un directorio")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def run_entry(entry: Dict[str, Any], base_url: str, timeout: float) -> Dict[str, Any]:
    payload = entry.get("payload", {})
    start = time.perf_counter()
    try:
        response = requests.post(base_url, json=payload, timeout=timeout)
        elapsed = time.perf_counter() - start
        response.raise_for_status()
        body = response.json()
        return {
            "id": entry.get("id"),
            "category": entry.get("category"),
            "description": entry.get("description"),
            "status_code": response.status_code,
            "elapsed_ms": round(elapsed * 1000, 2),
            "answer": body.get("answer"),
            "sources": body.get("sources", []),
            "usage": body.get("usage", {}),
            "debug": body.get("debug"),
            "raw_response": body,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - start
        status_code = None
        response_body = None
        if hasattr(exc, "response") and getattr(exc, "response") is not None:
            try:
                status_code = exc.response.status_code  # type: ignore[attr-defined]
                response_body = exc.response.text  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover
                status_code = None
                response_body = None

        return {
            "id": entry.get("id"),
            "category": entry.get("category"),
            "description": entry.get("description"),
            "status_code": status_code,
            "elapsed_ms": round(elapsed * 1000, 2),
            "answer": None,
            "sources": None,
            "usage": None,
            "debug": None,
            "raw_response": response_body,
            "error": str(exc),
        }


def _extract_question(payload: Dict[str, Any]) -> str:
    messages = payload.get("messages") or []
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return str(msg.get("content", "")).strip()
    return ""


def _prepare_ragas_rows(entries: List[Dict[str, Any]], results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for entry, result in zip(entries, results):
        if result.get("error") or not result.get("answer"):
            continue

        payload = entry.get("payload", {})
        question = _extract_question(payload)
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
                snippet = f"Sin extracto disponible ( {meta} )."
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


class GeminiRagasLLM(BaseRagasLLM):
    """Wrapper mínimo para utilizar Gemini dentro de RAGAS"""

    def __init__(self, model: str, api_key: str):
        super().__init__()
        if not model.startswith("models/"):
            model = f"models/{model}"
        self.model_name = model
        self.endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/{self.model_name}:generateContent"
        )
        self.api_key = api_key

        self.run_config.timeout = 180
        self.run_config.max_workers = 1
        self.run_config.max_retries = 1

    def _raw_generate(self, prompt: str, temperature: Optional[float]) -> str:
        generation_config = {
            "temperature": 0.2 if temperature is None else float(temperature),
            "maxOutputTokens": 512,
        }
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": generation_config,
            "systemInstruction": {
                "role": "system",
                "parts": [
                    {
                        "text": (
                            "Debes seguir exactamente las instrucciones del usuario y responder SIEMPRE "
                            "con JSON válido. No agregues comentarios ni texto fuera del JSON."
                        )
                    }
                ],
            },
        }
        response = requests.post(
            self.endpoint,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
            },
            json=payload,
            timeout=120,
        )
        if not response.ok:
            raise RuntimeError(
                f"Gemini falló al generar texto: {response.status_code} {response.text}"
            )
        data = response.json()
        if isinstance(data, dict):
            candidates = data.get("candidates") or []
        else:
            candidates = []

        parts: List[str] = []
        for cand in candidates:
            content = cand.get("content") or {}
            for part in content.get("parts", []):
                piece = part.get("text")
                if piece:
                    parts.append(piece)

        if parts:
            return "\n".join(parts)

        # fallback en caso de estructura distinta
        text = data.get("text") if isinstance(data, dict) else None
        if text:
            return str(text)
        return ""

    def _ensure_json(self, text: str, temperature: Optional[float], attempt: int = 1) -> str:
        cleaned = (text or "").strip()
        if cleaned.startswith("{") or cleaned.startswith("["):
            return cleaned
        if attempt >= 3:
            return cleaned  # devolvemos lo que haya para no fallar

        convert_prompt = (
            "Convierte la siguiente respuesta en JSON válido. Devuelve solo el JSON sin comentarios.\n"
            f"Respuesta original:\n{cleaned}"
        )
        converted = self._raw_generate(convert_prompt, 0.0)
        return self._ensure_json(converted, temperature, attempt + 1)

    def _generate(self, prompt: str, temperature: Optional[float]) -> str:
        first = self._raw_generate(prompt, temperature)
        return self._ensure_json(first, temperature)

    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 0.01,
        stop: Optional[List[str]] = None,
        callbacks=None,
    ) -> LLMResult:
        text = self._generate(prompt.to_string(), temperature)
        return LLMResult(generations=[[Generation(text=text)]])

    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: Optional[float] = 0.01,
        stop: Optional[List[str]] = None,
        callbacks=None,
    ) -> LLMResult:
        text = await asyncio.to_thread(self._generate, prompt.to_string(), temperature)
        return LLMResult(generations=[[Generation(text=text)]])

    def is_finished(self, response: LLMResult) -> bool:  # pragma: no cover
        return True


def build_ragas_llm(model_name: str, api_key: str) -> BaseRagasLLM:
    return GeminiRagasLLM(model=model_name, api_key=api_key)


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
        entry = {
            "question": row.get("question", ""),
            "answer": row.get("answer", ""),
            "contexts": row.get("contexts", []),
        }
        if row.get("ground_truth"):
            entry["ground_truth"] = row["ground_truth"]
        data.append(entry)

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
    rows = _prepare_ragas_rows(entries, results)
    if not rows:
        return {
            "total_rows": 0,
            "metrics": {},
            "notes": "No hay respuestas exitosas para evaluar",
        }

    metrics: Dict[str, Optional[float]] = {}

    try:
        metrics["faithfulness"] = _evaluate_metric(
            faithfulness, rows, requires_ground_truth=False, llm_wrapper=llm_wrapper
        )
    except Exception as exc:  # noqa: BLE001
        metrics["faithfulness_error"] = str(exc)

    try:
        metrics["context_precision"] = _evaluate_metric(
            context_precision, rows, requires_ground_truth=True, llm_wrapper=llm_wrapper
        )
    except Exception as exc:  # noqa: BLE001
        metrics["context_precision_error"] = str(exc)

    cleaned_metrics = {
        k: v for k, v in metrics.items() if not k.endswith("_error") and v is not None
    }

    return {
        "total_rows": len(rows),
        "metrics": cleaned_metrics,
        "errors": {k: v for k, v in metrics.items() if k.endswith("_error")},
    }


def load_baseline_metrics(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        print(f"[Aviso] No se encontró archivo de métricas base: {path}")
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def build_comparison(new_metrics: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
    comparison: Dict[str, Dict[str, Optional[float]]] = {}
    new_values = new_metrics.get("metrics", {})
    old_values = baseline.get("metrics", {})
    for metric_name, new_value in new_values.items():
        old_value = old_values.get(metric_name)
        if old_value is None or new_value is None:
            delta = None
        else:
            delta = round(float(new_value) - float(old_value), 4)
        comparison[metric_name] = {
            "baseline": old_value,
            "current": new_value,
            "delta": delta,
        }
    return comparison


def main() -> None:
    args = parse_args()
    golden_path = Path(args.golden_set)
    entries = load_golden_set(golden_path)

    if args.output:
        output_path = ensure_output_path(Path(args.output))
    else:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_path = ensure_output_path(Path("results") / f"golden_{timestamp}.jsonl")

    print(f"Golden set: {golden_path} ({len(entries)} escenarios)")
    print(f"Endpoint:   {args.base_url}")
    print(f"Resultados: {output_path}\n")

    successes = 0
    failures = 0
    all_results: List[Dict[str, Any]] = []

    with output_path.open("w", encoding="utf-8") as fh:
        for idx, entry in enumerate(entries, start=1):
            result = run_entry(entry, args.base_url, args.timeout)
            all_results.append(result)
            fh.write(json.dumps(result, ensure_ascii=False) + "\n")

            if result["error"]:
                failures += 1
                print(f"[{idx}/{len(entries)}] {entry.get('id')} ✗ {result['error']}")
            else:
                successes += 1
                sources_count = len(result.get("sources") or [])
                print(
                    f"[{idx}/{len(entries)}] {entry.get('id')} ✓ {result['elapsed_ms']} ms, "
                    f"fuentes={sources_count}"
                )

    api_key = args.gemini_api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Debes proporcionar la API key de Gemini (argumento --gemini-api-key o variable GEMINI_API_KEY)."
        )

    ragas_llm = build_ragas_llm(args.ragas_model, api_key)

    metrics_summary = run_ragas_evaluation(entries, all_results, ragas_llm)
    summary_path = output_path.with_suffix(".metrics.json")
    summary_payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "base_url": args.base_url,
        "results_file": str(output_path),
        "summary": metrics_summary,
    }

    if args.baseline_metrics:
        baseline_path = Path(args.baseline_metrics)
        baseline_data = load_baseline_metrics(baseline_path)
        if baseline_data:
            comparison = build_comparison(metrics_summary, baseline_data.get("summary", {}))
            summary_payload["comparison"] = comparison

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary_payload, fh, ensure_ascii=False, indent=2)

    print("\nResumen")
    print("-------")
    print(f"Total:     {len(entries)}")
    print(f"Éxitos:    {successes}")
    print(f"Fallidos:  {failures}")
    print(f"Archivo:   {output_path}")
    if metrics_summary.get("metrics"):
        print("\nMétricas RAGAS:")
        for name, value in metrics_summary["metrics"].items():
            print(f"  - {name}: {round(value, 4)}")
    if summary_payload.get("comparison"):
        print("\nComparación con baseline:")
        for metric, data in summary_payload["comparison"].items():
            delta = data.get("delta")
            delta_str = f"{delta:+.4f}" if delta is not None else "N/A"
            print(
                f"  - {metric}: actual={data.get('current')} | baseline={data.get('baseline')} | delta={delta_str}"
            )
    print(f"Resumen de métricas: {summary_path}")


if __name__ == "__main__":
    main()
