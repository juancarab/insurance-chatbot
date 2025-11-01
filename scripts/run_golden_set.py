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
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import requests


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
        "--timeout",
        type=float,
        default=60.0,
        help="Timeout por petición en segundos",
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

    with output_path.open("w", encoding="utf-8") as fh:
        for idx, entry in enumerate(entries, start=1):
            result = run_entry(entry, args.base_url, args.timeout)
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

    print("\nResumen")
    print("-------")
    print(f"Total:     {len(entries)}")
    print(f"Éxitos:    {successes}")
    print(f"Fallidos:  {failures}")
    print(f"Archivo:   {output_path}")


if __name__ == "__main__":
    main()
