from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ejecuta el golden set del chatbot")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000/chat",
        help="Endpoint completo del backend FastAPI (/chat)",
    )
    parser.add_argument(
        "--golden-set",
        default="data/test/ragas_eval/golden_set/golden_set.json",
        help="Ruta al archivo JSON con el golden set",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Archivo JSONL de salida (por defecto results/golden_<timestamp>.jsonl)",
    )
    parser.add_argument(
        "--baseline-metrics",
        default=None,
        help="Ruta a un archivo de métricas previo para comparar",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Timeout por petición al backend",
    )
    parser.add_argument(
        "--ragas-model",
        default="gemini-2.5-flash",
        help="Modelo de Gemini que usará RAGAS",
    )
    parser.add_argument(
        "--gemini-api-key",
        default=None,
        help="API key de Gemini (si no, usa GEMINI_API_KEY)",
    )
    parser.add_argument(
        "--debug-llm",
        action="store_true",
        help="Imprime la salida cruda del LLM que usa RAGAS",
    )
    return parser.parse_args()