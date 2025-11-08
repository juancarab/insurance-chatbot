from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import requests


def run_entry(entry: Dict[str, Any], base_url: str, timeout: float) -> Dict[str, Any]:
    payload = entry.get("payload", {})
    start = time.perf_counter()
    try:
        resp = requests.post(base_url, json=payload, timeout=timeout)
        elapsed = time.perf_counter() - start
        resp.raise_for_status()
        body = resp.json()
        return {
            "id": entry.get("id"),
            "category": entry.get("category"),
            "description": entry.get("description"),
            "status_code": resp.status_code,
            "elapsed_ms": round(elapsed * 1000, 2),
            "answer": body.get("answer"),
            "sources": body.get("sources", []),
            "usage": body.get("usage", {}),
            "debug": body.get("debug"),
            "raw_response": body,
            "error": None,
        }
    except Exception as exc:
        elapsed = time.perf_counter() - start
        status_code = None
        response_body = None
        if hasattr(exc, "response") and getattr(exc, "response") is not None:
            try:
                status_code = exc.response.status_code
                response_body = exc.response.text  
            except Exception:
                pass
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