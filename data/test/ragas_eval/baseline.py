from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


def load_baseline_metrics(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        print(f"[Aviso] No se encontró archivo de métricas base: {path}")
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def build_comparison(new_metrics: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
    comparison: Dict[str, Dict[str, Optional[float]]] = {}
    new_values = new_metrics.get("metrics", {})
    old_values = baseline.get("summary", {}).get("metrics", {}) or baseline.get("metrics", {})
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