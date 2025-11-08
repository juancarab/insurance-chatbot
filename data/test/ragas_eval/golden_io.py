from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


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