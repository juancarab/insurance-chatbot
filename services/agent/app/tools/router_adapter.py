# services/agent/app/tools/router_adapter.py
from __future__ import annotations
from typing import List, Union
from services.agent.app.tools.find_relevant_policies import FindRelevantPoliciesTool

def pick_candidate_policies(query: str, top_k: int = 5) -> List[str]:
    """
    Adapter que invoca la tool de Laura y garantiza:
      - lista de strings (file_name)
      - sin duplicados
      - tamaño <= top_k
    Cualquier error interno devuelve [] para permitir fallback sin romper
    """
    try:
        out: Union[List[str], str] = FindRelevantPoliciesTool()(query, top_k)
        if isinstance(out, str):  # por si  devueve "error: ..."
            return []
        if not out:
            return []
        seen, ordered = set(), []
        for fn in out:
            if isinstance(fn, str) and fn and fn not in seen:
                seen.add(fn)
                ordered.append(fn)
            if len(ordered) >= top_k:
                break
        return ordered
    except Exception:
        # failopen: el router puede decidir buscar sin filtros
        return []
