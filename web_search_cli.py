#!/usr/bin/env python
import sys, json
from services.agent.app.tools.web_search.web_search import WebSearchTool

def call_tool(tool, query: str):
    # Probar patrones comunes de LangChain Tools
    for fn in ("run", "__call__", "invoke"):
        if hasattr(tool, fn):
            try:
                if fn == "invoke":
                    # Algunos Tools aceptan dict; otros string
                    try:
                        return getattr(tool, fn)({"query": query})
                    except:
                        return getattr(tool, fn)(query)
                return getattr(tool, fn)(query)
            except Exception as e:
                print(f"[debug] método {fn} falló: {e}", file=sys.stderr)
    raise RuntimeError("No encontré método compatible (run, __call__, invoke).")

def main():
    if len(sys.argv) < 2:
        print("Uso: python web_search_cli.py \"tu consulta...\"")
        sys.exit(1)
    query = " ".join(sys.argv[1:])
    tool = WebSearchTool()
    resp = call_tool(tool, query)
    if isinstance(resp, (dict, list)):
        print(json.dumps(resp, ensure_ascii=False, indent=2))
    else:
        print(resp)

if __name__ == "__main__":
    main()
