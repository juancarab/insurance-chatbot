from typing import Any, Dict, List


def run_langchain_agent(
    *,
    messages: List[Dict[str, Any]],
    contexts: List[Dict[str, Any]],
    top_k: int,
    enable_web_search: bool,
    debug: bool,
    language: str,
) -> Dict[str, Any]:
    """Runner mínimo que responde con un placeholder amistoso y datos de depuración.

    Sustituye esta implementación con tu orquestación real de LangChain/LangGraph.
    """

    user_message = next(
        (msg for msg in reversed(messages) if msg.get("role") == "user"),
        None,
    )

    answer = (
        "⚠️ Demo sin agente configurado."
        "\nConfigura el runner de LangChain/LangGraph para obtener respuestas reales"
        f" en {language}."
    )

    return {
        "answer": answer,
        "sources": contexts or [],
        "debug": {
            "top_k": top_k,
            "enable_web_search": enable_web_search,
            "debug_mode": debug,
            "messages_count": len(messages),
        },
    }
