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
    """Minimal runner that responds with a friendly placeholder and debugging data.

    Replace this implementation with your actual LangChain/LangGraph orchestration.
    """

    user_message = next(
        (msg for msg in reversed(messages) if msg.get("role") == "user"),
        None,
    )

    answer = (
        "⚠️ Demo without agent configured."
        "\nConfigure the LangChain/LangGraph runner to obtain real answers"
        f" in {language}."
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
