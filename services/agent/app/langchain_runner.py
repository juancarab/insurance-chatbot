import time
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from .config import get_settings
from .agent_state import AgentState
from .tools import build_tools
from .nodes import AgentNodes
from .graph import build_graph

class AgentRunner:
    def __init__(self):
        settings = get_settings()
        self.llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
            temperature=settings.gemini_temperature or 0.0,
            top_p=settings.gemini_top_p or 0.95,
            max_output_tokens=settings.gemini_max_output_tokens or 1024,
        )
        self.nodes = AgentNodes(self.llm)
        self.graph = build_graph(self.nodes)

    async def run(
        self,
        *,
        messages: List[Dict[str, Any]],
        contexts: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 4,
        enable_web_search: bool = False,
        debug: bool = False,
        language: str = "es",
    ) -> Dict[str, Any]:
        
        start_total_time = time.perf_counter()

        langchain_messages = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif role in ("assistant", "ai"):
                langchain_messages.append(AIMessage(content=content))

        tools_for_request = build_tools(enable_web_search=enable_web_search)
        initial_state: AgentState = {
            "messages": langchain_messages,
            "contexts": contexts or [],
            "language": language,
            "top_k": top_k,
            "enable_web_search": enable_web_search,
            "debug": debug,
            "__tools__": tools_for_request,
            "tool_iterations": 0,
        }
        if debug:
            initial_state["debug_steps"] = []

        final_state = await self.graph.ainvoke(initial_state, config={"recursion_limit": 10})
        
        total_duration_ms = (time.perf_counter() - start_total_time) * 1000

        last_ai = next(
            (m for m in reversed(final_state.get("messages", [])) if isinstance(m, AIMessage)),
            None,
        )
        answer = last_ai.content if last_ai else "A response could not be generated."

        final_sources = final_state.get("contexts", []) or []
        final_sources = sorted(
            final_sources,
            key=lambda x: x.get("rerank_score", x.get("score", 0) or 0),
            reverse=True,
        )

        resp: Dict[str, Any] = {
            "answer": answer,
            "sources": final_sources,
            "usage": {
                "top_k": top_k,
                "enable_web_search": enable_web_search,
                "language": language,
            },
        }
        if debug:
            resp["debug"] = {
                "total_duration_ms": round(total_duration_ms, 2),
                "steps": final_state.get("debug_steps", []),
                "chunks": final_sources,
                "tool_iterations": final_state.get("tool_iterations", 0),
            }
        return resp

_agent_runner_instance = AgentRunner()
run_langchain_agent = _agent_runner_instance.run