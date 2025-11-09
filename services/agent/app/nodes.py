# services/agent/app/nodes.py
import json
from typing import Any, Dict, List
from .prompts import AGENT_SYSTEM_PROMPT, REFORMULATION_PROMPT

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    BaseMessage,
)
from langchain_google_genai import ChatGoogleGenerativeAI

from .agent_state import AgentState
from .prompts import AGENT_SYSTEM_PROMPT

class AgentNodes:
    MAX_CONTEXT_DOCS = 10
    MAX_SNIPPET_CHARS = 350
    TOOL_LOOP_LIMIT = 3

    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.llm = llm

    def _build_context_block(self, contexts: List[Dict[str, Any]]) -> str:
        if not contexts:
            return ""
        contexts = sorted(
            contexts,
            key=lambda x: x.get("score", 0) or 0,
            reverse=True,
        )[: self.MAX_CONTEXT_DOCS]

        lines = ["\n\n## Documentos relevantes:"]
        for c in contexts:
            title = c.get("title") or c.get("file_name") or "Fuente"
            page = c.get("page")
            snippet = c.get("snippet") or ""
            if len(snippet) > self.MAX_SNIPPET_CHARS:
                snippet = snippet[: self.MAX_SNIPPET_CHARS] + "..."
            if page:
                lines.append(f"- {title} (pág. {page}): {snippet}")
            else:
                lines.append(f"- {title}: {snippet}")
        return "\n".join(lines)

    def _format_history_for_reformulation(self, messages: List[BaseMessage]) -> str:
        formatted = []
        for msg in messages:
            role = "Usuario" if msg.type == "user" else "Asistente"
            content = (msg.content or "")[:500]
            formatted.append(f"{role}: {content}")
        return "\n".join(formatted[-10:])

    def call_model_node(self, state: AgentState) -> Dict[str, Any]:
        system_prompt = AGENT_SYSTEM_PROMPT.format(language=state.get("language", "es"))
        context_block = self._build_context_block(state.get("contexts", []))
        sys_msg = SystemMessage(content=system_prompt + context_block)

        llm_with_tools = self.llm.bind_tools(state.get("__tools__", []))
        msgs: List[BaseMessage] = [sys_msg]
        msgs.extend(state["messages"])

        response = llm_with_tools.invoke(msgs)
        return {"messages": [response]}

    def reformulate_for_tools_node(self, state: AgentState) -> Dict[str, Any]:
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return {}

        if state.get("tool_iterations", 0) >= self.TOOL_LOOP_LIMIT:
            return {}

        reformulation_targets = {"hybrid_opensearch_search", "web_search"}
        history = state["messages"][:-1]

        queries_to_fix = {}
        original_tool_calls = {}

        for tool_call in last_message.tool_calls:
            tool_id = tool_call.get("id")
            original_tool_calls[tool_id] = tool_call
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {}) or {}
            if tool_name in reformulation_targets and "query" in tool_args:
                queries_to_fix[tool_id] = (tool_args["query"] or "")

        if not queries_to_fix or not history:
            return {}

        queries_str = "\n".join(
            f'- ID_LLAMADA: "{tool_id}", CONSULTA_ORIGINAL: "{query}"'
            for tool_id, query in queries_to_fix.items()
        )

        reformulation_prompt = REFORMULATION_PROMPT.format(
            history=self._format_history_for_reformulation(history),
            queries=queries_str,
        )

        try:
            response = self.llm.invoke([HumanMessage(content=reformulation_prompt)])
            content = response.content.strip()

            reformulated_queries = {}
            try:
                reformulated_queries = json.loads(content)
            except Exception:
                if "```json" in content:
                    try:
                        part = content.split("```json", 1)[1].split("```", 1)[0]
                        reformulated_queries = json.loads(part.strip())
                    except Exception:
                        reformulated_queries = {}

            new_tool_calls = []
            last_user_msg = ""
            for m in reversed(state["messages"]):
                if m.type == "user":
                    last_user_msg = (m.content or "").strip()
                    break

            for tool_id, original_call in original_tool_calls.items():
                if tool_id in reformulated_queries:
                    candidate = (reformulated_queries.get(tool_id) or "").strip()
                    original_q = (queries_to_fix.get(tool_id) or "").strip()
                    new_query = candidate or original_q or last_user_msg or "información de la póliza del usuario"
                else:
                    original_q = (queries_to_fix.get(tool_id) or "").strip()
                    new_query = original_q or last_user_msg or "información de la póliza del usuario"

                new_args = (original_call.get("args") or {}).copy()
                new_args["query"] = new_query
                new_call_dict = original_call.copy()
                new_call_dict["args"] = new_args
                new_tool_calls.append(new_call_dict)

        except Exception as e:
            new_tool_calls = last_message.tool_calls
            if state.get("debug"):
                state.get("debug_steps", []).append(
                    {"step": "reformulate_batch_failed", "error": str(e)}
                )

        new_ai_msg = AIMessage(
            content=last_message.content,
            tool_calls=new_tool_calls,
        )
        return {"messages": state["messages"][:-1] + [new_ai_msg]}

    def _serialize_tool_output_for_llm(self, output: Any) -> str:
        if isinstance(output, list):
            items = output
        else:
            items = [output]

        simplified = []
        for item in items:
            if hasattr(item, "page_content"):
                simplified.append(
                    {
                        "text": getattr(item, "page_content", ""),
                        "metadata": getattr(item, "metadata", {}),
                    }
                )
            elif isinstance(item, dict):
                simplified.append(item)
            else:
                simplified.append({"value": str(item)})

        s = json.dumps(simplified, ensure_ascii=False)
        return s[:4000]

    def call_tool_node(self, state: AgentState) -> Dict[str, Any]:
        tool_calls = state["messages"][-1].tool_calls
        tool_outputs: List[ToolMessage] = []
        new_structured_sources: List[Dict[str, Any]] = []
        debug_steps: List[Dict[str, Any]] = state.get("debug_steps", [])

        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {}) or {}
            tool_id = tool_call.get("id")

            if tool_name == "hybrid_opensearch_search":
                if "k" not in tool_args and "top_k" not in tool_args:
                    tool_args["k"] = state.get("top_k", 4)

            requested = (tool_name or "").lower()
            tool = next(
                (t for t in state.get("__tools__", []) if t.name.lower() == requested),
                None,
            )
            if not tool:
                tool_outputs.append(
                    ToolMessage(
                        content=f"[ERROR] Tool '{tool_name}' is not available.",
                        tool_call_id=tool_id,
                    )
                )
                if state.get("debug"):
                    debug_steps.append(
                        {
                            "step": "call_tool",
                            "tool": tool_name,
                            "input": tool_args,
                            "error": "tool_not_available",
                        }
                    )
                continue

            output = tool.invoke(tool_args)

            if state.get("debug"):
                preview = str(output)
                if len(preview) > 500:
                    preview = preview[:500] + "..."
                debug_steps.append(
                    {
                        "step": "call_tool",
                        "tool": tool.name,
                        "input": tool_args,
                        "output_preview": preview,
                    }
                )

            if tool.name in ("hybrid_opensearch_search", "web_search"):
                items = output if isinstance(output, list) else [output]
                for item in items:
                    if hasattr(item, "page_content"):
                        meta = getattr(item, "metadata", {}) or {}
                        src = {
                            "title": meta.get("title")
                            or meta.get("file_name")
                            or meta.get("source")
                            or "Source",
                            "snippet": getattr(item, "page_content", "")
                            or meta.get("snippet"),
                            "url": meta.get("url"),
                            "file_name": meta.get("file_name"),
                            "page": meta.get("page"),
                            "chunk_id": meta.get("chunk_id"),
                            "score": float(meta.get("score", 0.0))
                            if meta.get("score") is not None
                            else None,
                        }
                        new_structured_sources.append(src)
                    elif isinstance(item, dict):
                        src = {
                            "title": item.get("title")
                            or item.get("file_name")
                            or item.get("source")
                            or "Source",
                            "snippet": item.get("snippet"),
                            "url": item.get("url"),
                            "file_name": item.get("file_name"),
                            "page": item.get("page"),
                            "chunk_id": item.get("chunk_id"),
                            "score": float(item.get("score", 0.0))
                            if item.get("score") is not None
                            else None,
                        }
                        new_structured_sources.append(src)

            tool_outputs.append(
                ToolMessage(
                    content=self._serialize_tool_output_for_llm(output),
                    tool_call_id=tool_id,
                )
            )

        prev_contexts = state.get("contexts", [])
        merged_contexts = prev_contexts + new_structured_sources
        merged_contexts = sorted(
            merged_contexts,
            key=lambda x: x.get("score", 0) or 0,
            reverse=True,
        )[: self.MAX_CONTEXT_DOCS]

        out: Dict[str, Any] = {
            "messages": tool_outputs,
            "contexts": merged_contexts,
            "tool_iterations": state.get("tool_iterations", 0) + 1,
        }
        if state.get("debug"):
            out["debug_steps"] = debug_steps
        return out

    def router(self, state: AgentState) -> str:
        if not state.get("messages"):
            return "end"

        last = state["messages"][-1]

        if state.get("tool_iterations", 0) >= self.TOOL_LOOP_LIMIT:
            return "end"

        if isinstance(last, AIMessage) and last.tool_calls:
            return "call_tool"

        return "end"