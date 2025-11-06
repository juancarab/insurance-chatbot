import os
import sys
import json
from typing import Annotated, Any, Dict, List, Optional, Type, TypedDict

from pydantic import BaseModel, Field

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
    SystemMessage,
)
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from .tools.web_search.web_search import WebSearchTool
from .tools.retrieval.haystack_opensearch_tool import retrieval_tool
from .config import get_settings


class AgentState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    contexts: List[Dict[str, Any]]
    language: str
    top_k: int
    enable_web_search: bool
    debug: bool
    __tools__: List[BaseTool]
    debug_steps: List[Dict[str, Any]]
    tool_iterations: int 


class CalculatorInput(BaseModel):
    expression: str = Field(
        description="Mathematical expression to be evaluated (e.g., '2+2*3')."
    )


class CalculatorTool(BaseTool):
    name: str = "calculator"
    description: str = "Useful when you need to perform mathematical calculations."
    args_schema: Type[BaseModel] = CalculatorInput

    def _run(self, expression: str) -> str:
        try:
            return str(eval(expression, {"__builtins__": None}, {}))
        except Exception as e:
            return f"Error evaluating expression: {e}"

    async def _arun(self, expression: str) -> str:
        return self._run(expression)


class AgentRunner:
    MAX_CONTEXT_DOCS = 10
    MAX_SNIPPET_CHARS = 350
    TOOL_LOOP_LIMIT = 3

    def __init__(self):
        self.llm = self._get_llm()
        self.graph = self._build_graph()

    def _get_llm(self) -> ChatGoogleGenerativeAI:
        settings = get_settings()
        if not settings.gemini_api_key or not settings.gemini_model:
            raise ValueError("GEMINI_API_KEY and GEMINI_MODEL must be configured.")
        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
            temperature=settings.gemini_temperature or 0.2,
            top_p=settings.gemini_top_p or 0.95,
            max_output_tokens=settings.gemini_max_output_tokens or 1024,
        )

    def _build_tools(self, *, enable_web_search: bool) -> List[BaseTool]:
        tools: List[BaseTool] = [CalculatorTool(), retrieval_tool]
        if enable_web_search:
            tools.append(WebSearchTool())
        return tools

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

    def _build_messages_for_llm(
        self,
        *,
        system_prompt: str,
        language: str,
        chat_history: List[BaseMessage],
        user_input: Optional[str],
        contexts: List[Dict[str, Any]],
    ) -> List[BaseMessage]:
        context_block = self._build_context_block(contexts)
        sys_msg = SystemMessage(content=(system_prompt.format(language=language) + context_block))
        msgs: List[BaseMessage] = [sys_msg]
        msgs.extend(chat_history)
        if user_input:
            msgs.append(HumanMessage(content=user_input))
        return msgs

    def call_model_node(self, state: AgentState) -> Dict[str, Any]:
        system_prompt = (
            "Eres un asistente experto en la industria de seguros. "
            "Tu objetivo es responder con precisión usando primero las fuentes internas. "
            "Responde SIEMPRE en {language}.\n\n"
            "## Política de uso de herramientas\n"
            "1. PRIORIDAD ABSOLUTA: Para cualquier pregunta sobre pólizas, coberturas, exclusiones, "
            "definiciones, condiciones generales/particulares o leyes mencionadas en documentos internos, "
            "DEBES usar primero `hybrid_opensearch_search`.\n"
            "2. NO uses `web_search` si todavía no has intentado `hybrid_opensearch_search` para esa duda.\n"
            "3. SOLO puedes usar `web_search` en alguno de estos dos casos:\n"
            "   3.a) La pregunta del usuario mezcla un evento externo + una póliza "
            "        (ej: 'hubo un terremoto anoche, ¿esto lo cubre mi seguro?'). Ahí usas AMBAS: "
            "        primero `hybrid_opensearch_search` para la parte de la póliza y además `web_search` para el evento.\n"
            "   3.b) Ya ejecutaste `hybrid_opensearch_search` en este mismo turno/conversación y "
            "        NO te devolvió documentos útiles (el contexto está vacío o casi vacío). "
            "        En ese caso puedes intentar `web_search` como respaldo.\n"
            "4. Si usas ambas herramientas, debes COMBINAR la respuesta: primero explica lo que pasó (web) "
            "   y luego di si la póliza lo cubre (índice interno).\n"
            "5. Si después de consultar las herramientas no hay información suficiente, dilo claramente y pide más datos "
            "(por ejemplo, el nombre exacto del plan o las condiciones particulares).\n"
            "6. Cita SIEMPRE las fuentes que estás usando.\n"
            "7. Si ya tienes suficiente contexto de llamadas anteriores (el sistema te muestra 'Documentos relevantes'), "
            "   responde con eso y NO vuelvas a llamar herramientas innecesariamente.\n"
        )

        tools = state.get("__tools__", [])
        llm_with_tools = self.llm.bind_tools(tools)

        messages_for_llm = self._build_messages_for_llm(
            system_prompt=system_prompt,
            language=state.get("language", "es"),
            chat_history=state["messages"],
            user_input=None,
            contexts=state.get("contexts", []),
        )

        response = llm_with_tools.invoke(messages_for_llm)
        return {"messages": [response]}


    def _format_history_for_reformulation(self, messages: List[BaseMessage]) -> str:
        formatted = []
        for msg in messages:
            role = "Usuario" if isinstance(msg, HumanMessage) else "Asistente"
            content = (msg.content or "")[:500]
            formatted.append(f"{role}: {content}")
        return "\n".join(formatted[-10:])

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
                queries_to_fix[tool_id] = tool_args["query"]

        if not queries_to_fix or not history:
            return {}

        queries_str = "\n".join(
            f'- ID_LLAMADA: "{tool_id}", CONSULTA_ORIGINAL: "{query}"'
            for tool_id, query in queries_to_fix.items()
        )

        reformulation_prompt = f"""Dada la siguiente conversación y una LISTA de consultas de búsqueda,
reescribe CADA consulta para que sea una pregunta independiente y completa.
Si una consulta ya es independiente, devuélvela tal cual.

Responde ÚNICAMENTE con un JSON válido donde cada clave sea el ID_LLAMADA
y el valor la consulta reescrita.

**Historial:**
{self._format_history_for_reformulation(history)}

**Consultas a arreglar:**
{queries_str}

JSON:
"""

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
            for tool_id, original_call in original_tool_calls.items():
                if tool_id in reformulated_queries:
                    new_query = reformulated_queries[tool_id] or queries_to_fix[tool_id]
                    new_args = (original_call.get("args") or {}).copy()
                    new_args["query"] = new_query
                    new_call_dict = original_call.copy()
                    new_call_dict["args"] = new_args
                    new_tool_calls.append(new_call_dict)
                else:
                    new_tool_calls.append(original_call)

        except Exception as e:
            new_tool_calls = last_message.tool_calls
            if state.get("debug"):
                state.get("debug_steps", []).append(
                    {"step": "reformulate_batch_failed", "error": str(e)}
                )

        new_ai_message = AIMessage(
            content=last_message.content,
            tool_calls=new_tool_calls,
            id=last_message.id,
        )
        updated_messages = state["messages"][:-1] + [new_ai_message]
        return {"messages": updated_messages}

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

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", self.call_model_node)
        workflow.add_node("call_tool", self.call_tool_node)
        workflow.add_node("reformulate_for_tools", self.reformulate_for_tools_node)
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            self.router,
            {"call_tool": "reformulate_for_tools", "end": END},
        )
        workflow.add_edge("reformulate_for_tools", "call_tool")
        workflow.add_edge("call_tool", "agent")
        return workflow.compile()

    def run(
        self,
        *,
        messages: List[Dict[str, Any]],
        contexts: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 4,
        enable_web_search: bool = False,
        debug: bool = False,
        language: str = "es",
    ) -> Dict[str, Any]:
        langchain_messages: List[BaseMessage] = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif role in ("assistant", "ai"):
                langchain_messages.append(AIMessage(content=content))

        tools_for_request = self._build_tools(enable_web_search=enable_web_search)
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

        config = {"recursion_limit": 10} 
        final_state = self.graph.invoke(initial_state, config=config)

        last_ai_message = next(
            (m for m in reversed(final_state.get("messages", [])) if isinstance(m, AIMessage)),
            None,
        )
        answer = last_ai_message.content if last_ai_message else "No se pudo generar una respuesta."

        final_sources = final_state.get("contexts", []) or []
        final_sources = sorted(
            final_sources,
            key=lambda x: x.get("score", 0) or 0,
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
                "steps": final_state.get("debug_steps", []),
                "chunks": final_sources,
                "tool_iterations": final_state.get("tool_iterations", 0),
            }
        return resp


_agent_runner_instance = AgentRunner()
run_langchain_agent = _agent_runner_instance.run