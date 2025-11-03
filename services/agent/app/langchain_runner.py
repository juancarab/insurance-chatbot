import os
import sys
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
    """
    Estado del agente en cada paso del grafo.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    contexts: List[Dict[str, Any]]

    language: str
    top_k: int
    enable_web_search: bool
    debug: bool

    __tools__: List[BaseTool]

    debug_steps: List[Dict[str, Any]]


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
    def __init__(self):
        self.llm = self._get_llm()
        self.graph = self._build_graph()

    def _get_llm(self) -> ChatGoogleGenerativeAI:
        """Inicializa el LLM usando la configuración centralizada de la app."""
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

    def _build_tools(
        self,
        *,
        enable_web_search: bool,
    ) -> List[BaseTool]:
        """
        Construye la lista de herramientas disponibles para ESTE request.
        - Siempre: Calculator + Retrieval.
        - Condicional: WebSearch, según flag.
        """
        tools: List[BaseTool] = [CalculatorTool(), retrieval_tool]
        if enable_web_search:
            tools.append(WebSearchTool())
        return tools

    def _build_messages_for_llm(
        self,
        *,
        system_prompt: str,
        language: str,
        chat_history: List[BaseMessage],
        user_input: Optional[str],
        contexts: List[Dict[str, Any]],
    ) -> List[BaseMessage]:
        """
        Construye la lista de mensajes para el LLM:
        1) SystemMessage con idioma y bloque de contexto.
        2) Historial completo (user/assistant).
        3) Mensaje actual del usuario (si llega aparte).
        """

        if contexts:
            context_block = "\n\n## Documentos relevantes:\n" + "\n\n".join(
                [
                    f"### {c.get('title', c.get('file_name', 'N/A'))}\n{c.get('snippet', '')}"
                    for c in contexts
                ]
            )
        else:
            context_block = ""

        sys_msg = SystemMessage(
            content=(system_prompt.format(language=language) + context_block)
        )

        msgs: List[BaseMessage] = [sys_msg]
        msgs.extend(chat_history)

        if user_input:
            msgs.append(HumanMessage(content=user_input))

        return msgs


    def _build_graph(self) -> StateGraph:
        """Construye y compila el grafo LangGraph."""
        workflow = StateGraph(AgentState)

        def call_model_node(state: AgentState) -> Dict[str, Any]:
            system_prompt = (
                "Eres un asistente experto en la industria de seguros, tu objetivo es "
                "proporcionar respuestas precisas y útiles. Responde SIEMPRE en {language}."
                "\n\n"
                "## Estrategia de Respuesta:\n"
                "Sigue estrictamente los siguientes pasos para decidir cómo actuar:\n\n"
                "1.  **Análisis Inicial de la Pregunta:** Primero, evalúa la pregunta del usuario.\n"
                "    -   Si la pregunta es un saludo, una despedida, una pregunta conversacional "
                "simple (ej: 'Hola', '¿quién eres?', 'gracias') o no está relacionada "
                "con seguros, RESPONDE DIRECTAMENTE usando tu conocimiento general SIN usar "
                "ninguna herramienta.\n"
                "    -   Si la pregunta es sobre seguros, procede al paso 2.\n\n"
                "2.  **Decisión de Herramientas:** Basado en la pregunta, decide qué herramienta "
                "es la más adecuada:\n"
                "    -   Para preguntas sobre detalles específicos de pólizas, coberturas, "
                "exclusiones o condiciones de un contrato, usa la herramienta "
                "\"hybrid_opensearch_search\".\n"
                "    -   Para preguntas sobre noticias recientes, tendencias del mercado, "
                "información de compañías de seguros que no esté en una póliza, o eventos "
                "actuales, usa la herramienta \"web_search\".\n"
                "    -   Es posible que necesites usar AMBAS herramientas si la pregunta conecta "
                "una póliza específica con un evento actual (ej: '¿Mi póliza cubre los "
                "daños de la tormenta de anoche?').\n\n"
                "3.  **Lógica de Combinación de Fuentes:**\n"
                "    -   Si solo usaste una herramienta, basa tu respuesta en la información "
                "que esta te proporcionó.\n"
                "    -   **IMPORTANTE:** Si has llamado a \"hybrid_opensearch_search\" y a "
                "\"web_search\" y ambas devuelven información relevante, tu tarea "
                "principal es COMBINAR Y SINTETIZAR los hallazgos. No presentes las "
                "fuentes por separado. Crea una respuesta unificada que conecte la "
                "información de la póliza con la información de la web. "
                "Por ejemplo: Si la póliza cubre 'daños por agua' y las noticias informan "
                "sobre 'inundaciones recientes en la zona X', debes conectar ambas piezas "
                "de información en tu respuesta.\n\n"
                "4.  **Reglas Finales de Respuesta:**\n"
                "    -   Si después de usar las herramientas no encuentras información suficiente "
                "para responder, indícalo claramente diciendo algo como: "
                "'No he podido encontrar información suficiente sobre este tema en los "
                "documentos o en la web.'\n"
                "    -   Al final de tu respuesta, cita SIEMPRE las fuentes utilizadas de manera "
                "clara, mencionando el título del documento o de la noticia."
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

        def call_tool_node(state: AgentState) -> Dict[str, Any]:
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

                tool = next((t for t in state.get("__tools__", []) if t.name == tool_name), None)

                if not tool:
                    tool_outputs.append(
                        ToolMessage(
                            content=f"[ERROR] Tool '{tool_name}' is not available in this request.",
                            tool_call_id=tool_id,
                        )
                    )
                    if state.get("debug"):
                        debug_steps.append(
                            {"tool": tool_name, "input": tool_args, "error": "tool_not_available"}
                        )
                    continue

                output = tool.invoke(tool_args)

                if state.get("debug"):
                    debug_steps.append(
                        {
                            "tool": tool.name,
                            "input": tool_args,
                            "output_preview": str(output)[:500],
                        }
                    )

                if tool.name in ("hybrid_opensearch_search", "web_search"):
                    if isinstance(output, list):
                        for item in output:
                            if hasattr(item, "page_content"):
                                meta = getattr(item, "metadata", {}) or getattr(item, "meta", {}) or {}
                                src = {
                                    "title": meta.get("title")
                                    or meta.get("file_name")
                                    or meta.get("source")
                                    or "Source",
                                    "snippet": getattr(item, "page_content", "") or meta.get("snippet"),
                                    "url": meta.get("url"),
                                    "file_name": meta.get("file_name"),
                                    "page": meta.get("page"),
                                    "chunk_id": meta.get("chunk_id"),
                                    "score": float(meta.get("score", 0.0)) if meta.get("score") is not None else None,
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
                                    "score": float(item.get("score", 0.0)) if item.get("score") is not None else None,
                                }
                                new_structured_sources.append(src)
                    else:
                        print(
                            f"[ToolNode] ADVERTENCIA: {tool.name} no devolvió una lista, no se capturaron fuentes.",
                            file=sys.stderr,
                        )

                tool_outputs.append(ToolMessage(content=str(output), tool_call_id=tool_id))

            prev_contexts = state.get("contexts", [])
            merged_contexts = prev_contexts + new_structured_sources

            out: Dict[str, Any] = {"messages": tool_outputs, "contexts": merged_contexts}
            if state.get("debug"):
                out["debug_steps"] = debug_steps
            return out

        def router(state: AgentState) -> str:
            if state["messages"] and isinstance(state["messages"][-1], AIMessage):
                if state["messages"][-1].tool_calls:
                    return "call_tool"
            return "end"

        workflow.add_node("agent", call_model_node)
        workflow.add_node("call_tool", call_tool_node)
        workflow.set_entry_point("agent")

        workflow.add_conditional_edges("agent", router, {"call_tool": "call_tool", "end": END})
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
        """
        Ejecuta el agente y devuelve la respuesta.
        - messages: [{'role': 'user'|'assistant', 'content': '...'}, ...]
        - contexts: lista inicial de contextos/fragmentos (opcional)
        - top_k: default para retrieval cuando el LLM no especifique k
        - enable_web_search: habilita o no la tool de web
        - debug: activa trazas y chunks en la respuesta
        - language: 'es' por defecto (configurable)
        """
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
        }
        if debug:
            initial_state["debug_steps"] = []

        final_state = self.graph.invoke(initial_state)

        last_ai_message = next(
            (m for m in reversed(final_state.get("messages", [])) if isinstance(m, AIMessage)),
            None,
        )
        answer = last_ai_message.content if last_ai_message else "No se pudo generar una respuesta."

        resp: Dict[str, Any] = {
            "answer": answer,
            "sources": final_state.get("contexts", []),
            "usage": {
                "top_k": top_k,
                "enable_web_search": enable_web_search,
                "language": language,
            },
        }
        if debug:
            resp["debug"] = {
                "steps": final_state.get("debug_steps", []),
                "chunks": final_state.get("contexts", []),
            }

        return resp


_agent_runner_instance = AgentRunner()
run_langchain_agent = _agent_runner_instance.run
