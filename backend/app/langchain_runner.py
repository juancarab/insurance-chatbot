import os
from typing import Annotated, Any, Dict, List, Optional, Type, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from .config import get_settings


class AgentState(TypedDict):
    """It represents the state of the agent at each step of the graph."""
    messages: Annotated[list[BaseMessage], add_messages]
    contexts: List[Dict[str, Any]]

class CalculatorInput(BaseModel):
    expression: str = Field(description="Mathematical expression to be evaluated (e.g., '2+2*3').")

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

AGENT_TOOLS: List[BaseTool] = [CalculatorTool()]

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

    def _build_graph(self) -> StateGraph:
        """Construye y compila el grafo LangGraph."""
        workflow = StateGraph(AgentState)
        llm_with_tools = self.llm.bind_tools(AGENT_TOOLS)

        def call_model_node(state: AgentState) -> Dict[str, Any]:
            system_prompt = (
                "You are a helpful and precise insurance assistant. Your main task is to answer questions "
                "about insurance policies based on the policy documents provided below. "
                "If the question can be answered directly from the information, do so concisely. "
                "If the user asks for a mathematical operation, use the 'calculator' tool. "
                "If you cannot answer based on the context or your tools, state that you do not have that information. "
                "You must always respond in English."
            )
            context_block = "\n\n## Policy Documents:\n" + "\n\n".join(
                f"### {c.get('title', 'N/A')}\n{c.get('snippet', '')}" for c in state["contexts"]
            )
            
            messages_with_context = [HumanMessage(content=system_prompt + context_block)] + state["messages"]
            
            response = llm_with_tools.invoke(messages_with_context)
            return {"messages": [response]}

        def call_tool_node(state: AgentState) -> Dict[str, Any]:
            tool_calls = state["messages"][-1].tool_calls
            tool_outputs = []
            for tool_call in tool_calls:
                tool = next((t for t in AGENT_TOOLS if t.name == tool_call["name"]), None)
                if tool:
                    output = tool.invoke(tool_call["args"])
                    tool_outputs.append(ToolMessage(content=str(output), tool_call_id=tool_call["id"]))
            return {"messages": tool_outputs}

        def router(state: AgentState) -> str:
            if state["messages"][-1].tool_calls:
                return "call_tool"
            return "end"

        workflow.add_node("agent", call_model_node)
        workflow.add_node("tool", call_tool_node)
        workflow.set_entry_point("agent")
        
        workflow.add_conditional_edges(
            "agent", router, {"call_tool": "tool", "end": END}
        )
        workflow.add_edge("tool", "agent")
        
        return workflow.compile()

    def run(self, *, messages: List[Dict[str, Any]], contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Executes the agent and returns the response in the expected format."""
        langchain_messages: List[BaseMessage] = [
            HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"])
            for msg in messages
        ]
        
        initial_state: AgentState = {"messages": langchain_messages, "contexts": contexts}
        final_state = self.graph.invoke(initial_state)
        
        last_ai_message = next((m for m in reversed(final_state["messages"]) if isinstance(m, AIMessage)), None)
        answer = last_ai_message.content if last_ai_message else "A response could not be generated."

        return {
            "answer": answer,
            "sources": contexts, 
        }

_agent_runner_instance = AgentRunner()
run_langchain_agent = _agent_runner_instance.run