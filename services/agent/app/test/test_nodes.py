from typing import Any, Type
from pathlib import Path
import sys
import asyncio

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = None
for parent in THIS_FILE.parents:
    if (parent / "services").is_dir():
        PROJECT_ROOT = parent
        break

if PROJECT_ROOT is None:
    raise RuntimeError("Direct root not found.")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from services.agent.app.nodes import AgentNodes
from services.agent.app.agent_state import AgentState


# --------------------------
# Dummies
# --------------------------

class DummyLLM:

    def __init__(self, response_content: str = ""):
        self._response_content = response_content
        self.last_messages = None

    async def ainvoke(self, messages):
        self.last_messages = messages

        class Resp:
            def __init__(self, content: str):
                self.content = content

        return Resp(self._response_content)


class DummyToolInput(BaseModel):
    query: str = Field(..., description="Dummy query")


class DummyTool(BaseTool):
    name: str = "hybrid_opensearch_search"
    description: str = "Dummy hybrid search tool for tests."
    args_schema: Type[BaseModel] = DummyToolInput

    def _run(self, query: str, run_manager=None):
        return [
            {
                "title": "Doc 1",
                "snippet": f"Snippet para {query}",
                "score": 0.9,
                "page": 1,
            },
            {
                "title": "Doc 2",
                "snippet": f"Otro snippet para {query}",
                "score": 0.8,
                "page": 2,
            },
        ]

    async def _arun(self, query: str, run_manager=None):
        return self._run(query, run_manager=run_manager)


# --------------------------
# Router
# --------------------------

def test_router_calls_tool_when_ai_has_tool_calls():
    nodes = AgentNodes(llm=DummyLLM())
    state: AgentState = {
        "messages": [
            HumanMessage(content="Hola"),
            AIMessage(
                content="Voy a usar herramientas",
                tool_calls=[
                    {
                        "id": "call-1",
                        "name": "hybrid_opensearch_search",
                        "args": {"query": "algo"},
                    }
                ],
            ),
        ],
        "tool_iterations": 0,
    }

    decision = nodes.router(state)
    assert decision == "call_tool"


def test_router_ends_when_no_tool_calls():
    nodes = AgentNodes(llm=DummyLLM())
    state: AgentState = {
        "messages": [
            HumanMessage(content="Hola"),
            AIMessage(content="Sin herramientas"),
        ],
        "tool_iterations": 0,
    }

    decision = nodes.router(state)
    assert decision == "end"


def test_router_ends_when_tool_loop_limit_reached():
    nodes = AgentNodes(llm=DummyLLM())
    state: AgentState = {
        "messages": [
            HumanMessage(content="Hola"),
            AIMessage(
                content="Con herramientas",
                tool_calls=[
                    {
                        "id": "call-1",
                        "name": "hybrid_opensearch_search",
                        "args": {"query": "algo"},
                    }
                ],
            ),
        ],
        "tool_iterations": nodes.TOOL_LOOP_LIMIT,
    }

    decision = nodes.router(state)
    assert decision == "end"


# --------------------------
# reformulate_for_tools_node
# --------------------------

def test_reformulate_for_tools_node_uses_llm_json():
    response_json = '{"call-1": "cobertura de hospitalización por accidente"}'
    llm = DummyLLM(response_content=response_json)
    nodes = AgentNodes(llm=llm)

    last_ai = AIMessage(
        content="Te ayudo con tu póliza",
        tool_calls=[
            {
                "id": "call-1",
                "name": "hybrid_opensearch_search",
                "args": {"query": "esta cirugía"},
            }
        ],
    )

    state: AgentState = {
        "messages": [
            HumanMessage(content="¿Mi cirugía está cubierta?"),
            last_ai,
        ],
        "tool_iterations": 0,
        "debug": False,
    }

    out = asyncio.run(nodes.reformulate_for_tools_node(state))

    assert "messages" in out
    assert len(out["messages"]) == 2
    new_last = out["messages"][-1]
    assert isinstance(new_last, AIMessage)
    assert len(new_last.tool_calls) == 1

    new_args = new_last.tool_calls[0]["args"]
    assert new_args["query"] == "cobertura de hospitalización por accidente"


def test_reformulate_for_tools_node_falls_back_on_invalid_json():
    """
    Si el LLM devuelve algo que NO es JSON, se debe conservar
    la query original.
    """
    llm = DummyLLM(response_content="esto no es json")
    nodes = AgentNodes(llm=llm)

    last_ai = AIMessage(
        content="Respuesta ambigua",
        tool_calls=[
            {
                "id": "call-1",
                "name": "hybrid_opensearch_search",
                "args": {"query": "esta cirugía"},
            }
        ],
    )

    state: AgentState = {
        "messages": [
            HumanMessage(content="¿Mi cirugía está cubierta?"),
            last_ai,
        ],
        "tool_iterations": 0,
        "debug": False,
    }

    out = asyncio.run(nodes.reformulate_for_tools_node(state))

    assert "messages" in out
    new_last = out["messages"][-1]
    new_args = new_last.tool_calls[0]["args"]
    assert new_args["query"] == "esta cirugía"


# --------------------------
# call_tool_node
# --------------------------

def test_call_tool_node_executes_tool_and_updates_contexts():
    nodes = AgentNodes(llm=DummyLLM())

    tool = DummyTool()
    last_ai = AIMessage(
        content="Voy a buscar en las pólizas",
        tool_calls=[
            {
                "id": "call-1",
                "name": "hybrid_opensearch_search",
                "args": {"query": "cobertura de maternidad"},
            }
        ],
    )

    prev_contexts = [
        {
            "title": "Contexto viejo",
            "snippet": "Texto antiguo",
            "score": 0.5,
        }
    ]

    state: AgentState = {
        "messages": [last_ai],
        "contexts": prev_contexts,
        "__tools__": [tool],
        "tool_iterations": 0,
        "debug": False,
        "top_k": 3,
    }

    out = asyncio.run(nodes.call_tool_node(state))

    assert "messages" in out
    tool_msgs = out["messages"]
    assert len(tool_msgs) == 1
    assert isinstance(tool_msgs[0].content, str)
    assert tool_msgs[0].tool_call_id == "call-1"

    assert "contexts" in out
    contexts = out["contexts"]
    assert len(contexts) >= 2

    titles = {c.get("title") for c in contexts}
    assert "Contexto viejo" in titles
    assert "Doc 1" in titles or "Doc 2" in titles

    assert out["tool_iterations"] == 1
