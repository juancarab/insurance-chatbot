from typing import Any, Dict, List
from pathlib import Path
import sys

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = None
for parent in THIS_FILE.parents:
    if (parent / "services").is_dir():
        PROJECT_ROOT = parent
        break

if PROJECT_ROOT is None:
    raise RuntimeError("File directory not found.")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import pytest

from services.agent.app import graph
from services.agent.app.nodes import AgentNodes
from services.agent.app.agent_state import AgentState


class DummyStateGraph:

    def __init__(self, state_type):
        self.state_type = state_type
        self.nodes = {}
        self.entry_point = None
        self.conditional_edges = []
        self.edges = []

    def add_node(self, name: str, func: Any) -> None:
        self.nodes[name] = func

    def set_entry_point(self, name: str) -> None:
        self.entry_point = name

    def add_conditional_edges(
        self,
        node_name: str,
        router_fn: Any,
        mapping: Dict[str, str],
    ) -> None:
        self.conditional_edges.append(
            {"node": node_name, "router": router_fn, "mapping": mapping}
        )

    def add_edge(self, from_node: str, to_node: str) -> None:
        self.edges.append((from_node, to_node))

    def compile(self) -> str:
        return "DUMMY_COMPILED_GRAPH"


class DummyLLM:
    pass


def test_build_graph_structure(monkeypatch):
    dummy_end = object()
    created_graphs: List[DummyStateGraph] = []

    def fake_state_graph(state_type):
        g = DummyStateGraph(state_type)
        created_graphs.append(g)
        return g

    monkeypatch.setattr(graph, "StateGraph", fake_state_graph)
    monkeypatch.setattr(graph, "END", dummy_end)

    nodes = AgentNodes(llm=DummyLLM())

    result = graph.build_graph(nodes)

    assert result == "DUMMY_COMPILED_GRAPH"
    assert len(created_graphs) == 1
    g = created_graphs[0]

    assert g.state_type is AgentState

    assert set(g.nodes.keys()) == {"agent", "call_tool", "reformulate_for_tools"}
    assert g.nodes["agent"] == nodes.call_model_node
    assert g.nodes["call_tool"] == nodes.call_tool_node
    assert g.nodes["reformulate_for_tools"] == nodes.reformulate_for_tools_node

    assert g.entry_point == "agent"

    assert len(g.conditional_edges) == 1
    cond = g.conditional_edges[0]
    assert cond["node"] == "agent"
    assert cond["router"] == nodes.router
    # mapping "call_tool" -> "reformulate_for_tools", "end" -> END
    assert cond["mapping"]["call_tool"] == "reformulate_for_tools"
    assert cond["mapping"]["end"] is dummy_end

    assert ("reformulate_for_tools", "call_tool") in g.edges
    assert ("call_tool", "agent") in g.edges
