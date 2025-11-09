# services/agent/app/graph.py
from langgraph.graph import END, StateGraph
from .agent_state import AgentState
from .nodes import AgentNodes

def build_graph(nodes: AgentNodes):

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", nodes.call_model_node)
    workflow.add_node("call_tool", nodes.call_tool_node)
    workflow.add_node("reformulate_for_tools", nodes.reformulate_for_tools_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        nodes.router,
        {"call_tool": "reformulate_for_tools", "end": END},
    )
    workflow.add_edge("reformulate_for_tools", "call_tool")
    workflow.add_edge("call_tool", "agent")
    
    return workflow.compile()