# services/agent/app/agent_state.py
from typing import Annotated, Any, Dict, List, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    contexts: List[Dict[str, Any]]
    language: str
    top_k: int
    enable_web_search: bool
    debug: bool
    __tools__: List[Any]
    debug_steps: List[Dict[str, Any]]
    tool_iterations: int