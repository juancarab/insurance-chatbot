from typing import List

from langchain_core.tools import BaseTool

from .tools.web_search.web_search import WebSearchTool
from .tools.retrieval.haystack_opensearch_tool import retrieval_tool


def build_tools(enable_web_search: bool) -> List[BaseTool]:
    tools: List[BaseTool] = [retrieval_tool]

    if enable_web_search:
        tools.append(WebSearchTool())

    return tools