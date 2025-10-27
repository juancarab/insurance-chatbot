"""
WebSearchTool - Tavily web search tool for Langchain agents.
This tool allows the agent to perform live web searches when the local
knowledge base (Opensearch) does not have enough context
"""

from __future__ import annotations
import os
from typing import List, Dict, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun
from tavily import TavilyClient

# Para que funcione local o en docker
load_dotenv()


class WebSearchInput(BaseModel):
    # Define the expected inputs for the web_search tool
    query: str = Field(..., description="User question to search on the web.")
    max_results: int = Field(5, ge=1, le=10, description="Number of results to return (<=10).")
    freshness_days: int = Field(30, ge=1, le=365, description="Prefer recency within N days.")


class WebSearchTool(BaseTool):
    """
    LangChain Tool for performing web searches using Tavily API
    Returns a list of dictionaries with title, url, snippet and source
    """
    name = "web_search"
    description = (
        "Search the web for up to date information using the tavily API"
        "Use this when the local knowledge base lacks context or recent data"
    )
    args_schema = WebSearchInput

    def _run(
        self,
        query: str,
        max_results: int = 5,
        freshness_days: int = 30,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> List[Dict]:
        """
        Run a web search using Tavily.
        Returns: List[{title, url, snippet, source='tavily'}]
        """
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY not found")

        client = TavilyClient(api_key=api_key)
        print(f"[WebSearchTool] Searching Tavily for: '{query}' (max_results={max_results})")

        resp = client.search(
            query=query,
            max_results=max_results,
            search_depth="advanced",
            days=freshness_days,
        )

        results = []
        for r in resp.get("results", [])[:max_results]:
            title = (r.get("title") or "").strip()
            url = (r.get("url") or "").strip()
            snippet = (r.get("content") or "").replace("\n", " ").strip()[:300]
            if url:
                results.append({
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "source": "tavily",
                })

        print(f"[WebSearchTool] [OK] Retrieved {len(results)} results from tavily")
        return results

    async def _arun(self, *args, **kwargs):
        raise NotImplementedError("Asynchronous mode not implemented for websearchtool")
