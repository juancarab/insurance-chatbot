"""
WebSearchTool - Tavily web search tool for LangChain agents.
This tool allows the agent to perform live web searches when the local
knowledge base (OpenSearch) does not have enough context.

Devuelve una lista de dicts [{title, url, snippet, source='tavily'}]
o un string de error que comienza con "Error: ...".
"""

from __future__ import annotations
import os
from typing import List, Dict, Optional, Union

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun

from tavily import TavilyClient
from tavily.errors import MissingAPIKeyError
import requests

# Cargar variables de entorno desde .env
load_dotenv()


class WebSearchInput(BaseModel):
    query: str = Field(..., description="User question to search on the web.")
    max_results: int = Field(5, ge=1, le=10, description="Number of results to return (<=10).")
    freshness_days: int = Field(30, ge=1, le=365, description="Prefer recency within N days.")


class WebSearchTool(BaseTool):
    """
    LangChain Tool for performing web searches using the Tavily API.
    Returns:
      - List[Dict] with keys: title, url, snippet, source='tavily'
      - OR str that begins with "Error: ..." when something fails
    """
    name = "web_search"
    description = (
        "Search the web for up-to-date information using the Tavily API. "
        "Return a concise list of {title, url, snippet, source='tavily'} or "
        "a descriptive error string starting with 'Error: ...'."
    )
    args_schema = WebSearchInput

    def _run(
        self,
        query: str,
        max_results: int = 5,
        freshness_days: int = 30,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Union[List[Dict], str]:
        """
        Run a web search using Tavily.
        Returns: List[{title, url, snippet, source='tavily'}] OR "Error: ..."
        """
        try:
            api_key = os.getenv("TAVILY_API_KEY")
            if not api_key:
                raise MissingAPIKeyError("TAVILY_API_KEY not set")

            client = TavilyClient(api_key=api_key)
            print(f"[WebSearchTool] Searching Tavily for: '{query}' (max_results={max_results})")

            resp = client.search(
                query=query,
                max_results=max_results,
                search_depth="advanced",
                days=freshness_days,
            )

            results: List[Dict] = []
            for r in resp.get("results", [])[:max_results]:
                title = (r.get("title") or "").strip()
                url = (r.get("url") or "").strip()
                if not url:
                    continue
                snippet = (r.get("content") or r.get("snippet") or "").replace("\n", " ").strip()[:300]
                results.append({
                    "title": title[:160],
                    "url": url,
                    "snippet": snippet,
                    "source": "tavily",
                })

            print(f"[WebSearchTool] [OK] Retrieved {len(results)} results from Tavily.")
            return results

        except MissingAPIKeyError:
            return "Error: API key not found."
        except requests.exceptions.Timeout: 
            return "Error: Tavily timeout."
        except requests.exceptions.RequestException as e:
            # Cubre ConnectionError, httprror, etc
            return f"Error: Network error while querying Tavily ({type(e).__name__})."
        except Exception as e:
            return f"Error: Web search unavailable ({type(e).__name__})."

    async def _arun(self, *args, **kwargs):
        raise NotImplementedError("Asynchronous mode not implemented for WebSearchTool.")
