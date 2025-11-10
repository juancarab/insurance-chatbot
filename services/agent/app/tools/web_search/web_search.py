"""
WebSearchTool - Tavily web search tool for LangChain agents.
Used when the local knowledge base (OpenSearch) does not provide enough context.
Returns a list of dicts with {title, url, snippet, source='tavily'}
or a single-item list containing an error dict.
"""

from __future__ import annotations
import os
import logging
import asyncio  # <--- AÑADIDO
from typing import List, Dict, Optional, Type

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun

from tavily import TavilyClient
from tavily.errors import MissingAPIKeyError
import requests

load_dotenv()

logger = logging.getLogger(__name__)


class WebSearchInput(BaseModel):
    query: str = Field(..., description="User question to search on the web.")
    max_results: int = Field(5, ge=1, le=10, description="Number of results to return (<=10).")
    freshness_days: int = Field(30, ge=1, le=365, description="Prefer recency within N days.")


class WebSearchTool(BaseTool):
    """
    LangChain Tool for performing web searches using Tavily API.
    Returns a list of dictionaries with title, url, snippet and source.
    On error, returns a list with one dict describing the error.
    """
    name: str = "web_search"
    description: str = (
        "Search the web for up-to-date information using the Tavily API. "
        "Use this when the local knowledge base lacks context or recent data."
    )
    args_schema: Type[WebSearchInput] = WebSearchInput

    def _error_result(self, message: str, error_type: str) -> List[Dict]:
        """Devuelve una lista con un solo dict de error para no romper al agente."""
        return [{
            "error": message,
            "error_type": error_type,
            "source": "tavily",
        }]

    def _run(
        self,
        query: str,
        max_results: int = 5,
        freshness_days: int = 30,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> List[Dict]:
        """
        Run a web search using Tavily.
        Returns: List[{title, url, snippet, source='tavily'}] OR List[{error, ...}]
        """
        try:
            api_key = os.getenv("TAVILY_API_KEY")
            if not api_key:
                logger.warning("WebSearchTool: TAVILY_API_KEY not found")
                raise MissingAPIKeyError("TAVILY_API_KEY not set")

            client = TavilyClient(api_key=api_key)
            logger.info(
                "[WebSearchTool] Searching Tavily for: '%s' (max_results=%s, freshness_days=%s)",
                query, max_results, freshness_days
            )

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

            logger.info("[WebSearchTool] [OK] Retrieved %d results from Tavily", len(results))
            return results

        except MissingAPIKeyError as e:
            logger.error("WebSearchTool: missing API key: %s", e)
            return self._error_result("Error: API key not found.", "missing_api_key")

        except requests.exceptions.Timeout as e:
            logger.error("WebSearchTool: Tavily timeout: %s", e)
            return self._error_result("Error: Tavily timeout.", "timeout")

        except requests.exceptions.RequestException as e:
            logger.error("WebSearchTool: network error to Tavily: %s", e, exc_info=True)
            return self._error_result(
                f"Error: Network error while querying Tavily ({type(e).__name__}).",
                "network_error"
            )

        except Exception as e:
            logger.exception("WebSearchTool: unexpected error")
            return self._error_result(
                f"Error: Web search unavailable ({type(e).__name__}).",
                "unexpected_error"
            )

    async def _arun(
        self,
        query: str,
        max_results: int = 5,
        freshness_days: int = 30,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> List[Dict]:
        """Ejecuta el _run síncrono en un hilo separado."""
        return await asyncio.to_thread(
            self._run,
            query=query,
            max_results=max_results,
            freshness_days=freshness_days,
            run_manager=run_manager
        )