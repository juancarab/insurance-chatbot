from pathlib import Path
import sys, pytest
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
load_dotenv(ROOT / ".env")

from backend.app.agent.tools.web_search import WebSearchTool, WebSearchInput
import backend.app.agent.tools.web_search as ws
import requests

def test_web_search_missing_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    tool = WebSearchTool()
    out = tool.run(WebSearchInput(query="test").dict())
    assert isinstance(out, str) and out.startswith("Error:")

def test_web_search_timeout(monkeypatch):
    class FakeClient:
        def __init__(self, api_key): pass
        def search(self, *a, **k):
            raise requests.exceptions.Timeout("boom")
    monkeypatch.setenv("TAVILY_API_KEY", "x")
    monkeypatch.setattr(ws, "TavilyClient", FakeClient)

    tool = WebSearchTool()
    out = tool.run(WebSearchInput(query="test").dict())
    assert out.startswith("Error: Tavily timeout.")

def test_web_search_network_error(monkeypatch):
    class FakeClient:
        def __init__(self, api_key): pass
        def search(self, *a, **k):
            raise requests.exceptions.ConnectionError("no net")
    monkeypatch.setenv("TAVILY_API_KEY", "x")
    monkeypatch.setattr(ws, "TavilyClient", FakeClient)

    tool = WebSearchTool()
    out = tool.run(WebSearchInput(query="test").dict())
    assert out.startswith("Error: Network error while querying Tavily (ConnectionError).")

def test_web_search_generic_exception(monkeypatch):
    class FakeClient:
        def __init__(self, api_key): pass
        def search(self, *a, **k):
            raise Exception("unexpected")
    monkeypatch.setenv("TAVILY_API_KEY", "x")
    monkeypatch.setattr(ws, "TavilyClient", FakeClient)

    tool = WebSearchTool()
    out = tool.run(WebSearchInput(query="test").dict())
    assert out.startswith("Error: Web search unavailable (Exception).")
