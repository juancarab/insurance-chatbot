from pathlib import Path
import sys
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
load_dotenv(dotenv_path=ROOT / ".env")

from agent.app.tools.web_search import WebSearchTool, WebSearchInput

def test_web_search_returns_results():
    tool = WebSearchTool()
    args = WebSearchInput(query="health insurance Argentina", max_results=3)
    out = tool.run(args.model_dump())
    assert isinstance(out, list) and len(out) >= 1
    assert all("title" in r and "url" in r for r in out)
    for r in out:
        assert r["source"] == "tavily"
        assert r["url"].startswith("http")