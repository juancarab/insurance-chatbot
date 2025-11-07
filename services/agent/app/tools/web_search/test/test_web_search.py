from pathlib import Path
import sys
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
load_dotenv(dotenv_path=ROOT / ".env")

from agent.app.tools.web_search import WebSearchTool, WebSearchInput


def test_web_search_returns_results_or_error():
    tool = WebSearchTool()
    args = WebSearchInput(
        query="health insurance Argentina",
        max_results=3,
        freshness_days=30,
    )
    out = tool.run(args.model_dump())

    assert isinstance(out, list)

    if out and "error" not in out[0]:
        assert len(out) >= 1
        assert all("title" in r and "url" in r for r in out)
        assert all(r["source"] == "tavily" for r in out)
        assert all(r["url"].startswith("http") for r in out)
    else:
        err = out[0]
        assert "error" in err
        assert "error_type" in err
        assert err.get("source") == "tavily"
