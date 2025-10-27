from pathlib import Path
import os
import pytest
from tavily import TavilyClient

try:
    from dotenv import load_dotenv  # noqa: E402
    ROOT = Path(__file__).resolve().parents[1]
    load_dotenv(ROOT / ".env")
except Exception:
    pass

@pytest.fixture(scope="module")
def client():
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        pytest.skip("TAVILY_API_KEY not found", allow_module_level=True)
    return TavilyClient(api_key=api_key)

@pytest.mark.parametrize(
    "query",
    [
        "latest news on health insurance regulation in Argentina",
        "health insurance market Argentina 2025",
    ],
)

def test_tavily_search_returns_results(client, query):
    resp = client.search(query, max_results=3)
    assert isinstance(resp, dict), "The response must be Dict"
    assert "results" in resp, "The response must be include the key: results"
    results = resp.get("results", [])

    # Si la API es accesible pero sin resultados por query el test sera un fail informativo
    if not results:
        pytest.xfail(f"No results for query: {query!r}. (Red/recencia/ratelimit)")

    assert len(results) >= 1, "Must have one results at least"
    for r in results:
        assert isinstance(r, dict)
        assert r.get("url", "").startswith("http"), f"Invalid URL: {r}"
        assert r.get("title", "") != "", "Title required"
        # content puede llamarse content o snippet segun la libreria
        assert (r.get("content") or r.get("snippet") or "").strip() != "", "snippet or content required"
