from pathlib import Path
import sys
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
load_dotenv(ROOT / ".env")

from backend.app.agent.tools.retriever_tool import HybridOpenSearchTool, RetrieverInput

# Simula el cliente caido
class BoomClient:
    def search(self, *a, **k):
        from opensearchpy import ConnectionError
        raise ConnectionError("down")

def test_retriever_connection_error():
    tool = HybridOpenSearchTool(client=BoomClient(), index_name="policies")
    out = tool.run(RetrieverInput(query="test", k=3).dict())
    assert isinstance(out, str) and out.startswith("Error:")
