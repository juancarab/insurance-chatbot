"""
HybridOpenSearchTool - LangChain Tool for hybrid search (BM25 + vector) on internal PDFs.
Returns a list of {text, metadata} or a string starting with 'Error: ...' on failures.
"""

from __future__ import annotations
from typing import Optional, List, Dict, Union, Any
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun
from opensearchpy import OpenSearch, ConnectionError as OSConnectionError, TransportError, RequestError


class RetrieverInput(BaseModel):
    query: str = Field(..., description="User query for internal policy documents")
    k: int = Field(8, ge=1, le=20, description="Top-K documents to retrieve")


class HybridOpenSearchTool(BaseTool):
    """
    Hybrid search over internal PDFs (BM25 + kNN).
    Returns a list of {text, metadata} or an error string starting with 'Error: ...'.
    """
    name = "hybrid_opensearch_search"
    description = (
        "Hybrid search over internal PDFs (BM25 + kNN). "
        "Returns a list of {text, metadata} or a string starting with 'Error: ...' if something fails."
    )
    args_schema = RetrieverInput

    # ⬇️ Declarar los campos como parte del modelo Pydantic
    client: Any  # OpenSearch o un fake/mocked client en tests
    index_name: str

    def _run(
        self,
        query: str,
        k: int = 8,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Union[List[Dict], str]:
        try:
            body = {
                "query": {"match": {"text": query}},
                "_source": ["text", "metadata"],
                "size": k,
            }
            response = self.client.search(index=self.index_name, body=body)
            hits = response.get("hits", {}).get("hits", [])

            results = [
                {
                    "text": h["_source"].get("text", ""),
                    "metadata": h["_source"].get("metadata", {}),
                }
                for h in hits
            ]
            return results

        except OSConnectionError:
            return "Error: Unable to connect to the OpenSearch database."
        except RequestError as e:
            return f"Error: Invalid Query OpenSearch ({e.error})."
        except TransportError as e:
            return f"Error: Transport error with OpenSearch (status {e.status_code})."
        except Exception as e:
            return f"Error: Internal search unavailable ({type(e).__name__})."

    async def _arun(self, *args, **kwargs):
        raise NotImplementedError("Asynchronous mode not implemented for HybridOpenSearchTool.")
