import pytest
from opensearchpy import OpenSearch

INDEX_NAME = "policies"

@pytest.fixture(scope="module")
def os_client():
    client = OpenSearch(
        hosts=[{"host": "localhost", "port": 9200}],
        use_ssl=False, verify_certs=False, timeout=20
    )
    yield client

def test_cluster_health(os_client):
    health = os_client.cluster.health()
    assert health["status"] in ["green", "yellow"]

def test_index_exists(os_client):
    assert os_client.indices.exists(index=INDEX_NAME)

def test_index_mapping(os_client):
    mapping = os_client.indices.get_mapping(index=INDEX_NAME)
    props = mapping[INDEX_NAME]["mappings"]["properties"]
    assert "text" in props
    assert "embedding" in props
    assert props["embedding"]["type"] == "knn_vector"

def test_search_returns_results(os_client):
    query = {
        "query": {"match": {"text": "asegurador"}},
        "size": 1
    }
    result = os_client.search(index=INDEX_NAME, body=query)
    assert result["hits"]["total"]["value"] > 0, f"No results found for query: {query}"

import random

def test_knn_search_returns_results(os_client):
    dim = 384
    qvec = [random.random() for _ in range(dim)]
    query = {
        "size": 1,
        "query": {
            "knn": {
                "embedding": {
                    "vector": qvec,
                    "k": 1
                }
            }
        }
    }
    result = os_client.search(index=INDEX_NAME, body=query)
    assert result["hits"]["total"]["value"] > 0, "kNN search returned no results"
