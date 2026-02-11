import os
import pytest
from fastapi.testclient import TestClient

from src.service.app import app


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_readyz(client):
    r = client.get("/readyz")
    assert r.status_code == 200
    assert r.json() == {"status": "ready"}


def test_search_returns_results(client):
    r = client.post("/search", json={"query": "kubernetes", "top_k": 3})
    assert r.status_code == 200
    data = r.json()
    assert data["query"] == "kubernetes"
    assert data["top_k"] == 3
    assert isinstance(data["results"], list)
    assert len(data["results"]) >= 1
    assert "id" in data["results"][0]
    assert "text" in data["results"][0]
    assert "score" in data["results"][0]

def test_classify_smoke(client):
    if os.getenv("RUN_MODEL_TESTS", "0") != "1":
        pytest.skip("Set RUN_MODEL_TESTS=1 to run model-loading tests")

    r = client.post("/classify", json={"text": "this movie was great"})
    assert r.status_code == 200
    data = r.json()
    assert data["label"] in ("positive", "negative")
    assert 0.0 <= data["prob"] <= 1.0
    assert len(data["probs"]) == 2

