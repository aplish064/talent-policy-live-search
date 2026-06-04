from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient

from talent_policy_search.models import SearchResponse
from talent_policy_search.server import app, get_pipeline


class FakePipeline:
    async def search(self, query: str) -> SearchResponse:
        return SearchResponse(
            query=query,
            normalized_query=query,
            searched_at=datetime.now(timezone.utc),
            duration_seconds=0.01,
            official_sources_checked=1,
            candidate_pages_seen=1,
            results_returned=0,
            warnings=[],
            results=[],
        )


def test_search_endpoint_returns_stateless_response():
    app.dependency_overrides[get_pipeline] = lambda: FakePipeline()
    client = TestClient(app)

    try:
        response = client.post("/api/search", json={"query": "深圳"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "深圳"
    assert payload["persistence"] == "none"


def test_index_returns_html():
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_frontend_javascript_does_not_use_browser_storage():
    client = TestClient(app)

    response = client.get("/static/app.js")

    assert response.status_code == 200
    assert "localStorage" not in response.text
    assert "sessionStorage" not in response.text
    assert "indexedDB" not in response.text
    assert "document.cookie" not in response.text


def test_frontend_javascript_renders_summary_markdown():
    client = TestClient(app)

    response = client.get("/static/app.js")

    assert response.status_code == 200
    assert "summary_markdown" in response.text


def test_search_endpoint_rejects_empty_query():
    client = TestClient(app)

    response = client.post("/api/search", json={"query": "   "})

    assert response.status_code == 422


def test_search_endpoint_rejects_overlong_query():
    client = TestClient(app)

    response = client.post("/api/search", json={"query": "x" * 201})

    assert response.status_code == 422
