from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from talent_policy_search.config import Settings, get_settings


class WebSearchUnavailableError(RuntimeError):
    """Raised when a configured web search provider cannot return candidates."""


@dataclass(frozen=True)
class WebSearchResult:
    title: str
    url: str
    snippet: str | None = None
    source: str = "web_search"


class WebSearchProvider(Protocol):
    async def search(
        self,
        query: str,
        *,
        max_results: int | None = None,
    ) -> list[WebSearchResult]:
        """Return ranked public-web search results for a query."""


class NullWebSearchProvider:
    async def search(
        self,
        query: str,
        *,
        max_results: int | None = None,
    ) -> list[WebSearchResult]:
        return []


class AnySearchProvider:
    def __init__(
        self,
        settings: Settings | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._transport = transport

    async def search(
        self,
        query: str,
        *,
        max_results: int | None = None,
    ) -> list[WebSearchResult]:
        api_key = self._settings.anysearch_api_key.get_secret_value().strip()
        headers = {"content-type": "application/json"}
        if api_key:
            headers["authorization"] = f"Bearer {api_key}"

        payload = {
            "query": query,
            "max_results": max_results or self._settings.max_web_search_results,
            "content_types": ["web"],
            "zone": self._settings.anysearch_zone,
        }
        try:
            async with httpx.AsyncClient(
                timeout=self._settings.web_search_timeout_seconds,
                transport=self._transport,
            ) as client:
                response = await client.post(
                    self._settings.anysearch_endpoint,
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
        except Exception as exc:
            raise WebSearchUnavailableError(f"AnySearch request failed: {exc}") from None

        try:
            response_json = response.json()
        except ValueError:
            raise WebSearchUnavailableError("AnySearch response was not JSON.") from None

        return parse_anysearch_results(
            response_json,
            limit=max_results or self._settings.max_web_search_results,
        )


def parse_anysearch_results(payload: Any, *, limit: int) -> list[WebSearchResult]:
    raw_results = _raw_results(payload)
    parsed: list[WebSearchResult] = []
    seen_urls: set[str] = set()
    for item in raw_results:
        if len(parsed) >= limit:
            break
        if not isinstance(item, dict):
            continue
        url = _safe_text(item.get("url"))
        if url is None or url in seen_urls:
            continue
        seen_urls.add(url)
        title = _safe_text(item.get("title")) or url
        snippet = (
            _safe_text(item.get("content"))
            or _safe_text(item.get("description"))
            or _safe_text(item.get("snippet"))
        )
        parsed.append(
            WebSearchResult(
                title=title,
                url=url,
                snippet=snippet,
                source="anysearch",
            )
        )
    return parsed


def _raw_results(payload: Any) -> list[Any]:
    if not isinstance(payload, dict):
        return []

    data = payload.get("data")
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        return data["results"]
    if isinstance(payload.get("results"), list):
        return payload["results"]
    return []


def _safe_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = " ".join(value.split())
    return stripped or None
