from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import parse_qs, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from talent_activity_search.config import Settings, get_settings


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


PUBLIC_SEARCH_PAGE_ENDPOINT = "https://html.duckduckgo.com/html/"
PUBLIC_SEARCH_USER_AGENT = (
    "Mozilla/5.0 (compatible; TalentActivityLiveSearch/0.1; public-search-fallback)"
)


class AnySearchProvider:
    def __init__(
        self,
        settings: Settings | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        fallback: WebSearchProvider | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._transport = transport
        self._fallback = fallback
        if self._fallback is None and self._settings.enable_public_search_page_fallback:
            self._fallback = PublicSearchPageProvider(self._settings, transport=self._transport)

    async def search(
        self,
        query: str,
        *,
        max_results: int | None = None,
    ) -> list[WebSearchResult]:
        try:
            return await self._search_anysearch(query, max_results=max_results)
        except Exception as exc:
            if self._fallback is None:
                raise
            try:
                return await self._fallback.search(query, max_results=max_results)
            except Exception as fallback_exc:
                raise WebSearchUnavailableError(
                    f"{exc}; public search fallback failed: {fallback_exc}"
                ) from None

    async def _search_anysearch(
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


class PublicSearchPageProvider:
    """Fallback that scrapes a public search results page when AnySearch is unavailable."""

    def __init__(
        self,
        settings: Settings | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        endpoint: str = PUBLIC_SEARCH_PAGE_ENDPOINT,
    ) -> None:
        self._settings = settings or get_settings()
        self._transport = transport
        self._endpoint = endpoint

    async def search(
        self,
        query: str,
        *,
        max_results: int | None = None,
    ) -> list[WebSearchResult]:
        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=self._settings.web_search_timeout_seconds,
                transport=self._transport,
            ) as client:
                response = await client.get(
                    self._endpoint,
                    params={"q": query},
                    headers={
                        "accept": "text/html,application/xhtml+xml,text/plain,*/*;q=0.8",
                        "user-agent": PUBLIC_SEARCH_USER_AGENT,
                    },
                )
                response.raise_for_status()
        except Exception as exc:
            raise WebSearchUnavailableError(f"Public search page request failed: {exc}") from None

        return parse_public_search_page_results(
            response.text,
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


def parse_public_search_page_results(html_text: str, *, limit: int) -> list[WebSearchResult]:
    soup = BeautifulSoup(html_text, "lxml")
    anchors = list(soup.select("a.result__a")) or list(soup.select("a[href]"))
    parsed: list[WebSearchResult] = []
    seen_urls: set[str] = set()

    for anchor in anchors:
        if len(parsed) >= limit:
            break
        url = _public_search_result_url(anchor.get("href"))
        if url is None or url in seen_urls:
            continue
        seen_urls.add(url)

        title = _safe_text(anchor.get_text(" ", strip=True)) or url
        snippet = _public_search_result_snippet(anchor)
        parsed.append(
            WebSearchResult(
                title=title,
                url=url,
                snippet=snippet,
                source="public_search_page",
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


def _public_search_result_url(href: object) -> str | None:
    href_text = _safe_text(href)
    if href_text is None:
        return None

    absolute_url = urljoin(PUBLIC_SEARCH_PAGE_ENDPOINT, href_text)
    try:
        parsed = urlparse(absolute_url)
    except ValueError:
        return None

    host = (parsed.hostname or "").lower().removeprefix("www.")
    if host.endswith("duckduckgo.com"):
        unwrapped = parse_qs(parsed.query).get("uddg", [None])[0]
        if unwrapped is None:
            return None
        return _safe_search_result_url(unwrapped)

    return _safe_search_result_url(absolute_url)


def _safe_search_result_url(url: str) -> str | None:
    try:
        parsed = urlparse(url)
    except ValueError:
        return None
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return None
    return url


def _public_search_result_snippet(anchor: Any) -> str | None:
    result = anchor.find_parent(class_="result")
    if result is None:
        return None
    snippet = result.select_one(".result__snippet")
    if snippet is None:
        return None
    return _safe_text(snippet.get_text(" ", strip=True))


def _safe_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = " ".join(value.split())
    return stripped or None
