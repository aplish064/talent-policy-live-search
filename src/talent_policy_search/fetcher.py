import asyncio
from collections.abc import Callable
from dataclasses import dataclass
import ssl
from typing import Any
from urllib.parse import urljoin

import httpx

from talent_policy_search.config import Settings, get_settings


ACCEPT_HEADER = "text/html,application/xhtml+xml,application/pdf,text/plain,*/*;q=0.8"
USER_AGENT = "TalentPolicyLiveSearch/0.1 (+official-source-retrieval)"
REDIRECT_STATUS_CODES = {301, 302, 303, 307, 308}
MAX_REDIRECTS = 20
_TLS_HANDSHAKE_ERROR_MARKERS = (
    "bad ecpoint",
    "bad_ecpoint",
    "sslv3",
    "ssl/v3",
    "ssl error",
    "handshake failure",
    "tlsv1 alert",
    "tlsv1.",
    "wrong version number",
)


class FetchTooLargeError(RuntimeError):
    """Raised when an in-memory fetch would exceed the configured byte limit."""


class FetchDisallowedRedirectError(RuntimeError):
    """Raised before fetching a redirect target outside the allowed boundary."""


@dataclass(frozen=True)
class FetchedPage:
    url: str
    final_url: str
    content_type: str
    content: bytes


class Fetcher:
    def __init__(
        self,
        settings: Settings | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        scrapling_fetcher: Any | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._transport = transport
        self._scrapling_fetcher = scrapling_fetcher
        if self._scrapling_fetcher is None and self._settings.enable_scrapling_fallback:
            self._scrapling_fetcher = _make_scrapling_fetcher(self._settings)

    async def fetch(
        self,
        url: str,
        allowed_redirect_url: Callable[[str], bool] | None = None,
    ) -> FetchedPage:
        headers = {
            "accept": ACCEPT_HEADER,
            "user-agent": USER_AGENT,
        }
        max_bytes = self._settings.max_fetch_bytes_per_page
        try:
            if allowed_redirect_url is not None:
                return await self._fetch_with_redirect_policy(
                    url,
                    allowed_redirect_url,
                    headers,
                    max_bytes,
                )
            return await self._fetch_httpx(url, headers, max_bytes)
        except Exception as exc:
            if self._is_tls_handshake_error(exc):
                try:
                    if allowed_redirect_url is not None:
                        return await self._fetch_with_redirect_policy(
                            url,
                            allowed_redirect_url,
                            headers,
                            max_bytes,
                            ssl_context=self._legacy_tls_compatible_context(),
                        )
                    return await self._fetch_httpx(
                        url,
                        headers,
                        max_bytes,
                        ssl_context=self._legacy_tls_compatible_context(),
                    )
                except Exception as tls_exc:
                    exc = tls_exc

            if not self._can_try_scrapling_fallback(exc):
                raise
            page = await self._scrapling_fetcher.fetch(url)
            if allowed_redirect_url is not None and not allowed_redirect_url(page.final_url):
                raise FetchDisallowedRedirectError(
                    f"Disallowed Scrapling final URL: {page.final_url}"
                ) from None
            if len(page.content) > max_bytes:
                raise FetchTooLargeError(
                    f"Scrapling response body exceeds max fetch bytes: "
                    f"{len(page.content)} > {max_bytes}"
                )
            return page

    async def _fetch_httpx(
        self,
        url: str,
        headers: dict[str, str],
        max_bytes: int,
        ssl_context: ssl.SSLContext | None = None,
    ) -> FetchedPage:
        async with httpx.AsyncClient(
            follow_redirects=True,
            headers=headers,
            timeout=min(self._settings.search_timeout_seconds, 30),
            verify=ssl_context or True,
            transport=self._transport,
        ) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                self._raise_if_content_length_exceeds_limit(response, max_bytes)
                content = await self._read_bounded_content(response, max_bytes)

        return FetchedPage(
            url=url,
            final_url=str(response.url),
            content_type=response.headers.get("content-type") or "application/octet-stream",
            content=content,
        )

    def _can_try_scrapling_fallback(self, exc: Exception) -> bool:
        if self._scrapling_fetcher is None:
            return False
        if isinstance(exc, (FetchTooLargeError, FetchDisallowedRedirectError)):
            return False
        return True

    @staticmethod
    def _is_tls_handshake_error(exc: Exception) -> bool:
        message = str(exc).casefold()
        return any(marker in message for marker in _TLS_HANDSHAKE_ERROR_MARKERS)

    @staticmethod
    def _legacy_tls_compatible_context() -> ssl.SSLContext:
        context = ssl.create_default_context()
        try:
            context.set_ciphers("DEFAULT:@SECLEVEL=1")
        except ssl.SSLError:
            pass
        return context

    async def _fetch_with_redirect_policy(
        self,
        url: str,
        allowed_redirect_url: Callable[[str], bool],
        headers: dict[str, str],
        max_bytes: int,
        ssl_context: ssl.SSLContext | None = None,
    ) -> FetchedPage:
        current_url = url
        async with httpx.AsyncClient(
            follow_redirects=False,
            headers=headers,
            timeout=min(self._settings.search_timeout_seconds, 30),
            verify=ssl_context or True,
            transport=self._transport,
        ) as client:
            for _ in range(MAX_REDIRECTS + 1):
                async with client.stream("GET", current_url) as response:
                    location = response.headers.get("location")
                    if response.status_code in REDIRECT_STATUS_CODES and location:
                        next_url = urljoin(str(response.url), location)
                        if not allowed_redirect_url(next_url):
                            raise FetchDisallowedRedirectError(
                                f"Disallowed redirect URL: {next_url}"
                            )
                        current_url = next_url
                        continue

                    response.raise_for_status()
                    self._raise_if_content_length_exceeds_limit(response, max_bytes)
                    content = await self._read_bounded_content(response, max_bytes)
                    return FetchedPage(
                        url=url,
                        final_url=str(response.url),
                        content_type=response.headers.get("content-type")
                        or "application/octet-stream",
                        content=content,
                    )

        raise httpx.TooManyRedirects(f"Exceeded maximum redirects: {MAX_REDIRECTS}")

    @staticmethod
    def _raise_if_content_length_exceeds_limit(
        response: httpx.Response,
        max_bytes: int,
    ) -> None:
        content_length = response.headers.get("content-length")
        if content_length is None:
            return

        try:
            byte_count = int(content_length)
        except ValueError:
            return

        if byte_count > max_bytes:
            raise FetchTooLargeError(
                f"Response content length exceeds max fetch bytes: {byte_count} > {max_bytes}"
            )

    @staticmethod
    async def _read_bounded_content(response: httpx.Response, max_bytes: int) -> bytes:
        total_bytes = 0
        chunks = []
        async for chunk in response.aiter_bytes():
            total_bytes += len(chunk)
            if total_bytes > max_bytes:
                raise FetchTooLargeError(
                    f"Response body exceeds max fetch bytes: {total_bytes} > {max_bytes}"
                )
            chunks.append(chunk)

        return b"".join(chunks)


class ScraplingFetcherAdapter:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def fetch(self, url: str) -> FetchedPage:
        return await asyncio.to_thread(self._fetch_sync, url)

    def _fetch_sync(self, url: str) -> FetchedPage:
        from scrapling.fetchers import FetcherSession

        timeout = min(self._settings.search_timeout_seconds, 30)
        with FetcherSession(impersonate="chrome") as session:
            page = session.get(
                url,
                stealthy_headers=True,
                timeout=timeout,
            )
        content = _scrapling_page_content(page)
        return FetchedPage(
            url=url,
            final_url=str(getattr(page, "url", None) or url),
            content_type=str(getattr(page, "content_type", None) or "text/html; charset=utf-8"),
            content=content,
        )


def _make_scrapling_fetcher(settings: Settings) -> ScraplingFetcherAdapter | None:
    try:
        import scrapling.fetchers  # noqa: F401
    except Exception:
        return None
    return ScraplingFetcherAdapter(settings)


def _scrapling_page_content(page: Any) -> bytes:
    for attribute in ("html", "content", "body", "text"):
        value = getattr(page, attribute, None)
        if value is None:
            continue
        if callable(value):
            value = value()
        if isinstance(value, bytes):
            return value
        if isinstance(value, str):
            return value.encode("utf-8", errors="replace")
    return str(page).encode("utf-8", errors="replace")
