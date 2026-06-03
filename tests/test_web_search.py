from __future__ import annotations

import httpx
import pytest

from talent_activity_search.config import Settings
from talent_activity_search.web_search import (
    AnySearchProvider,
    WebSearchUnavailableError,
    parse_anysearch_results,
    parse_public_search_page_results,
)


def test_parse_anysearch_results_reads_v1_response_shape():
    payload = {
        "code": 0,
        "message": "success",
        "data": {
            "results": [
                {
                    "title": "人才政策-广州市人力资源和社会保障局网站",
                    "url": "https://rsj.gz.gov.cn/ywzt/rcgz/renczc/",
                    "content": "广州人才政策",
                },
                {
                    "title": "duplicate",
                    "url": "https://rsj.gz.gov.cn/ywzt/rcgz/renczc/",
                    "content": "duplicate",
                },
                {
                    "title": "黄埔人才津贴",
                    "url": "http://www.hp.gov.cn/post_10358329.html",
                    "description": "人才津贴",
                },
            ]
        },
    }

    results = parse_anysearch_results(payload, limit=5)

    assert [result.url for result in results] == [
        "https://rsj.gz.gov.cn/ywzt/rcgz/renczc/",
        "http://www.hp.gov.cn/post_10358329.html",
    ]
    assert results[0].title == "人才政策-广州市人力资源和社会保障局网站"
    assert results[0].snippet == "广州人才政策"
    assert results[0].source == "anysearch"


def test_parse_anysearch_results_ignores_malformed_items_and_honors_limit():
    payload = {
        "results": [
            "bad",
            {"title": "missing url"},
            {"url": "https://www.gz.gov.cn/a.html"},
            {"url": "https://www.gz.gov.cn/b.html"},
        ]
    }

    results = parse_anysearch_results(payload, limit=1)

    assert len(results) == 1
    assert results[0].title == "https://www.gz.gov.cn/a.html"


def test_parse_public_search_page_results_reads_duckduckgo_html_results():
    html = """
    <html>
      <body>
        <div class="result">
          <a class="result__a" href="/l/?uddg=https%3A%2F%2Fwww.sz.gov.cn%2Fa.html">
            深圳人才活动
          </a>
          <a class="result__a" href="/l/?uddg=https%3A%2F%2Fwww.sz.gov.cn%2Fa.html">
            duplicate
          </a>
          <a class="result__a" href="https://hrss.sz.gov.cn/b.html">深圳人社活动</a>
          <a class="result__a" href="/about">internal</a>
          <div class="result__snippet">官方活动报名通知</div>
        </div>
      </body>
    </html>
    """

    results = parse_public_search_page_results(html, limit=5)

    assert [result.url for result in results] == [
        "https://www.sz.gov.cn/a.html",
        "https://hrss.sz.gov.cn/b.html",
    ]
    assert results[0].title == "深圳人才活动"
    assert results[0].snippet == "官方活动报名通知"
    assert results[0].source == "public_search_page"


@pytest.mark.asyncio
async def test_anysearch_provider_uses_public_search_page_fallback_when_api_fails():
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.host == "api.anysearch.com":
            return httpx.Response(402, json={"error": "quota exhausted"}, request=request)
        return httpx.Response(
            200,
            text="""
            <html>
              <div class="result">
                <a class="result__a" href="/l/?uddg=https%3A%2F%2Fwww.sz.gov.cn%2Fact.html">
                  深圳人才活动
                </a>
                <div class="result__snippet">报名入口</div>
              </div>
            </html>
            """,
            request=request,
        )

    settings = Settings(
        ANYSEARCH_API_KEY="anysearch-secret",
        ENABLE_PUBLIC_SEARCH_PAGE_FALLBACK=True,
        _env_file=None,
    )
    provider = AnySearchProvider(settings=settings, transport=httpx.MockTransport(handler))

    results = await provider.search("深圳 人才活动", max_results=3)

    assert [request.url.host for request in requests] == [
        "api.anysearch.com",
        "html.duckduckgo.com",
    ]
    assert results[0].url == "https://www.sz.gov.cn/act.html"
    assert results[0].source == "public_search_page"


@pytest.mark.asyncio
async def test_anysearch_provider_raises_when_api_fails_and_public_fallback_is_disabled():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(402, json={"error": "quota exhausted"}, request=request)

    settings = Settings(ENABLE_PUBLIC_SEARCH_PAGE_FALLBACK=False, _env_file=None)
    provider = AnySearchProvider(settings=settings, transport=httpx.MockTransport(handler))

    with pytest.raises(WebSearchUnavailableError, match="AnySearch request failed"):
        await provider.search("深圳 人才活动", max_results=3)
