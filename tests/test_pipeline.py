from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from talent_policy_search.config import Settings
from talent_policy_search.fetcher import FetchedPage
from talent_policy_search.models import PolicyCard
from talent_policy_search import pipeline as pipeline_module
from talent_policy_search.pipeline import SearchPipeline
from talent_policy_search.ranker import rank_policies
from talent_policy_search.web_search import NullWebSearchProvider, WebSearchResult

HTML_PARSE_FAILURE_SECRET = "SECRET-DO-NOT-LEAK-" + ("A" * 500)
NO_WEB_SEARCH = NullWebSearchProvider()


class FakeLLM:
    async def identify_query(self, query, entities):
        return {
            "normalized_query": "深圳",
            "query_type": "region",
            "matched_entity_ids": ["shenzhen"],
            "confidence": 0.99,
        }

    async def generate_discovery_terms(self, query, entity):
        return {
            "terms": [
                {
                    "term": "高层次人才 安家补贴",
                    "intent": "人才补贴",
                    "language": "zh",
                    "priority": 1,
                }
            ],
            "url_path_hints": ["talent", "policy"],
        }

    async def judge_candidate(self, query, candidate):
        return {
            "is_relevant": True,
            "is_official": True,
            "content_kind": "policy",
            "policy_types": ["安家补贴"],
            "should_extract_fulltext": True,
            "confidence": 0.9,
        }

    async def extract_policies(self, source, query_context, fulltext):
        return {
            "policies": [
                {
                    "title": "深圳市高层次人才奖励补贴办法",
                    "source_name": "深圳市人力资源和社会保障局",
                    "source_type": "government",
                    "official_url": source["url"],
                    "matched_entity": "深圳",
                    "jurisdiction": "中国大陆",
                    "policy_types": ["安家补贴"],
                    "applicable_to": ["高层次人才"],
                    "benefits": [
                        {
                            "type": "安家补贴",
                            "amount_text": "最高300万元",
                            "currency": "CNY",
                            "evidence": "给予高层次人才奖励补贴",
                        }
                    ],
                    "eligibility": [],
                    "application": {
                        "entry_url": None,
                        "materials": [],
                        "process": None,
                        "evidence": None,
                    },
                    "dates": {
                        "published_date": "2025-03-01",
                        "effective_date": None,
                        "deadline": None,
                        "valid_until": None,
                    },
                    "evidence_snippets": ["给予高层次人才奖励补贴"],
                    "summary": "深圳高层次人才可获得奖励补贴。",
                    "confidence": 0.9,
                    "completeness": 0.8,
                }
            ],
            "discard_reason": None,
        }


@dataclass
class FakeFetcher:
    async def fetch(self, url: str) -> FetchedPage:
        if url.endswith("/robots.txt"):
            return FetchedPage(
                url=url,
                final_url=url,
                content_type="text/plain",
                content=b"Sitemap: https://hrss.sz.gov.cn/sitemap.xml",
            )
        if url.endswith("/sitemap.xml"):
            content = b"""
            <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
              <url><loc>https://hrss.sz.gov.cn/talent-policy.html</loc></url>
            </urlset>
            """
            return FetchedPage(
                url=url,
                final_url=url,
                content_type="application/xml",
                content=content,
            )
        if url.endswith("/talent-policy.html"):
            content = (
                "<html><body><h1>深圳人才政策</h1>"
                "<p>给予高层次人才奖励补贴。</p></body></html>"
            ).encode()
            return FetchedPage(url=url, final_url=url, content_type="text/html", content=content)
        return FetchedPage(
            url=url,
            final_url=url,
            content_type="text/html",
            content=(
                '<a href="https://hrss.sz.gov.cn/talent-policy.html">人才政策</a>'
            ).encode(),
        )


@dataclass
class FakeWebSearch:
    urls: list[str]
    queries: list[str] = field(default_factory=list)

    async def search(self, query: str, *, max_results: int | None = None):
        self.queries.append(query)
        return [
            WebSearchResult(title=f"result-{index}", url=url, snippet="人才政策")
            for index, url in enumerate(self.urls)
        ]


class GenericThenPolicyWebSearch:
    async def search(self, query: str, *, max_results: int | None = None):
        return [
            WebSearchResult(title="首页", url="https://rsj.gz.gov.cn/", snippet="广州市人社局首页"),
            WebSearchResult(
                title="人才政策-广州市人力资源和社会保障局网站",
                url="https://rsj.gz.gov.cn/ywzt/rcgz/renczc/",
                snippet="关于做好广州市境外人才财政补贴申报准备工作的通知",
            ),
    ]


class GenericPolicyWebSearch:
    async def search(self, query: str, *, max_results: int | None = None):
        return [
            WebSearchResult(
                title="政策文件库",
                url="https://hrss.sz.gov.cn/zwgk/",
                snippet="政务公开 政府信息公开",
            ),
            WebSearchResult(
                title="深圳市高层次人才奖励补贴申报指南",
                url="https://hrss.sz.gov.cn/talent-policy/notice.html",
                snippet="高层次人才 申报补贴 措施",
            ),
        ]


@dataclass
class PolicyWebSearchFetcher:
    fetched_urls: list[str] = field(default_factory=list)

    async def fetch(self, url: str, allowed_redirect_url=None) -> FetchedPage:
        self.fetched_urls.append(url)
        if url == "https://hrss.sz.gov.cn/talent-policy/notice.html":
            return FetchedPage(
                url=url,
                final_url=url,
                content_type="text/html",
                content=(
                    "<html><body>"
                    "<h1>深圳市高层次人才奖励补贴申报指南</h1>"
                    "<p>高层次人才补贴申报期限为2025-12-31。</p>"
                    "</body></html>"
                ).encode(),
            )
        raise AssertionError(f"unexpected fetch: {url}")


class ApplicationEntryUrlLLM(FakeLLM):
    def __init__(self, entry_url: str) -> None:
        self.entry_url = entry_url

    async def extract_policies(self, source, query_context, fulltext):
        extracted = await super().extract_policies(source, query_context, fulltext)
        extracted["policies"][0]["application"]["entry_url"] = self.entry_url
        return extracted


@pytest.mark.asyncio
async def test_pipeline_returns_policy_cards_without_persistence(registry, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pipeline = SearchPipeline(
        registry=registry,
        llm=FakeLLM(),
        fetcher=FakeFetcher(),
        web_search=NO_WEB_SEARCH,
    )

    response = await pipeline.search("深圳")
    shenzhen = registry.match_entity("深圳")[0]
    allowed_seed_count = sum(
        1
        for seed_url in shenzhen.seed_urls
        if pipeline.domain_filter.is_allowed_for_entity(seed_url, shenzhen)
    )

    assert response.persistence == "none"
    assert response.official_sources_checked == allowed_seed_count
    assert response.results_returned == 1
    assert response.results[0].title == "深圳市高层次人才奖励补贴办法"
    assert response.results[0].official_url == "https://hrss.sz.gov.cn/talent-policy.html"
    assert response.results[0].score > 0
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_pipeline_clears_off_domain_application_entry_url(registry):
    pipeline = SearchPipeline(
        registry=registry,
        llm=ApplicationEntryUrlLLM("https://evil.example/apply"),
        fetcher=FakeFetcher(),
        web_search=NO_WEB_SEARCH,
    )

    response = await pipeline.search("深圳")

    assert response.results_returned == 1
    assert response.results[0].application.entry_url is None
    warning = next(
        (
            warning
            for warning in response.warnings
            if warning.message
            == "Extracted application entry URL is outside the allowed entity domain."
        ),
        None,
    )
    assert warning is not None
    assert len(warning.message) <= pipeline_module.MAX_WARNING_TEXT
    assert len(warning.source) <= pipeline_module.MAX_WARNING_SOURCE


@pytest.mark.asyncio
async def test_pipeline_keeps_official_domain_application_entry_url(registry):
    entry_url = "https://hrss.sz.gov.cn/apply"
    pipeline = SearchPipeline(
        registry=registry,
        llm=ApplicationEntryUrlLLM(entry_url),
        fetcher=FakeFetcher(),
        web_search=NO_WEB_SEARCH,
    )

    response = await pipeline.search("深圳")

    assert response.results_returned == 1
    assert response.results[0].application.entry_url == entry_url
    assert not any(
        "application entry URL is outside" in warning.message
        for warning in response.warnings
    )


@dataclass
class RecordingFetcher:
    fetched_urls: list[str] = field(default_factory=list)

    async def fetch(self, url: str) -> FetchedPage:
        self.fetched_urls.append(url)
        if url == "https://hrss.sz.gov.cn/":
            return FetchedPage(
                url=url,
                final_url=url,
                content_type="text/html",
                content=b"""
                <html><body>
                  <a href="https://evil.example/talent-policy.html">bad</a>
                  <a href="https://hrss.sz.gov.cn/talent-policy.html">good</a>
                </body></html>
                """,
            )
        if url.endswith("/robots.txt"):
            return FetchedPage(url=url, final_url=url, content_type="text/plain", content=b"")
        if url == "https://hrss.sz.gov.cn/talent-policy.html":
            return FetchedPage(
                url=url,
                final_url=url,
                content_type="text/html",
                content=(
                    "<html><body><p>给予高层次人才奖励补贴。</p></body></html>"
                ).encode(),
            )
        raise AssertionError(f"unexpected fetch: {url}")


@pytest.mark.asyncio
async def test_pipeline_does_not_fetch_off_domain_links(registry):
    fetcher = RecordingFetcher()
    pipeline = SearchPipeline(
        registry=registry,
        llm=FakeLLM(),
        fetcher=fetcher,
        web_search=NO_WEB_SEARCH,
    )

    await pipeline.search("深圳")

    assert "https://evil.example/talent-policy.html" not in fetcher.fetched_urls
    assert "https://hrss.sz.gov.cn/talent-policy.html" in fetcher.fetched_urls


class RaisingJudgeLLM(FakeLLM):
    def __init__(self) -> None:
        self.extract_calls = 0

    async def judge_candidate(self, query, candidate):
        raise RuntimeError("judge unavailable with payload that should stay bounded")

    async def extract_policies(self, source, query_context, fulltext):
        self.extract_calls += 1
        return await super().extract_policies(source, query_context, fulltext)


@pytest.mark.asyncio
async def test_pipeline_uses_deterministic_candidate_fallback_when_judge_fails(registry):
    llm = RaisingJudgeLLM()
    pipeline = SearchPipeline(
        registry=registry,
        llm=llm,
        fetcher=FakeFetcher(),
        web_search=NO_WEB_SEARCH,
    )

    response = await pipeline.search("深圳")

    assert llm.extract_calls >= 1
    assert response.results_returned == 1
    assert response.results[0].official_url == "https://hrss.sz.gov.cn/talent-policy.html"
    assert any(
        "candidate judgment failed" in warning.message.lower()
        for warning in response.warnings
    )


class MissingUrlLLM(FakeLLM):
    async def extract_policies(self, source, query_context, fulltext):
        extracted = await super().extract_policies(source, query_context, fulltext)
        del extracted["policies"][0]["official_url"]
        del extracted["policies"][0]["source_name"]
        del extracted["policies"][0]["matched_entity"]
        return extracted


class AliasFieldLLM(FakeLLM):
    async def extract_policies(self, source, query_context, fulltext):
        return {
            "policies": [
                {
                    "official_name": "广州市境外职业资格比照对应职称目录",
                    "official_url": source["url"],
                    "category": "职称评审",
                    "applicable_to": "境外专业人才",
                    "benefits": "职称比照认定支持",
                    "eligibility": "符合目录对应职业资格",
                    "application": "在线申报",
                    "evidence_snippets": "官方目录发布",
                    "published_date": "2025-12-22",
                    "confidence": "高",
                    "completeness": "60%",
                }
            ],
            "discard_reason": None,
        }


class NoRegistryLLM(FakeLLM):
    async def identify_query(self, query, entities):
        return {
            "normalized_query": query,
            "query_type": "region",
            "matched_entity_ids": [],
            "confidence": 0.9,
        }

    async def generate_discovery_terms(self, query, entity):
        return {
            "terms": [
                {"term": f"{query} 人才政策", "intent": "人才政策", "priority": 1},
                {"term": "人才补贴", "intent": "补贴", "priority": 2},
            ],
            "url_path_hints": ["rcgz", "rencai"],
        }

    async def extract_policies(self, source, query_context, fulltext):
        extracted = await super().extract_policies(source, query_context, fulltext)
        extracted["policies"][0]["title"] = "广州市人才政策"
        extracted["policies"][0]["source_name"] = "广州市人力资源和社会保障局"
        extracted["policies"][0]["official_url"] = source["url"]
        extracted["policies"][0]["matched_entity"] = query_context["entity"]["display_name"]
        return extracted


class NoRegistryJudgeFailLLM(NoRegistryLLM):
    async def judge_candidate(self, query, candidate):
        raise AssertionError("web search candidates should not require LLM judgment")


class OrganizerFailLLM(NoRegistryLLM):
    async def extract_policies(self, source, query_context, page_evidence):
        raise RuntimeError("organizer unavailable")


class MarkdownFailLLM(OrganizerFailLLM):
    async def summarize_markdown(self, query_context, policies):
        raise RuntimeError("markdown organizer unavailable")


class RecordingOrganizerLLM(NoRegistryLLM):
    def __init__(self) -> None:
        self.page_evidence = None

    async def extract_policies(self, source, query_context, page_evidence):
        self.page_evidence = page_evidence
        return await super().extract_policies(source, query_context, page_evidence)


class MarkdownLLM(NoRegistryLLM):
    async def summarize_markdown(self, query_context, policies):
        return "## Agent 汇总\n\n- 由 agent 整理的 Markdown。"


@dataclass
class WebOnlyFetcher:
    fetched_urls: list[str] = field(default_factory=list)

    async def fetch(self, url: str, allowed_redirect_url=None) -> FetchedPage:
        self.fetched_urls.append(url)
        if url == "https://rsj.gz.gov.cn/ywzt/rcgz/renczc/":
            return FetchedPage(
                url=url,
                final_url=url,
                content_type="text/html",
                content="<html><body><p>广州人才政策 人才补贴 申报指南。</p></body></html>".encode(),
            )
        raise AssertionError(f"unexpected fetch: {url}")


@dataclass
class WebListFetcher:
    fetched_urls: list[str] = field(default_factory=list)

    async def fetch(self, url: str, allowed_redirect_url=None) -> FetchedPage:
        self.fetched_urls.append(url)
        if url == "https://rsj.gz.gov.cn/ywzt/rcgz/renczc/":
            return FetchedPage(
                url=url,
                final_url=url,
                content_type="text/html",
                content="""
                <html>
                  <head><title>人才政策-广州市人力资源和社会保障局网站</title></head>
                  <body>
                    <h1>人才政策</h1>
                    <ul>
                      <li>
                        <a href="/ywzt/rcgz/renczc/content/post_1.html">关于做好广州市境外人才财政补贴申报准备工作的通知</a>
                        <span>2025-03-12</span>
                      </li>
                    </ul>
                  </body>
                </html>
                """.encode(),
            )
        raise AssertionError(f"unexpected fetch: {url}")


@dataclass
class GenericDetailFetcher:
    async def fetch(self, url: str, allowed_redirect_url=None) -> FetchedPage:
        if url == "http://szfb.sz.gov.cn/gkmlpt/content/11/11300/post_11300099.html":
            return FetchedPage(
                url=url,
                final_url=url,
                content_type="text/html",
                content="""
                <html>
                  <head><title>政府信息公开</title></head>
                  <body>
                    <h1>政府信息公开</h1>
                    <p>深圳市境外高端人才和紧缺人才2023纳税年度个人所得税财政补贴申报指南</p>
                    <p>发布日期：2024-03-29</p>
                  </body>
                </html>
                """.encode(),
            )
        raise AssertionError(f"unexpected fetch: {url}")


@dataclass
class NavigationLinkFetcher:
    async def fetch(self, url: str, allowed_redirect_url=None) -> FetchedPage:
        if url == "https://hrss.sz.gov.cn/links/":
            return FetchedPage(
                url=url,
                final_url=url,
                content_type="text/html",
                content=(
                    "<html><body>"
                    '<a href="/rsc/">人才科</a>'
                    '<a href="/rencai/">服务指南</a>'
                    '<a href="/talent-policy/2025-08-01-notice.html">深圳市青年人才引进支持办法</a>'
                    '<span>2025-08-01</span>'
                    "</body></html>"
                ).encode(),
            )
        if url == "https://hrss.sz.gov.cn/talent-policy/2025-08-01-notice.html":
            return FetchedPage(
                url=url,
                final_url=url,
                content_type="text/html",
                content="<html><body><p>政策支持。</p></body></html>".encode(),
            )
        raise AssertionError(f"unexpected fetch: {url}")


@pytest.mark.asyncio
async def test_pipeline_uses_broad_web_search_when_registry_has_no_match(registry):
    fetcher = WebOnlyFetcher()
    web_search = FakeWebSearch(
        urls=[
            "https://rsj.gz.gov.cn/ywzt/rcgz/renczc/",
            "https://gz.bendibao.com/news/rencai/",
        ]
    )
    pipeline = SearchPipeline(
        registry=registry,
        llm=NoRegistryLLM(),
        fetcher=fetcher,
        web_search=web_search,
        settings=Settings(MAX_WEB_SEARCH_QUERIES=1, _env_file=None),
    )

    response = await pipeline.search("广州")

    assert response.results_returned == 1
    assert response.results[0].official_url == "https://rsj.gz.gov.cn/ywzt/rcgz/renczc/"
    assert "https://gz.bendibao.com/news/rencai/" not in fetcher.fetched_urls
    assert response.official_sources_checked == 1
    assert any("broad official-domain web search" in warning.message for warning in response.warnings)
    assert web_search.queries
    assert "广州 人才政策" in web_search.queries[0]


@pytest.mark.asyncio
async def test_pipeline_skips_generic_web_search_results_before_fetching(registry):
    fetcher = WebListFetcher()
    pipeline = SearchPipeline(
        registry=registry,
        llm=OrganizerFailLLM(),
        fetcher=fetcher,
        web_search=GenericThenPolicyWebSearch(),
        settings=Settings(MAX_WEB_SEARCH_QUERIES=1, _env_file=None),
    )

    response = await pipeline.search("广州")

    assert response.results_returned == 1
    assert response.results[0].official_url.endswith("/content/post_1.html")
    assert "https://rsj.gz.gov.cn/" not in fetcher.fetched_urls


@pytest.mark.asyncio
async def test_pipeline_skips_generic_policy_file_web_search_results_without_talent_signal(registry):
    fetcher = PolicyWebSearchFetcher()
    pipeline = SearchPipeline(
        registry=registry,
        llm=OrganizerFailLLM(),
        fetcher=fetcher,
        web_search=GenericPolicyWebSearch(),
        settings=Settings(MAX_WEB_SEARCH_QUERIES=1, _env_file=None),
    )

    response = await pipeline.search("深圳")

    assert response.results_returned == 1
    assert response.results[0].title == "深圳市高层次人才奖励补贴申报指南"
    assert response.results[0].official_url == "https://hrss.sz.gov.cn/talent-policy/notice.html"
    assert fetcher.fetched_urls == ["https://hrss.sz.gov.cn/talent-policy/notice.html"]


@pytest.mark.asyncio
async def test_pipeline_extracts_broad_web_search_candidates_without_llm_judgment(registry):
    pipeline = SearchPipeline(
        registry=registry,
        llm=NoRegistryJudgeFailLLM(),
        fetcher=WebOnlyFetcher(),
        web_search=FakeWebSearch(urls=["https://rsj.gz.gov.cn/ywzt/rcgz/renczc/"]),
        settings=Settings(MAX_WEB_SEARCH_QUERIES=1, _env_file=None),
    )

    response = await pipeline.search("广州")

    assert response.results_returned == 1
    assert not any("candidate judgment failed" in warning.message.lower() for warning in response.warnings)


@pytest.mark.asyncio
async def test_pipeline_returns_code_extracted_cards_when_agent_organizer_fails(registry):
    pipeline = SearchPipeline(
        registry=registry,
        llm=OrganizerFailLLM(),
        fetcher=WebListFetcher(),
        web_search=FakeWebSearch(urls=["https://rsj.gz.gov.cn/ywzt/rcgz/renczc/"]),
        settings=Settings(MAX_WEB_SEARCH_QUERIES=1, _env_file=None),
    )

    response = await pipeline.search("广州")

    assert response.results_returned == 1
    assert response.results[0].title == "关于做好广州市境外人才财政补贴申报准备工作的通知"
    assert response.results[0].official_url == (
        "https://rsj.gz.gov.cn/ywzt/rcgz/renczc/content/post_1.html"
    )
    assert response.results[0].dates.published_date == "2025-03-12"
    assert response.results[0].evidence_snippets
    assert any("agent organizer failed" in warning.message.lower() for warning in response.warnings)


@pytest.mark.asyncio
async def test_pipeline_returns_markdown_summary_when_agent_markdown_fails(registry):
    pipeline = SearchPipeline(
        registry=registry,
        llm=MarkdownFailLLM(),
        fetcher=WebListFetcher(),
        web_search=FakeWebSearch(urls=["https://rsj.gz.gov.cn/ywzt/rcgz/renczc/"]),
        settings=Settings(MAX_WEB_SEARCH_QUERIES=1, _env_file=None),
    )

    response = await pipeline.search("广州")

    assert response.summary_markdown.startswith("## 广州 人才政策搜索摘要")
    assert "关于做好广州市境外人才财政补贴申报准备工作的通知" in response.summary_markdown
    assert "https://rsj.gz.gov.cn/ywzt/rcgz/renczc/content/post_1.html" in response.summary_markdown


@pytest.mark.asyncio
async def test_pipeline_uses_agent_markdown_summary_when_available(registry):
    pipeline = SearchPipeline(
        registry=registry,
        llm=MarkdownLLM(),
        fetcher=WebListFetcher(),
        web_search=FakeWebSearch(urls=["https://rsj.gz.gov.cn/ywzt/rcgz/renczc/"]),
        settings=Settings(MAX_WEB_SEARCH_QUERIES=1, _env_file=None),
    )

    response = await pipeline.search("广州")

    assert response.summary_markdown == "## Agent 汇总\n\n- 由 agent 整理的 Markdown。"


@pytest.mark.asyncio
async def test_pipeline_passes_code_extracted_page_evidence_to_agent_organizer(registry):
    llm = RecordingOrganizerLLM()
    pipeline = SearchPipeline(
        registry=registry,
        llm=llm,
        fetcher=WebListFetcher(),
        web_search=FakeWebSearch(urls=["https://rsj.gz.gov.cn/ywzt/rcgz/renczc/"]),
        settings=Settings(MAX_WEB_SEARCH_QUERIES=1, _env_file=None),
    )

    await pipeline.search("广州")

    assert llm.page_evidence is not None
    assert llm.page_evidence["title"] == "人才政策"
    assert llm.page_evidence["policy_items"][0]["title"] == (
        "关于做好广州市境外人才财政补贴申报准备工作的通知"
    )
    assert "fulltext" not in llm.page_evidence
    assert "fulltext_preview" in llm.page_evidence


@pytest.mark.asyncio
async def test_pipeline_fallback_uses_policy_like_fulltext_title_when_page_title_is_generic(
    registry,
):
    pipeline = SearchPipeline(
        registry=registry,
        llm=OrganizerFailLLM(),
        fetcher=GenericDetailFetcher(),
        web_search=FakeWebSearch(
            urls=["http://szfb.sz.gov.cn/gkmlpt/content/11/11300/post_11300099.html"]
        ),
        settings=Settings(MAX_WEB_SEARCH_QUERIES=1, _env_file=None),
    )

    response = await pipeline.search("深圳")

    assert response.results_returned == 1
    assert response.results[0].title == (
        "深圳市境外高端人才和紧缺人才2023纳税年度个人所得税财政补贴申报指南"
    )
    assert response.results[0].dates.published_date == "2024-03-29"


@pytest.mark.asyncio
async def test_pipeline_fallback_drops_navigation_links_when_organizer_unavailable(registry):
    pipeline = SearchPipeline(
        registry=registry,
        llm=OrganizerFailLLM(),
        fetcher=NavigationLinkFetcher(),
        web_search=FakeWebSearch(
            urls=["https://hrss.sz.gov.cn/links/"],
        ),
        settings=Settings(MAX_WEB_SEARCH_QUERIES=1, _env_file=None),
    )

    response = await pipeline.search("深圳")

    assert response.results_returned == 1
    assert response.results[0].title == "深圳市青年人才引进支持办法"
    assert response.results[0].official_url == (
        "https://hrss.sz.gov.cn/talent-policy/2025-08-01-notice.html"
    )


@dataclass
class CoordinatedFetcher:
    active_fetches: int = 0
    concurrent_fetch_seen: asyncio.Event = field(default_factory=asyncio.Event)

    async def fetch(self, url: str, allowed_redirect_url=None) -> FetchedPage:
        self.active_fetches += 1
        if self.active_fetches >= 2:
            self.concurrent_fetch_seen.set()
        try:
            await asyncio.wait_for(self.concurrent_fetch_seen.wait(), timeout=0.2)
            return FetchedPage(
                url=url,
                final_url=url,
                content_type="text/html",
                content=f"""
                <html><body>
                  <h1>{url.rsplit('/', 1)[-1]} 人才补贴申报指南</h1>
                  <p>2025-03-12 高层次人才补贴。</p>
                </body></html>
                """.encode(),
            )
        finally:
            self.active_fetches -= 1


@pytest.mark.asyncio
async def test_pipeline_processes_multiple_candidate_urls_concurrently(registry):
    fetcher = CoordinatedFetcher()
    pipeline = SearchPipeline(
        registry=registry,
        llm=OrganizerFailLLM(),
        fetcher=fetcher,
        web_search=FakeWebSearch(
            urls=[
                "https://rsj.gz.gov.cn/policy-a.html",
                "https://rsj.gz.gov.cn/policy-b.html",
            ]
        ),
        settings=Settings(
            MAX_WEB_SEARCH_QUERIES=1,
            MAX_EXTRACTED_PAGES=2,
            MAX_CONCURRENT_EXTRACTIONS=2,
            _env_file=None,
        ),
    )

    response = await pipeline.search("广州")

    assert response.results_returned == 2


@dataclass
class RegistryAndWebFetcher:
    fetched_urls: list[str] = field(default_factory=list)

    async def fetch(self, url: str, allowed_redirect_url=None) -> FetchedPage:
        self.fetched_urls.append(url)
        if url.endswith("/robots.txt"):
            return FetchedPage(url=url, final_url=url, content_type="text/plain", content=b"")
        if url in {
            "https://www.sz.gov.cn/",
            "https://hrss.sz.gov.cn/",
            "https://stic.sz.gov.cn/",
        }:
            return FetchedPage(
                url=url,
                final_url=url,
                content_type="text/html",
                content=b"<html><body></body></html>",
            )
        if url == "http://www.sz.gov.cn/zfgb/2022/gb1236/content/post_9684588.html":
            return FetchedPage(
                url=url,
                final_url=url,
                content_type="text/html",
                content="<html><body><p>深圳高层次人才奖励补贴。</p></body></html>".encode(),
            )
        raise AssertionError(f"unexpected fetch: {url}")


@pytest.mark.asyncio
async def test_pipeline_adds_broad_web_search_candidates_for_registry_match(registry):
    fetcher = RegistryAndWebFetcher()
    pipeline = SearchPipeline(
        registry=registry,
        llm=FakeLLM(),
        fetcher=fetcher,
        web_search=FakeWebSearch(
            urls=["http://www.sz.gov.cn/zfgb/2022/gb1236/content/post_9684588.html"]
        ),
        settings=Settings(MAX_WEB_SEARCH_QUERIES=1, _env_file=None),
    )

    response = await pipeline.search("深圳")

    assert response.results_returned == 1
    assert response.results[0].official_url == (
        "http://www.sz.gov.cn/zfgb/2022/gb1236/content/post_9684588.html"
    )
    assert "http://www.sz.gov.cn/zfgb/2022/gb1236/content/post_9684588.html" in fetcher.fetched_urls


@dataclass
class BadEcpointFetcher:
    fetched_urls: list[str] = field(default_factory=list)

    async def fetch(self, url: str, allowed_redirect_url=None) -> FetchedPage:
        self.fetched_urls.append(url)
        if url.startswith("https://www.sz.gov.cn"):
            raise RuntimeError("[SSL: BAD_ECPOINT] bad ecpoint")
        if url == "http://www.sz.gov.cn/robots.txt":
            return FetchedPage(url=url, final_url=url, content_type="text/plain", content=b"")
        if url == "http://www.sz.gov.cn/":
            return FetchedPage(
                url=url,
                final_url=url,
                content_type="text/html",
                content=b'<a href="http://www.sz.gov.cn/talent-policy.html">policy</a>',
            )
        if url == "http://www.sz.gov.cn/talent-policy.html":
            return FetchedPage(
                url=url,
                final_url=url,
                content_type="text/html",
                content="<html><body><p>深圳高层次人才奖励补贴。</p></body></html>".encode(),
            )
        if url.endswith("/robots.txt"):
            return FetchedPage(url=url, final_url=url, content_type="text/plain", content=b"")
        return FetchedPage(
            url=url,
            final_url=url,
            content_type="text/html",
            content=b"<html><body></body></html>",
        )


@pytest.mark.asyncio
async def test_pipeline_fetches_http_fallback_for_https_bad_ecpoint(registry):
    fetcher = BadEcpointFetcher()
    pipeline = SearchPipeline(
        registry=registry,
        llm=FakeLLM(),
        fetcher=fetcher,
        web_search=NO_WEB_SEARCH,
        settings=Settings(MAX_SEED_URLS_PER_ENTITY=1, _env_file=None),
    )

    response = await pipeline.search("深圳")

    assert response.results_returned == 1
    assert response.results[0].official_url == "http://www.sz.gov.cn/talent-policy.html"
    assert "http://www.sz.gov.cn/" in fetcher.fetched_urls
    assert any("HTTP fallback" in warning.message for warning in response.warnings)


@pytest.mark.asyncio
async def test_pipeline_defaults_missing_extracted_policy_url_from_page(registry):
    pipeline = SearchPipeline(
        registry=registry,
        llm=MissingUrlLLM(),
        fetcher=FakeFetcher(),
        web_search=NO_WEB_SEARCH,
    )

    response = await pipeline.search("深圳")

    assert response.results_returned == 1
    assert response.results[0].official_url == "https://hrss.sz.gov.cn/talent-policy.html"
    assert response.results[0].matched_entity == "深圳"
    assert response.results[0].source_name == "深圳"


@pytest.mark.asyncio
async def test_pipeline_normalizes_common_llm_policy_field_aliases(registry):
    pipeline = SearchPipeline(
        registry=registry,
        llm=AliasFieldLLM(),
        fetcher=FakeFetcher(),
        web_search=NO_WEB_SEARCH,
    )

    response = await pipeline.search("深圳")

    assert response.results_returned == 1
    assert response.results[0].title == "广州市境外职业资格比照对应职称目录"
    assert response.results[0].source_type == "government"
    assert response.results[0].policy_types == ["职称评审"]
    assert response.results[0].applicable_to == ["境外专业人才"]
    assert response.results[0].benefits[0].amount_text == "职称比照认定支持"
    assert response.results[0].eligibility[0].condition == "符合目录对应职业资格"
    assert response.results[0].application.process == "在线申报"
    assert response.results[0].evidence_snippets == ["官方目录发布"]
    assert response.results[0].dates.published_date == "2025-12-22"
    assert response.results[0].confidence == 0.85
    assert response.results[0].completeness == 0.6


@dataclass
class RedirectPolicyFetcher:
    saw_redirect_policy: bool = False
    disallowed_redirect_blocked: bool = False

    async def fetch(self, url: str, allowed_redirect_url=None) -> FetchedPage:
        if url.endswith("/robots.txt"):
            return FetchedPage(url=url, final_url=url, content_type="text/plain", content=b"")
        if url == "https://hrss.sz.gov.cn/":
            return FetchedPage(
                url=url,
                final_url=url,
                content_type="text/html",
                content=b'<a href="https://hrss.sz.gov.cn/redirect.html">policy</a>',
            )
        if url == "https://hrss.sz.gov.cn/redirect.html":
            self.saw_redirect_policy = allowed_redirect_url is not None
            if allowed_redirect_url and not allowed_redirect_url("https://evil.example/secret"):
                self.disallowed_redirect_blocked = True
                raise RuntimeError("Disallowed redirect target")
            return FetchedPage(
                url=url,
                final_url="https://evil.example/secret",
                content_type="text/html",
                content=b"<html><body>secret policy body should not be used</body></html>",
            )
        return FetchedPage(
            url=url,
            final_url=url,
            content_type="text/html",
            content=b'<a href="https://hrss.sz.gov.cn/redirect.html">policy</a>',
        )


@pytest.mark.asyncio
async def test_pipeline_passes_entity_redirect_policy_to_fetcher(registry):
    fetcher = RedirectPolicyFetcher()
    pipeline = SearchPipeline(
        registry=registry,
        llm=FakeLLM(),
        fetcher=fetcher,
        web_search=NO_WEB_SEARCH,
    )

    response = await pipeline.search("深圳")

    assert fetcher.saw_redirect_policy is True
    assert fetcher.disallowed_redirect_blocked is True
    assert response.results == []
    assert any("disallowed redirect" in warning.message.lower() for warning in response.warnings)


@dataclass
class ManyCandidateFetcher:
    async def fetch(self, url: str) -> FetchedPage:
        if url.endswith("/robots.txt"):
            return FetchedPage(url=url, final_url=url, content_type="text/plain", content=b"")
        if url == "https://hrss.sz.gov.cn/":
            return FetchedPage(
                url=url,
                final_url=url,
                content_type="text/html",
                content=b"""
                <html><body>
                  <a href="https://hrss.sz.gov.cn/talent-policy-a.html">a</a>
                  <a href="https://hrss.sz.gov.cn/talent-policy-b.html">b</a>
                </body></html>
                """,
            )
        return FetchedPage(
            url=url,
            final_url=url,
            content_type="text/html",
            content=(
                "<html><body><p>给予高层次人才奖励补贴。</p></body></html>"
            ).encode(),
        )


@pytest.mark.asyncio
async def test_pipeline_candidate_pages_seen_counts_before_candidate_cap(registry):
    pipeline = SearchPipeline(
        registry=registry,
        llm=FakeLLM(),
        fetcher=ManyCandidateFetcher(),
        web_search=NO_WEB_SEARCH,
        settings=Settings(MAX_CANDIDATE_PAGES=1, _env_file=None),
    )

    response = await pipeline.search("深圳")

    assert response.candidate_pages_seen > 1


@dataclass
class FeedFailureFetcher:
    async def fetch(self, url: str) -> FetchedPage:
        if url.endswith("/robots.txt"):
            return FetchedPage(url=url, final_url=url, content_type="text/plain", content=b"")
        if url == "https://hrss.sz.gov.cn/":
            return FetchedPage(
                url=url,
                final_url=url,
                content_type="text/html",
                content=b"""
                <html>
                  <head>
                    <link rel="alternate" type="application/rss+xml" href="/feed.xml">
                  </head>
                  <body>
                    <a href="https://hrss.sz.gov.cn/talent-policy.html">policy</a>
                  </body>
                </html>
                """,
            )
        if url == "https://hrss.sz.gov.cn/feed.xml":
            return FetchedPage(
                url=url,
                final_url=url,
                content_type="application/rss+xml",
                content=b"bad feed",
            )
        return FetchedPage(
            url=url,
            final_url=url,
            content_type="text/html",
            content=(
                "<html><body><p>给予高层次人才奖励补贴。</p></body></html>"
            ).encode(),
        )


@pytest.mark.asyncio
async def test_pipeline_degrades_feed_parse_failures_to_warnings(registry, monkeypatch):
    def fail_feed_parse(feed_text: str):
        raise RuntimeError("feed parser failed with large payload hidden")

    monkeypatch.setattr("talent_policy_search.pipeline.parse_feed_urls", fail_feed_parse)
    pipeline = SearchPipeline(
        registry=registry,
        llm=FakeLLM(),
        fetcher=FeedFailureFetcher(),
        web_search=NO_WEB_SEARCH,
    )

    response = await pipeline.search("深圳")

    assert response.results_returned == 1
    assert any("Feed parsing failed" in warning.message for warning in response.warnings)


@dataclass
class FormCandidateFetcher:
    async def fetch(self, url: str) -> FetchedPage:
        if url.endswith("/robots.txt"):
            return FetchedPage(url=url, final_url=url, content_type="text/plain", content=b"")
        if url == "https://www.sz.gov.cn/":
            return FetchedPage(
                url=url,
                final_url=url,
                content_type="text/html",
                content=b"""
                <html><body>
                  <form action="/search.html" method="get">
                    <input type="search" name="q">
                  </form>
                </body></html>
                """,
            )
        if url.startswith("https://www.sz.gov.cn/search.html?"):
            return FetchedPage(
                url=url,
                final_url=url,
                content_type="text/html",
                content=(
                    "<html><body><p>给予高层次人才奖励补贴。</p></body></html>"
                ).encode(),
            )
        raise AssertionError(f"unexpected fetch: {url}")


@dataclass
class LinkCandidateFetcher:
    async def fetch(self, url: str) -> FetchedPage:
        if url.endswith("/robots.txt"):
            return FetchedPage(url=url, final_url=url, content_type="text/plain", content=b"")
        if url == "https://www.sz.gov.cn/":
            return FetchedPage(
                url=url,
                final_url=url,
                content_type="text/html",
                content=b"""
                <html><body>
                  <a href="https://www.sz.gov.cn/talent-policy.html">policy</a>
                </body></html>
                """,
            )
        if url == "https://www.sz.gov.cn/talent-policy.html":
            return FetchedPage(
                url=url,
                final_url=url,
                content_type="text/html",
                content=(
                    "<html><body><p>给予高层次人才奖励补贴。</p></body></html>"
                ).encode(),
            )
        raise AssertionError(f"unexpected fetch: {url}")


def _single_seed_settings() -> Settings:
    return Settings(MAX_SEED_URLS_PER_ENTITY=1, _env_file=None)


def _assert_bounded_warning(response, prefix: str) -> None:
    warning = next(
        (
            warning
            for warning in response.warnings
            if warning.message.startswith(prefix)
        ),
        None,
    )
    assert warning is not None
    assert len(warning.message) <= pipeline_module.MAX_WARNING_TEXT
    assert HTML_PARSE_FAILURE_SECRET not in warning.message


@pytest.mark.asyncio
async def test_pipeline_degrades_html_link_extraction_failures_to_search_form_candidates(
    registry,
    monkeypatch,
):
    def fail_extract_links(html_text: str, base_url: str):
        raise RuntimeError(f"link parser saw {HTML_PARSE_FAILURE_SECRET}")

    monkeypatch.setattr(pipeline_module, "extract_links", fail_extract_links)
    pipeline = SearchPipeline(
        registry=registry,
        llm=FakeLLM(),
        fetcher=FormCandidateFetcher(),
        web_search=NO_WEB_SEARCH,
        settings=_single_seed_settings(),
    )

    response = await pipeline.search("深圳")

    assert response.results_returned == 1
    assert response.results[0].official_url.startswith("https://www.sz.gov.cn/search.html?")
    _assert_bounded_warning(response, "HTML link extraction failed")


@pytest.mark.asyncio
async def test_pipeline_degrades_html_feed_link_extraction_failures_to_page_links(
    registry,
    monkeypatch,
):
    def fail_extract_feed_links(html_text: str, base_url: str):
        raise RuntimeError(f"feed link parser saw {HTML_PARSE_FAILURE_SECRET}")

    monkeypatch.setattr(pipeline_module, "extract_feed_links", fail_extract_feed_links)
    pipeline = SearchPipeline(
        registry=registry,
        llm=FakeLLM(),
        fetcher=LinkCandidateFetcher(),
        web_search=NO_WEB_SEARCH,
        settings=_single_seed_settings(),
    )

    response = await pipeline.search("深圳")

    assert response.results_returned == 1
    assert response.results[0].official_url == "https://www.sz.gov.cn/talent-policy.html"
    _assert_bounded_warning(response, "HTML feed link extraction failed")


@pytest.mark.asyncio
async def test_pipeline_degrades_html_search_form_discovery_failures_to_page_links(
    registry,
    monkeypatch,
):
    def fail_detect_get_search_forms(html_text: str, base_url: str, terms: list[str]):
        raise RuntimeError(f"search form parser saw {HTML_PARSE_FAILURE_SECRET}")

    monkeypatch.setattr(
        pipeline_module,
        "detect_get_search_forms",
        fail_detect_get_search_forms,
    )
    pipeline = SearchPipeline(
        registry=registry,
        llm=FakeLLM(),
        fetcher=LinkCandidateFetcher(),
        web_search=NO_WEB_SEARCH,
        settings=_single_seed_settings(),
    )

    response = await pipeline.search("深圳")

    assert response.results_returned == 1
    assert response.results[0].official_url == "https://www.sz.gov.cn/talent-policy.html"
    _assert_bounded_warning(response, "HTML search form discovery failed")


@pytest.mark.asyncio
async def test_pipeline_degrades_search_page_link_extraction_failures(registry, monkeypatch):
    original_extract_links = pipeline_module.extract_links

    def fail_search_page_extract_links(html_text: str, base_url: str):
        if base_url.startswith("https://www.sz.gov.cn/search.html?"):
            raise RuntimeError(f"search page link parser saw {HTML_PARSE_FAILURE_SECRET}")
        return original_extract_links(html_text, base_url)

    monkeypatch.setattr(pipeline_module, "extract_links", fail_search_page_extract_links)
    pipeline = SearchPipeline(
        registry=registry,
        llm=FakeLLM(),
        fetcher=FormCandidateFetcher(),
        web_search=NO_WEB_SEARCH,
        settings=_single_seed_settings(),
    )

    response = await pipeline.search("深圳")

    assert response.results_returned == 1
    assert response.results[0].official_url.startswith("https://www.sz.gov.cn/search.html?")
    _assert_bounded_warning(response, "HTML link extraction failed")


def test_rank_policies_scores_and_sorts_policy_cards():
    older = PolicyCard(
        title="Older",
        official_url="https://hrss.sz.gov.cn/old.html",
        confidence=0.4,
        completeness=0.4,
        dates={"published_date": "2020-01-01"},
        evidence_snippets=[],
    )
    newer = PolicyCard(
        title="Newer",
        official_url="https://hrss.sz.gov.cn/new.html",
        confidence=0.9,
        completeness=0.8,
        dates={"published_date": "2025-03-01"},
        evidence_snippets=["snippet"],
    )

    ranked = rank_policies([older, newer])

    assert [policy.title for policy in ranked] == ["Newer", "Older"]
    assert ranked[0].score > ranked[1].score > 0


def test_pipeline_final_ranking_drops_generic_navigation_policy_cards(registry):
    pipeline = SearchPipeline(registry=registry, llm=FakeLLM(), fetcher=FakeFetcher())
    generic = PolicyCard(title="政务公开", official_url="https://www.sz.gov.cn/?dw=zwgk")
    generic_column = PolicyCard(title="政府公报", official_url="https://www.sz.gov.cn/zfgb/")
    policy = PolicyCard(
        title="深圳市境外高端人才个税补贴申报指南",
        official_url="https://www.sz.gov.cn/policy.html",
    )

    results = pipeline._rank_and_dedupe_policies([generic, generic_column, policy])

    assert [result.title for result in results] == ["深圳市境外高端人才个税补贴申报指南"]


def test_pipeline_policy_title_looks_like_requires_talent_signal():
    assert pipeline_module._looks_like_policy_title("深圳市人才政策") is False
    assert pipeline_module._looks_like_policy_title("深圳市高层次人才奖励补贴申报指南") is True
    assert pipeline_module._looks_like_policy_title("政务公开 政策文件库") is False


def test_pipeline_low_value_policy_filter_drops_generic_non_talent_cards():
    title = "惠企纾困政策"
    policy = PolicyCard(
        title=title,
        official_url="https://www.sz.gov.cn/news/notice.html",
        confidence=0.66,
        completeness=0.42,
        dates={"published_date": "2025-03-01"},
        evidence_snippets=[title],
    )
    assert pipeline_module._is_low_value_policy(policy)


def test_bad_ecpoint_classifier_catches_ssl_handshake_errors():
    assert pipeline_module._looks_like_bad_ecpoint_error(
        RuntimeError("[SSL: SSLV3_ALERT_HANDSHAKE_FAILURE] ssl/tls alert handshake failure")
    )
    assert pipeline_module._looks_like_bad_ecpoint_error(
        RuntimeError("[SSL: BAD_ECPOINT] bad ecpoint")
    )
    assert not pipeline_module._looks_like_bad_ecpoint_error(
        RuntimeError("normal connect reset")
    )
