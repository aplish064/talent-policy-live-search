from __future__ import annotations

import asyncio
import inspect
import hashlib
import html
import re
import time
from collections.abc import Iterable
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any
from urllib.parse import unquote_plus, urlparse, urlunparse

from pydantic import ValidationError

from talent_policy_search.config import Settings, get_settings
from talent_policy_search.discovery import (
    detect_get_search_forms,
    extract_feed_links,
    extract_links,
    parse_feed_urls,
    parse_robots_sitemaps,
    parse_sitemap_urls,
    rank_candidate_urls,
)
from talent_policy_search.domain_filter import DomainFilter
from talent_policy_search.extractor import (
    TALENT_POLICY_SIGNALS,
    PageExtraction,
    extract_page_info,
)
from talent_policy_search.fetcher import FetchedPage, Fetcher
from talent_policy_search.llm import LLMClient, LLMUnavailableError
from talent_policy_search.models import (
    OfficialEntity,
    PolicyCard,
    SearchRequest,
    SearchResponse,
    SearchWarning,
)
from talent_policy_search.ranker import rank_policies
from talent_policy_search.sources import SourceRegistry
from talent_policy_search.web_search import AnySearchProvider, NullWebSearchProvider


QUERY_IDENTIFICATION_PROMPT = """
Identify which official-source registry entities match the user's talent policy search query.
Return only JSON with normalized_query, query_type, matched_entity_ids, and confidence.
Prefer exact registry entity IDs from the supplied entity list.
""".strip()

DISCOVERY_TERMS_PROMPT = """
Generate concise discovery terms for official talent policy pages for one entity.
Return only JSON with terms as objects containing term, intent, language, priority,
and url_path_hints as short URL path keywords.
""".strip()

CANDIDATE_JUDGMENT_PROMPT = """
Judge whether an already fetched official candidate page is relevant to the query.
Return only JSON with is_relevant, is_official, content_kind, policy_types,
should_extract_fulltext, and confidence. Do not invent facts.
""".strip()

POLICY_EXTRACTION_PROMPT = """
Organize already extracted official webpage evidence into structured talent policy cards.
The code has already extracted page title, links, policy list items, dates, attachments,
application links, and text evidence. Do not perform raw webpage extraction and do not
invent facts outside the supplied evidence.
Return only JSON with policies as an array and discard_reason as a short string or null.
Every policy official_url must be a safe absolute http(s) URL from the supplied evidence.
Each policy must include title, official_url, source_name, matched_entity, policy_types,
applicable_to, benefits, eligibility, application, dates, evidence_snippets, summary,
confidence, and completeness. Put published_date/effective_date/deadline/valid_until
inside dates.
""".strip()

MARKDOWN_SUMMARY_PROMPT = """
Organize the supplied official talent policy search results into concise Chinese Markdown.
Do not invent facts outside the supplied results. Prefer the style of a policy search report:
title, category/type, publish date when available, one-sentence summary, source link, and
notable dimensions such as benefit, applicable population, application entry, or materials.
Return Markdown text only, no JSON and no code fences.
""".strip()

MAX_WARNING_COUNT = 25
MAX_WARNING_TEXT = 180
MAX_WARNING_SOURCE = 120
MAX_FEED_LINKS_PER_PAGE = 10
MAX_SEARCH_FORM_URLS_PER_PAGE = 10
MAX_SITEMAP_FILES_PER_ENTITY = 20
MIN_FULLTEXT_CHARS = 8
SOURCE_TYPES = {
    "government",
    "university",
    "research_agency",
    "immigration_agency",
    "public_service_portal",
}
SOURCE_TYPE_BY_ENTITY_TYPE = {
    "region": "government",
    "country_or_jurisdiction": "government",
    "agency": "government",
    "university": "university",
}
GENERIC_PAGE_TITLES = {
    "首页",
    "政务公开",
    "政府信息公开",
    "政策法规",
    "政策解读",
    "通知公告",
    "人才政策",
    "政府公报",
    "人才科",
    "人事科",
    "师资科",
    "劳资科",
    "综合科",
    "师德师风",
    "人才招聘",
    "服务指南",
    "下载专区",
    "政策文件",
    "政策文件库",
    "解读回应",
    "新闻动态",
    "网站首页",
    "学校首页",
}
POLICY_TITLE_KEYWORDS = (
    "政策",
    "通知",
    "指南",
    "办法",
    "措施",
    "申报",
    "补贴",
    "津贴",
    "人才",
    "博士后",
    "资助",
    "奖励",
    "policy",
    "talent",
    "faculty",
    "目录",
    "hiring",
    "recruitment",
    "fellowship",
    "grant",
    "benefit",
)
TALENT_POLICY_SIGNAL_KEYWORDS = (
    *TALENT_POLICY_SIGNALS,
    "talent",
    "recruitment",
    "faculty",
    "hiring",
    "postdoctoral",
    "postdoc",
    "postgraduate",
    "fellowship",
    "目录",
    "职称",
    "资格",
    "名单",
)


class LLMOrchestrator:
    def __init__(self, client: LLMClient | None = None) -> None:
        self._client = client or LLMClient()

    async def identify_query(self, query: str, entities: list[dict[str, Any]]) -> dict[str, Any]:
        return await self._client.complete_json(
            QUERY_IDENTIFICATION_PROMPT,
            {"query": query, "entities": entities},
            max_tokens=1024,
        )

    async def generate_discovery_terms(
        self,
        query: str,
        entity: dict[str, Any],
    ) -> dict[str, Any]:
        generated = await self._client.complete_json_value(
            DISCOVERY_TERMS_PROMPT,
            {"query": query, "entity": entity},
            max_tokens=2048,
        )
        if isinstance(generated, list):
            return {"terms": generated, "url_path_hints": []}
        if isinstance(generated, dict):
            return generated
        raise LLMUnavailableError(f"Expected JSON object or list, got {type(generated).__name__}")

    async def judge_candidate(
        self,
        query: dict[str, Any],
        candidate: dict[str, Any],
    ) -> dict[str, Any]:
        return await self._client.complete_json(
            CANDIDATE_JUDGMENT_PROMPT,
            {"query": query, "candidate": candidate},
            max_tokens=1024,
        )

    async def extract_policies(
        self,
        source: dict[str, Any],
        query_context: dict[str, Any],
        page_evidence: dict[str, Any],
    ) -> dict[str, Any]:
        extracted = await self._client.complete_json_value(
            POLICY_EXTRACTION_PROMPT,
            {
                "source": source,
                "query_context": query_context,
                "page_evidence": page_evidence,
            },
            max_tokens=4096,
        )
        if isinstance(extracted, list):
            return {"policies": extracted, "discard_reason": None}
        if isinstance(extracted, dict):
            return extracted
        raise LLMUnavailableError(f"Expected JSON object or list, got {type(extracted).__name__}")

    async def summarize_markdown(
        self,
        query_context: dict[str, Any],
        policies: list[dict[str, Any]],
    ) -> str:
        response = await self._client.complete_text(
            MARKDOWN_SUMMARY_PROMPT,
            {"query_context": query_context, "policies": policies},
            max_tokens=4096,
        )
        return response.strip()


class SearchPipeline:
    def __init__(
        self,
        registry: SourceRegistry,
        llm: Any | None = None,
        fetcher: Any | None = None,
        web_search: Any | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.registry = registry
        self.settings = settings or get_settings()
        self.settings.assert_no_persistence()
        self.domain_filter = DomainFilter(registry)
        self.llm = llm or LLMOrchestrator(LLMClient(self.settings))
        self.fetcher = fetcher or Fetcher(self.settings)
        self.web_search = web_search or (
            AnySearchProvider(self.settings)
            if self.settings.enable_web_search
            else NullWebSearchProvider()
        )
        self._fetcher_accepts_redirect_policy = _fetcher_accepts_redirect_policy(self.fetcher)

    async def search(self, query: str) -> SearchResponse:
        started_at = datetime.now(timezone.utc)
        start_monotonic = time.monotonic()
        request = SearchRequest(query=query)
        warnings: list[SearchWarning] = []
        checked_urls: set[str] = set()
        checked_seed_urls: set[str] = set()
        candidate_urls_seen: set[str] = set()

        normalized_query, query_info, entities = await self._identify_entities(
            request.query,
            warnings,
        )
        if not entities:
            entities = [_synthetic_entity(request.query, normalized_query)]
            _add_warning(
                warnings,
                "web_search",
                "No registry entity matched; using broad official-domain web search.",
            )

        policies: list[PolicyCard] = []
        extracted_pages = 0
        for entity in entities:
            terms, path_hints = await self._discovery_terms(request.query, entity, warnings)
            web_candidates = await self._discover_web_search_candidates(
                query=request.query,
                normalized_query=normalized_query,
                entity=entity,
                terms=terms,
                warnings=warnings,
            )
            filtered_web_candidates = self._allowed_for_entity(web_candidates, entity)
            if filtered_web_candidates:
                filtered_site_candidates: list[str] = []
            else:
                site_candidates = await self._discover_entity_candidates(
                    entity=entity,
                    terms=terms,
                    warnings=warnings,
                    checked_urls=checked_urls,
                    checked_seed_urls=checked_seed_urls,
                )
                filtered_site_candidates = self._allowed_for_entity(site_candidates, entity)
            filtered_candidates = _dedupe_strings(
                [*filtered_web_candidates, *filtered_site_candidates]
            )
            web_candidate_set = set(filtered_web_candidates)
            checked_seed_urls.update(filtered_web_candidates)
            candidate_urls_seen.update(filtered_candidates)

            ranked_site_candidates = rank_candidate_urls(
                filtered_site_candidates,
                terms=terms,
                path_hints=path_hints,
                limit=self.settings.max_candidate_pages,
            )
            ranked_candidates = _dedupe_strings(
                [*filtered_web_candidates, *ranked_site_candidates]
            )
            if not ranked_candidates:
                ranked_candidates = filtered_candidates[: self.settings.max_candidate_pages]
            else:
                ranked_candidates = ranked_candidates[: self.settings.max_candidate_pages]

            query_context = self._query_context(
                query=request.query,
                normalized_query=normalized_query,
                query_info=query_info,
                entity=entity,
                terms=terms,
                path_hints=path_hints,
            )
            cursor = 0
            while extracted_pages < self.settings.max_extracted_pages and cursor < len(
                ranked_candidates
            ):
                remaining_pages = self.settings.max_extracted_pages - extracted_pages
                batch_size = min(
                    self.settings.max_concurrent_extractions,
                    remaining_pages,
                    len(ranked_candidates) - cursor,
                )
                batch = ranked_candidates[cursor : cursor + batch_size]
                cursor += batch_size
                batch_results = await asyncio.gather(
                    *[
                        self._process_candidate_url(
                            candidate_url=candidate_url,
                            entity=entity,
                            query_context=query_context,
                            web_candidate_set=web_candidate_set,
                            warnings=warnings,
                            checked_urls=checked_urls,
                        )
                        for candidate_url in batch
                    ]
                )
                for page_policies, did_extract_page in batch_results:
                    if did_extract_page:
                        extracted_pages += 1
                    policies.extend(page_policies)

            if extracted_pages >= self.settings.max_extracted_pages:
                break

        results = self._rank_and_dedupe_policies(policies)
        summary_markdown = await self._summarize_results_markdown(
            request.query,
            normalized_query,
            query_info,
            results,
            warnings,
        )
        return self._response(
            query=request.query,
            normalized_query=normalized_query,
            started_at=started_at,
            start_monotonic=start_monotonic,
            checked_seed_urls=checked_seed_urls,
            candidate_pages_seen=len(candidate_urls_seen),
            summary_markdown=summary_markdown,
            warnings=warnings,
            results=results,
        )

    async def _identify_entities(
        self,
        query: str,
        warnings: list[SearchWarning],
    ) -> tuple[str, dict[str, Any], list[OfficialEntity]]:
        registry_matches = self.registry.match_entity(query)
        entity_payload = [_entity_payload(entity) for entity in self.registry.entities]
        query_info: dict[str, Any] = {}

        try:
            query_info = await self.llm.identify_query(query, entity_payload)
        except Exception as exc:
            _add_warning(
                warnings,
                "llm.identify_query",
                _short_error("LLM query matching failed", exc),
            )

        normalized_query = _safe_text(query_info.get("normalized_query")) or query
        by_id = {entity.id: entity for entity in self.registry.entities}
        llm_matches: list[OfficialEntity] = []
        matched_entity_ids = query_info.get("matched_entity_ids")
        if isinstance(matched_entity_ids, list):
            for entity_id in matched_entity_ids:
                if isinstance(entity_id, str) and entity_id in by_id:
                    llm_matches.append(by_id[entity_id])

        return normalized_query, query_info, _dedupe_entities([*llm_matches, *registry_matches])

    async def _discovery_terms(
        self,
        query: str,
        entity: OfficialEntity,
        warnings: list[SearchWarning],
    ) -> tuple[list[str], list[str]]:
        try:
            generated = await self.llm.generate_discovery_terms(query, _entity_payload(entity))
            terms = _terms_from_generated(generated, self.settings.max_llm_planned_queries)
            path_hints = _strings_from_list(generated.get("url_path_hints"), limit=20)
            if terms:
                return terms, path_hints
            _add_warning(
                warnings,
                entity.id,
                "LLM discovery terms were empty; using registry focus terms.",
            )
        except Exception as exc:
            _add_warning(
                warnings,
                "llm.generate_discovery_terms",
                _short_error("LLM discovery term generation failed", exc),
            )

        fallback_terms = _dedupe_strings(entity.focus_terms or entity.searchable_names)
        return fallback_terms[: self.settings.max_llm_planned_queries], []

    async def _discover_entity_candidates(
        self,
        entity: OfficialEntity,
        terms: list[str],
        warnings: list[SearchWarning],
        checked_urls: set[str],
        checked_seed_urls: set[str],
    ) -> list[str]:
        candidates: list[str] = []
        seed_urls = [
            str(url) for url in entity.seed_urls[: self.settings.max_seed_urls_per_entity]
        ]
        for seed_url in seed_urls:
            if not self.domain_filter.is_allowed_for_entity(seed_url, entity):
                _add_warning(
                    warnings,
                    entity.id,
                    f"Skipped disallowed seed URL: {_trim(seed_url)}",
                )
                continue

            candidates.append(seed_url)
            robots_url = _robots_url(seed_url)
            if robots_url and self.domain_filter.is_allowed_for_entity(robots_url, entity):
                robots_page = await self._fetch_optional(
                    robots_url,
                    entity,
                    warnings,
                    checked_urls,
                )
                if robots_page and self.domain_filter.is_allowed_for_entity(
                    robots_page.final_url,
                    entity,
                ):
                    candidates.extend(
                        await self._discover_sitemap_candidates(
                            entity=entity,
                            sitemap_urls=parse_robots_sitemaps(_decode_page_text(robots_page)),
                            warnings=warnings,
                            checked_urls=checked_urls,
                        )
                    )

            checked_seed_urls.add(seed_url)
            seed_page = await self._fetch_optional(seed_url, entity, warnings, checked_urls)
            if seed_page is None:
                continue
            if not self.domain_filter.is_allowed_for_entity(seed_page.final_url, entity):
                _add_warning(
                    warnings,
                    seed_url,
                    "Seed URL redirected outside the allowed entity domain.",
                )
                continue
            candidates.extend(
                await self._discover_from_page(
                    entity=entity,
                    page=seed_page,
                    terms=terms,
                    warnings=warnings,
                    checked_urls=checked_urls,
                )
            )

        return _dedupe_strings(candidates)

    async def _discover_web_search_candidates(
        self,
        query: str,
        normalized_query: str,
        entity: OfficialEntity,
        terms: list[str],
        warnings: list[SearchWarning],
    ) -> list[str]:
        candidates: list[str] = []
        for search_query in _web_search_queries(
            query=query,
            normalized_query=normalized_query,
            entity=entity,
            terms=terms,
            limit=self.settings.max_web_search_queries,
        ):
            try:
                results = await self.web_search.search(
                    search_query,
                    max_results=self.settings.max_web_search_results,
                )
            except Exception as exc:
                _add_warning(
                    warnings,
                    "web_search",
                    _short_error("Web search failed", exc),
                )
                continue
            candidates.extend(
                result.url for result in results if _web_search_result_looks_policy_like(result)
            )
        return _dedupe_strings(candidates)

    async def _discover_sitemap_candidates(
        self,
        entity: OfficialEntity,
        sitemap_urls: Iterable[str],
        warnings: list[SearchWarning],
        checked_urls: set[str],
    ) -> list[str]:
        candidates: list[str] = []
        queue = self._allowed_for_entity(list(sitemap_urls), entity)
        seen_sitemaps: set[str] = set()

        while queue and len(seen_sitemaps) < MAX_SITEMAP_FILES_PER_ENTITY:
            sitemap_url = queue.pop(0)
            if sitemap_url in seen_sitemaps:
                continue
            seen_sitemaps.add(sitemap_url)

            page = await self._fetch_optional(sitemap_url, entity, warnings, checked_urls)
            if page is None:
                continue
            if not self.domain_filter.is_allowed_for_entity(page.final_url, entity):
                _add_warning(
                    warnings,
                    sitemap_url,
                    "Sitemap redirected outside the allowed entity domain.",
                )
                continue

            try:
                loc_urls = parse_sitemap_urls(_decode_page_text(page))
            except Exception as exc:
                _add_warning(warnings, sitemap_url, _short_error("Sitemap parsing failed", exc))
                continue

            for loc_url in self._allowed_for_entity(loc_urls, entity):
                if len(candidates) >= self.settings.max_sitemap_urls_per_entity:
                    return _dedupe_strings(candidates)
                if _looks_like_sitemap_url(loc_url) and loc_url not in seen_sitemaps:
                    queue.append(loc_url)
                    continue
                candidates.append(loc_url)

        return _dedupe_strings(candidates)

    async def _discover_from_page(
        self,
        entity: OfficialEntity,
        page: FetchedPage,
        terms: list[str],
        warnings: list[SearchWarning],
        checked_urls: set[str],
    ) -> list[str]:
        media_type = _media_type(page.content_type)
        text = _decode_page_text(page)
        base_url = page.final_url or page.url
        candidates: list[str] = []

        if _is_sitemap_media(media_type, base_url):
            try:
                candidates.extend(parse_sitemap_urls(text))
            except Exception as exc:
                _add_warning(warnings, base_url, _short_error("Sitemap parsing failed", exc))
        elif _is_feed_media(media_type, base_url):
            try:
                candidates.extend(parse_feed_urls(text))
            except Exception as exc:
                _add_warning(warnings, base_url, _short_error("Feed parsing failed", exc))

        if _is_html_media(media_type):
            try:
                candidates.extend(extract_links(text, base_url))
            except Exception as exc:
                _add_warning(
                    warnings,
                    base_url,
                    _short_error("HTML link extraction failed", exc),
                )

            try:
                feed_links = self._allowed_for_entity(extract_feed_links(text, base_url), entity)
            except Exception as exc:
                _add_warning(
                    warnings,
                    base_url,
                    _short_error("HTML feed link extraction failed", exc),
                )
                feed_links = []
            for feed_url in feed_links[:MAX_FEED_LINKS_PER_PAGE]:
                candidates.extend(
                    await self._discover_feed_entries(
                        entity=entity,
                        feed_url=feed_url,
                        warnings=warnings,
                        checked_urls=checked_urls,
                    )
                )

            try:
                search_urls = self._allowed_for_entity(
                    detect_get_search_forms(text, base_url, terms),
                    entity,
                )
            except Exception as exc:
                _add_warning(
                    warnings,
                    base_url,
                    _short_error("HTML search form discovery failed", exc),
                )
                search_urls = []
            for search_url in search_urls[:MAX_SEARCH_FORM_URLS_PER_PAGE]:
                candidates.append(search_url)
                search_page = await self._fetch_optional(
                    search_url,
                    entity,
                    warnings,
                    checked_urls,
                )
                if search_page is None:
                    continue
                if not self.domain_filter.is_allowed_for_entity(search_page.final_url, entity):
                    _add_warning(
                        warnings,
                        search_url,
                        "Search page redirected outside the allowed entity domain.",
                    )
                    continue
                if _is_html_media(_media_type(search_page.content_type)):
                    try:
                        candidates.extend(
                            extract_links(_decode_page_text(search_page), search_page.final_url)
                        )
                    except Exception as exc:
                        _add_warning(
                            warnings,
                            search_page.final_url,
                            _short_error("HTML link extraction failed", exc),
                        )

        return self._allowed_for_entity(candidates, entity)

    async def _discover_feed_entries(
        self,
        entity: OfficialEntity,
        feed_url: str,
        warnings: list[SearchWarning],
        checked_urls: set[str],
    ) -> list[str]:
        if not self.domain_filter.is_allowed_for_entity(feed_url, entity):
            return []
        feed_page = await self._fetch_optional(feed_url, entity, warnings, checked_urls)
        if feed_page is None:
            return []
        if not self.domain_filter.is_allowed_for_entity(feed_page.final_url, entity):
            _add_warning(warnings, feed_url, "Feed redirected outside the allowed entity domain.")
            return []
        try:
            entry_urls = parse_feed_urls(_decode_page_text(feed_page))
        except Exception as exc:
            _add_warning(warnings, feed_url, _short_error("Feed parsing failed", exc))
            return []
        return self._allowed_for_entity(entry_urls, entity)

    async def _process_candidate_url(
        self,
        candidate_url: str,
        entity: OfficialEntity,
        query_context: dict[str, Any],
        web_candidate_set: set[str],
        warnings: list[SearchWarning],
        checked_urls: set[str],
    ) -> tuple[list[PolicyCard], bool]:
        if not self.domain_filter.is_allowed_for_entity(candidate_url, entity):
            return [], False
        page = await self._fetch_optional(candidate_url, entity, warnings, checked_urls)
        if page is None:
            return [], False
        if not self.domain_filter.is_allowed_for_entity(page.final_url, entity):
            _add_warning(
                warnings,
                candidate_url,
                "Fetched page redirected outside the allowed entity domain.",
            )
            return [], False

        page_info = extract_page_info(
            page.content,
            page.content_type,
            self.settings.max_fulltext_chars_per_page,
            page.final_url,
        )
        fulltext = page_info.fulltext.strip()
        if len(fulltext) < MIN_FULLTEXT_CHARS:
            return [], False

        needs_candidate_judgment = candidate_url not in web_candidate_set
        if needs_candidate_judgment and not await self._should_extract_candidate(
            query_context,
            page,
            fulltext,
            warnings,
        ):
            return [], True

        policies = await self._extract_policy_cards(
            entity=entity,
            page=page,
            query_context=query_context,
            page_info=page_info,
            warnings=warnings,
        )
        return policies, True

    async def _should_extract_candidate(
        self,
        query_context: dict[str, Any],
        page: FetchedPage,
        fulltext: str,
        warnings: list[SearchWarning],
    ) -> bool:
        candidate = {
            "url": page.final_url,
            "content_type": page.content_type,
            "text_preview": fulltext[:1200],
        }
        try:
            judgment = await self.llm.judge_candidate(query_context, candidate)
        except Exception as exc:
            _add_warning(
                warnings,
                page.final_url,
                _short_error("LLM candidate judgment failed", exc),
            )
            return _candidate_fallback_matches(query_context, page, fulltext)

        if not bool(judgment.get("is_relevant")):
            return False
        if judgment.get("is_official") is False:
            return False
        return judgment.get("should_extract_fulltext", True) is not False

    async def _extract_policy_cards(
        self,
        entity: OfficialEntity,
        page: FetchedPage,
        query_context: dict[str, Any],
        page_info: PageExtraction,
        warnings: list[SearchWarning],
    ) -> list[PolicyCard]:
        source = {
            "url": page.final_url,
            "fetched_url": page.url,
            "content_type": page.content_type,
            "entity_id": entity.id,
            "entity_name": entity.display_name,
            "source_type": entity.type,
        }
        fallback_policies = self._fallback_policy_cards_from_page_info(
            entity=entity,
            page=page,
            page_info=page_info,
            warnings=warnings,
        )
        try:
            extracted = await self.llm.extract_policies(
                source,
                query_context,
                _page_extraction_payload(page_info),
            )
        except Exception as exc:
            _add_warning(
                warnings,
                page.final_url,
                _short_error("LLM agent organizer failed", exc),
            )
            return fallback_policies

        raw_policies = extracted.get("policies", [])
        if not isinstance(raw_policies, list):
            _add_warning(
                warnings,
                page.final_url,
                "LLM agent organizer returned a non-list policies field.",
            )
            return fallback_policies

        policies: list[PolicyCard] = []
        for raw_policy in raw_policies:
            if not isinstance(raw_policy, dict):
                _add_warning(
                    warnings,
                    page.final_url,
                    "LLM agent organizer returned a non-object policy item.",
                )
                continue
            policy_data = raw_policy.copy()
            _normalize_extracted_policy_data(policy_data, entity, page.final_url)
            policy_data.setdefault("official_url", page.final_url)
            policy_data.setdefault("matched_entity", entity.display_name)
            policy_data.setdefault("source_name", entity.display_name)
            try:
                policy = PolicyCard.model_validate(policy_data)
            except ValidationError as exc:
                _add_warning(
                    warnings,
                    page.final_url,
                    _short_error("Invalid extracted policy card", exc),
                )
                continue
            if not self.domain_filter.is_allowed_for_entity(policy.official_url, entity):
                _add_warning(
                    warnings,
                    page.final_url,
                    "Extracted policy URL is outside the allowed entity domain.",
                )
                continue
            self._sanitize_policy_urls(policy, entity, page.final_url, warnings)
            policies.append(policy)

        if policies:
            return policies

        if fallback_policies:
            _add_warning(
                warnings,
                page.final_url,
                "LLM agent organizer produced no valid policy cards; using code-extracted evidence.",
            )
        return fallback_policies

    def _fallback_policy_cards_from_page_info(
        self,
        entity: OfficialEntity,
        page: FetchedPage,
        page_info: PageExtraction,
        warnings: list[SearchWarning],
    ) -> list[PolicyCard]:
        raw_cards = _fallback_policy_data_from_page_info(entity, page, page_info)
        policies: list[PolicyCard] = []
        for raw_card in raw_cards:
            if _is_low_value_fallback_card(raw_card):
                continue
            try:
                policy = PolicyCard.model_validate(raw_card)
            except ValidationError as exc:
                _add_warning(
                    warnings,
                    page.final_url,
                    _short_error("Invalid code-extracted policy card", exc),
                )
                continue
            if not self.domain_filter.is_allowed_for_entity(policy.official_url, entity):
                continue
            self._sanitize_policy_urls(policy, entity, page.final_url, warnings)
            policies.append(policy)
        return policies

    async def _fetch_optional(
        self,
        url: str,
        entity: OfficialEntity,
        warnings: list[SearchWarning],
        checked_urls: set[str],
    ) -> FetchedPage | None:
        checked_urls.add(url)
        try:
            return await self._fetch_with_entity_redirect_policy(url, entity)
        except Exception as exc:
            fallback_url = _http_fallback_url(url, exc)
            if fallback_url and self.domain_filter.is_allowed_for_entity(fallback_url, entity):
                checked_urls.add(fallback_url)
                try:
                    page = await self._fetch_with_entity_redirect_policy(fallback_url, entity)
                except Exception as fallback_exc:
                    _add_warning(
                        warnings,
                        fallback_url,
                        _short_error("HTTP fallback fetch failed", fallback_exc),
                    )
                else:
                    _add_warning(
                        warnings,
                        url,
                        "HTTPS fetch failed with bad ecpoint; fetched HTTP fallback.",
                    )
                    return page
            _add_warning(warnings, url, _short_error("Fetch failed", exc))
            return None

    async def _fetch_with_entity_redirect_policy(
        self,
        url: str,
        entity: OfficialEntity,
    ) -> FetchedPage:
        if self._fetcher_accepts_redirect_policy:
            return await self.fetcher.fetch(
                url,
                allowed_redirect_url=lambda redirect_url: (
                    self.domain_filter.is_allowed_for_entity(redirect_url, entity)
                ),
            )
        return await self.fetcher.fetch(url)

    def _allowed_for_entity(self, urls: Iterable[str], entity: OfficialEntity) -> list[str]:
        return _dedupe_strings(
            [url for url in urls if self.domain_filter.is_allowed_for_entity(url, entity)]
        )

    def _sanitize_policy_urls(
        self,
        policy: PolicyCard,
        entity: OfficialEntity,
        source_url: str,
        warnings: list[SearchWarning],
    ) -> None:
        entry_url = policy.application.entry_url
        if entry_url and not self.domain_filter.is_allowed_for_entity(entry_url, entity):
            policy.application.entry_url = None
            _add_warning(
                warnings,
                source_url,
                "Extracted application entry URL is outside the allowed entity domain.",
            )

    async def _summarize_results_markdown(
        self,
        query: str,
        normalized_query: str,
        query_info: dict[str, Any],
        results: list[PolicyCard],
        warnings: list[SearchWarning],
    ) -> str:
        query_context = {
            "query": query,
            "normalized_query": normalized_query,
            "query_type": query_info.get("query_type"),
        }
        payload = [_policy_markdown_payload(policy) for policy in results[:20]]
        try:
            summary_markdown = await self.llm.summarize_markdown(query_context, payload)
        except Exception as exc:
            _add_warning(
                warnings,
                "llm.summarize_markdown",
                _short_error("LLM markdown organizer failed", exc),
            )
            return _fallback_summary_markdown(query, results)

        return summary_markdown.strip() or _fallback_summary_markdown(query, results)

    def _query_context(
        self,
        query: str,
        normalized_query: str,
        query_info: dict[str, Any],
        entity: OfficialEntity,
        terms: list[str],
        path_hints: list[str],
    ) -> dict[str, Any]:
        return {
            "query": query,
            "normalized_query": normalized_query,
            "query_type": query_info.get("query_type"),
            "matched_entity_ids": [entity.id],
            "entity": _entity_payload(entity),
            "terms": terms,
            "url_path_hints": path_hints,
        }

    def _rank_and_dedupe_policies(self, policies: list[PolicyCard]) -> list[PolicyCard]:
        ranked = rank_policies([policy for policy in policies if not _is_low_value_policy(policy)])
        results: list[PolicyCard] = []
        seen: set[tuple[str, str]] = set()
        for policy in ranked:
            key = (_canonical_url(policy.official_url), policy.title.strip().casefold())
            if key in seen:
                continue
            seen.add(key)
            results.append(policy)
        return results

    def _response(
        self,
        query: str,
        normalized_query: str,
        started_at: datetime,
        start_monotonic: float,
        checked_seed_urls: set[str],
        candidate_pages_seen: int,
        summary_markdown: str,
        warnings: list[SearchWarning],
        results: list[PolicyCard],
    ) -> SearchResponse:
        return SearchResponse(
            query=query,
            normalized_query=normalized_query,
            searched_at=started_at,
            duration_seconds=round(time.monotonic() - start_monotonic, 3),
            official_sources_checked=len(checked_seed_urls),
            candidate_pages_seen=candidate_pages_seen,
            results_returned=len(results),
            summary_markdown=summary_markdown,
            warnings=warnings,
            results=results,
        )


def _entity_payload(entity: OfficialEntity) -> dict[str, Any]:
    return entity.model_dump(mode="json")


def _policy_markdown_payload(policy: PolicyCard) -> dict[str, Any]:
    return {
        "title": policy.title,
        "category": " / ".join(policy.policy_types),
        "summary": policy.summary,
        "source_url": policy.official_url,
        "publish_date": policy.dates.published_date,
        "matched_entity": policy.matched_entity,
        "applicable_to": policy.applicable_to,
        "benefits": [benefit.model_dump(mode="json") for benefit in policy.benefits],
        "application": policy.application.model_dump(mode="json"),
        "evidence_snippets": policy.evidence_snippets,
        "score": policy.score,
    }


def _fallback_summary_markdown(query: str, results: list[PolicyCard]) -> str:
    lines = [f"## {query} 人才政策搜索摘要", ""]
    if not results:
        lines.extend(["未找到可汇总的官方政策结果。", ""])
        return "\n".join(lines)

    lines.extend(
        [
            f"共找到 {len(results)} 条官方来源结果。以下按相关性、完整度和发布日期排序。",
            "",
            "### 重点结果",
            "",
        ]
    )
    for index, policy in enumerate(results[:10], start=1):
        category = " / ".join(policy.policy_types) or "人才政策"
        published_date = policy.dates.published_date or "未提取"
        summary = policy.summary or (policy.evidence_snippets[0] if policy.evidence_snippets else "")
        lines.extend(
            [
                f"{index}. **{_markdown_escape(policy.title)}**",
                f"   - 类别：{_markdown_escape(category)}",
                f"   - 发布日期：{_markdown_escape(published_date)}",
                f"   - 摘要：{_markdown_escape(summary) if summary else '暂无摘要，建议查看原文。'}",
                f"   - 来源：[{_markdown_escape(_display_url(policy.official_url))}]({policy.official_url})",
            ]
        )
        if policy.applicable_to:
            lines.append(f"   - 适用对象：{_markdown_escape('；'.join(policy.applicable_to))}")
        benefit_text = _benefit_summary(policy)
        if benefit_text:
            lines.append(f"   - 待遇/支持：{_markdown_escape(benefit_text)}")
        if policy.application.entry_url:
            lines.append(f"   - 申报入口：{policy.application.entry_url}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _benefit_summary(policy: PolicyCard) -> str | None:
    parts: list[str] = []
    for benefit in policy.benefits[:3]:
        item = " ".join(
            value
            for value in [benefit.type, benefit.amount_text, benefit.currency]
            if value
        )
        if item:
            parts.append(item)
    return "；".join(parts) or None


def _display_url(url: str) -> str:
    parsed = urlparse(url)
    return parsed.netloc + parsed.path if parsed.netloc else url


def _markdown_escape(value: str) -> str:
    return value.replace("[", "\\[").replace("]", "\\]")


def _page_extraction_payload(page_info: PageExtraction) -> dict[str, Any]:
    payload = asdict(page_info)
    fulltext = payload.pop("fulltext", "")
    payload["fulltext_preview"] = fulltext[:12000]
    return payload


def _fallback_policy_data_from_page_info(
    entity: OfficialEntity,
    page: FetchedPage,
    page_info: PageExtraction,
) -> list[dict[str, Any]]:
    if page_info.policy_items:
        return [
            _fallback_policy_data_from_item(entity, page, page_info, item)
            for item in page_info.policy_items
        ]

    title = _best_fallback_title(page_info) or page.final_url
    return [
        _fallback_policy_data(
            entity=entity,
            page=page,
            title=title,
            official_url=page.final_url,
            published_date=(page_info.date_candidates[0] if page_info.date_candidates else None),
            evidence=_fallback_evidence(title, page_info.fulltext),
            page_info=page_info,
        )
    ]


def _fallback_policy_data_from_item(
    entity: OfficialEntity,
    page: FetchedPage,
    page_info: PageExtraction,
    item: Any,
) -> dict[str, Any]:
    return _fallback_policy_data(
        entity=entity,
        page=page,
        title=item.title,
        official_url=item.url,
        published_date=item.published_date,
        evidence=_fallback_evidence(item.snippet or item.title, page_info.fulltext),
        page_info=page_info,
    )


def _fallback_policy_data(
    entity: OfficialEntity,
    page: FetchedPage,
    title: str,
    official_url: str,
    published_date: str | None,
    evidence: list[str],
    page_info: PageExtraction,
) -> dict[str, Any]:
    application_link = page_info.application_links[0] if page_info.application_links else None
    attachment_links = [link.url for link in page_info.attachment_links[:10]]
    return {
        "title": _clean_policy_title(title),
        "source_name": entity.display_name,
        "source_type": SOURCE_TYPE_BY_ENTITY_TYPE.get(entity.type, "government"),
        "official_url": official_url,
        "matched_entity": entity.display_name,
        "jurisdiction": entity.jurisdiction,
        "policy_types": _infer_policy_types(title, evidence),
        "applicable_to": _infer_applicable_to(title, evidence),
        "benefits": _infer_benefits(title, evidence),
        "eligibility": [],
        "application": {
            "entry_url": application_link.url if application_link else None,
            "materials": attachment_links,
            "process": application_link.text if application_link else None,
            "evidence": application_link.snippet if application_link else None,
        },
        "dates": {"published_date": published_date},
        "evidence_snippets": evidence[:3],
        "summary": evidence[0] if evidence else title,
        "confidence": 0.62,
        "completeness": 0.38,
    }


def _fallback_evidence(primary: str, fulltext: str) -> list[str]:
    evidence = [_trim(primary, 260)] if primary else []
    for line in fulltext.splitlines():
        text = _safe_text(line)
        if text and text not in evidence:
            evidence.append(_trim(text, 260))
        if len(evidence) >= 3:
            break
    return evidence


def _infer_policy_types(title: str, evidence: list[str]) -> list[str]:
    haystack = f"{title} {' '.join(evidence)}"
    policy_types: list[str] = []
    keyword_types = [
        ("补贴", "人才补贴"),
        ("津贴", "人才津贴"),
        ("安家", "安家补贴"),
        ("住房", "住房支持"),
        ("博士后", "博士后政策"),
        ("科研", "科研资助"),
        ("职称", "职称评审"),
        ("个税", "个税补贴"),
        ("所得税", "个税补贴"),
        ("申报", "申报指南"),
    ]
    for keyword, policy_type in keyword_types:
        if keyword in haystack:
            policy_types.append(policy_type)
    return _dedupe_strings(policy_types) or ["人才政策"]


def _infer_applicable_to(title: str, evidence: list[str]) -> list[str]:
    haystack = f"{title} {' '.join(evidence)}"
    applicable_to: list[str] = []
    for keyword, label in [
        ("高层次人才", "高层次人才"),
        ("境外人才", "境外人才"),
        ("海外人才", "海外人才"),
        ("博士后", "博士后"),
        ("毕业生", "毕业生"),
        ("专业技术人才", "专业技术人才"),
    ]:
        if keyword in haystack:
            applicable_to.append(label)
    return _dedupe_strings(applicable_to)


def _infer_benefits(title: str, evidence: list[str]) -> list[dict[str, str]]:
    haystack = f"{title} {' '.join(evidence)}"
    benefits: list[dict[str, str]] = []
    for keyword, benefit_type in [
        ("补贴", "补贴"),
        ("津贴", "津贴"),
        ("奖励", "奖励"),
        ("资助", "资助"),
        ("安家", "安家支持"),
        ("住房", "住房支持"),
        ("个税", "个税补贴"),
        ("所得税", "个税补贴"),
    ]:
        if keyword in haystack:
            benefits.append(
                {
                    "type": benefit_type,
                    "evidence": evidence[0] if evidence else title,
                }
            )
    return benefits


def _first_fulltext_line(fulltext: str) -> str | None:
    for line in fulltext.splitlines():
        text = _safe_text(line)
        if text:
            return text
    return None


def _best_fallback_title(page_info: PageExtraction) -> str | None:
    title = _clean_policy_title(page_info.title or "")
    if title and not _is_generic_page_title(title) and _looks_like_policy_title(title):
        return title

    for line in page_info.fulltext.splitlines():
        candidate = _clean_policy_title(line)
        if _looks_like_policy_title(candidate):
            return candidate
    return ""


def _clean_policy_title(title: str) -> str:
    cleaned = html.unescape(title)
    for marker in ("<br/>", "<br>", "<br />"):
        cleaned = cleaned.replace(marker, " ")
    return " ".join(cleaned.split())


def _is_generic_page_title(title: str) -> bool:
    normalized = title.strip()
    if not normalized:
        return False
    if normalized in GENERIC_PAGE_TITLES or normalized.endswith("政府信息公开"):
        return True
    return _looks_like_navigation_breadcrumb(normalized)


def _looks_like_navigation_breadcrumb(value: str) -> bool:
    if not value:
        return False
    if ">" not in value and "|" not in value and "｜" not in value:
        compact = "".join(value.split())
        if compact in GENERIC_PAGE_TITLES:
            return True
        return compact in {"学校", "首页"} and len(compact) <= 4

    for token in re.split(r"[>|｜/]", value):
        compact = "".join(token.split())
        if compact in GENERIC_PAGE_TITLES:
            return True
    return False


def _is_low_value_fallback_card(card: dict[str, Any]) -> bool:
    title = _safe_text(card.get("title")) or ""
    evidence = " ".join(_string_list(card.get("evidence_snippets")))
    official_url = _safe_text(card.get("official_url")) or ""
    if _is_generic_page_title(title):
        return True
    if _looks_like_navigation_breadcrumb(title):
        return True
    if not _looks_like_policy_title(title):
        return not _looks_like_policy_title(evidence)
    if _is_generic_page_title(evidence):
        return True
    if _is_generic_policy_url(official_url):
        return True
    return False


def _is_generic_policy_url(url: str) -> bool:
    path = urlparse(url).path.lower().rstrip("/")
    return path in {"", "/", "/index.html", "/index.htm", "/home.html", "/home.htm"}


def _is_low_value_policy(policy: PolicyCard) -> bool:
    if _is_generic_policy_url(policy.official_url):
        return True
    if _looks_like_navigation_breadcrumb(policy.title):
        return True
    if _is_generic_page_title(policy.title):
        return not _looks_like_policy_title(policy.title)
    if not _looks_like_policy_title(policy.title):
        evidence = " ".join(policy.evidence_snippets or [])
        if evidence:
            return not _looks_like_policy_title(evidence)
        return True
    evidence = " ".join(policy.evidence_snippets or [])
    if not evidence:
        return False
    return not _looks_like_policy_title(evidence)


def _looks_like_policy_title(
    title: str,
    *,
    allow_talent_policy_phrase: bool = False,
) -> bool:
    normalized = title.casefold()
    if len(title) < 6 or len(title) > 200:
        return False
    if _is_generic_page_title(title):
        return False
    if (
        not allow_talent_policy_phrase
        and normalized.rstrip().endswith("人才政策")
        and not any(
        token in normalized
        for token in (
            "高层次",
            "引进",
            "认定",
            "补贴",
            "奖励",
            "申报",
            "入选",
            "安家",
            "住房",
            "落户",
            "目录",
            "职称",
            "支持",
            "计划",
            "政策细则",
            "方案",
            "办法",
        )
        )
    ):
        return False
    if title.startswith(("发布日期", "发布时间", "索 引 号", "文 号")):
        return False
    if not any(keyword in normalized for keyword in POLICY_TITLE_KEYWORDS):
        return False
    return any(keyword.casefold() in normalized for keyword in TALENT_POLICY_SIGNAL_KEYWORDS)


def _synthetic_entity(query: str, normalized_query: str) -> OfficialEntity:
    name = _safe_text(normalized_query) or _safe_text(query) or "泛搜索"
    aliases = [alias for alias in _dedupe_strings([query, normalized_query]) if alias != name]
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:12]
    return OfficialEntity(
        id=f"web_query_{digest}",
        type=_infer_entity_type(name),
        display_name=name,
        aliases=aliases,
        jurisdiction=_infer_jurisdiction(name),
        official_domains=[],
        seed_urls=[],
        focus_terms=_dedupe_strings(
            [
                name,
                "人才政策",
                "人才引进",
                "高层次人才",
                "人才补贴",
                "申报指南",
                "安家补贴",
                "博士后",
                "科研资助",
            ]
        ),
    )


def _infer_entity_type(query: str) -> str:
    lowered = query.casefold()
    if "大学" in query or "university" in lowered or "college" in lowered:
        return "university"
    return "region"


def _infer_jurisdiction(query: str) -> str:
    lowered = query.casefold()
    if "香港" in query or "hong kong" in lowered:
        return "hong_kong"
    if "澳门" in query or "macau" in lowered or "macao" in lowered:
        return "macau"
    if "新加坡" in query or "singapore" in lowered:
        return "singapore"
    return "mainland_china"


def _web_search_queries(
    query: str,
    normalized_query: str,
    entity: OfficialEntity,
    terms: list[str],
    limit: int,
) -> list[str]:
    names = _dedupe_strings(
        [
            item
            for item in [
                _safe_text(query),
                _safe_text(normalized_query),
                entity.display_name,
            ]
            if item
        ]
    )
    if not names:
        names = [entity.display_name]

    queries: list[str] = []
    for name in names[:2]:
        if entity.type == "university":
            queries.extend(
                [
                    f"{name} 人才政策 申报 官方",
                    f"{name} 高层次人才 人才引进 官方",
                    f"{name} faculty recruitment benefits official",
                ]
            )
        else:
            queries.extend(
                [
                    f"{name} 人才政策 申报 补贴 官方",
                    f"{name} 高层次人才 人才补贴 政府",
                    f"{name} 人才引进 博士后 科研资助 官方",
                ]
            )
        for term in terms[:2]:
            queries.append(f"{name} {term} 官方")

    return _dedupe_strings(queries)[:limit]


def _web_search_result_looks_policy_like(result: Any) -> bool:
    title = _clean_policy_title(_safe_text(getattr(result, "title", "")) or "")
    snippet = _safe_text(getattr(result, "snippet", "")) or ""
    haystack = f"{title} {snippet}"
    if not haystack.strip():
        return False
    if _is_generic_page_title(title) and _is_generic_page_title(snippet):
        return False
    if _looks_like_policy_title(title):
        return True
    return _looks_like_policy_title(haystack, allow_talent_policy_phrase=True)


def _fetcher_accepts_redirect_policy(fetcher: Any) -> bool:
    try:
        parameters = inspect.signature(fetcher.fetch).parameters.values()
    except (AttributeError, TypeError, ValueError):
        return False

    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        or parameter.name == "allowed_redirect_url"
        for parameter in parameters
    )


def _candidate_fallback_matches(
    query_context: dict[str, Any],
    page: FetchedPage,
    fulltext: str,
) -> bool:
    preview = fulltext[:1200]
    haystack = _search_haystack(page.final_url, preview)
    search_values = [
        *_strings_from_list(query_context.get("terms"), limit=20),
        *_strings_from_list(query_context.get("url_path_hints"), limit=20),
    ]
    for key in ("query", "normalized_query"):
        value = _safe_text(query_context.get(key))
        if value:
            search_values.append(value)

    return any(
        _has_meaningful_search_hit(value, haystack)
        for value in _dedupe_strings(search_values)
    )


def _search_haystack(url: str, text_preview: str) -> str:
    parsed = urlparse(url)
    url_text = unquote_plus(f"{parsed.netloc} {parsed.path} {parsed.query}")
    return f"{url_text} {text_preview}".casefold()


def _has_meaningful_search_hit(value: str, haystack: str) -> bool:
    return any(term in haystack for term in _meaningful_search_terms(value))


def _meaningful_search_terms(value: str) -> list[str]:
    normalized = unquote_plus(" ".join(value.casefold().split()))
    if not normalized:
        return []

    terms = [normalized]
    if " " in normalized:
        terms.extend(part for part in normalized.split(" ") if part)

    return [
        term
        for term in _dedupe_strings(terms)
        if _is_meaningful_search_term(term)
    ]


def _is_meaningful_search_term(term: str) -> bool:
    if len(term) >= 2 and any(ord(char) > 127 for char in term):
        return True
    return len(term) >= 3


def _terms_from_generated(generated: dict[str, Any], limit: int) -> list[str]:
    raw_terms = generated.get("terms")
    if not isinstance(raw_terms, list):
        return []

    ranked_terms: list[tuple[int, int, str]] = []
    for index, item in enumerate(raw_terms):
        term: str | None = None
        priority = index + 100
        if isinstance(item, str):
            term = item
        elif isinstance(item, dict):
            term = _safe_text(item.get("term"))
            raw_priority = item.get("priority")
            if isinstance(raw_priority, int):
                priority = raw_priority
        if term:
            ranked_terms.append((priority, index, term))

    ranked_terms.sort(key=lambda item: (item[0], item[1]))
    return _dedupe_strings([term for _, _, term in ranked_terms])[:limit]


def _normalize_extracted_policy_data(
    policy_data: dict[str, Any],
    entity: OfficialEntity,
    page_url: str,
) -> None:
    title = _first_text(
        policy_data,
        ["title", "official_name", "policy_name", "policy_title", "name"],
    )
    if title:
        policy_data["title"] = _clean_policy_title(title)

    policy_data.setdefault("official_url", page_url)
    policy_data.setdefault("matched_entity", entity.display_name)
    policy_data.setdefault("source_name", entity.display_name)

    source_type = _safe_text(policy_data.get("source_type"))
    if source_type not in SOURCE_TYPES:
        policy_data["source_type"] = SOURCE_TYPE_BY_ENTITY_TYPE.get(entity.type, "government")

    category = _safe_text(policy_data.get("category"))
    if category and not policy_data.get("policy_types"):
        policy_data["policy_types"] = [category]
    policy_data["policy_types"] = _string_list(policy_data.get("policy_types"))
    policy_data["applicable_to"] = _string_list(policy_data.get("applicable_to"))
    policy_data["evidence_snippets"] = _string_list(policy_data.get("evidence_snippets"))
    policy_data["benefits"] = _benefit_list(policy_data.get("benefits"))
    policy_data["eligibility"] = _eligibility_list(policy_data.get("eligibility"))

    application = policy_data.get("application")
    if isinstance(application, str):
        policy_data["application"] = {"process": application}
    elif not isinstance(application, dict):
        policy_data["application"] = {}
    elif isinstance(application.get("materials"), str):
        application["materials"] = [application["materials"]]

    policy_data["confidence"] = _score_value(policy_data.get("confidence"), default=0.5)
    policy_data["completeness"] = _score_value(policy_data.get("completeness"), default=0.5)

    dates = policy_data.get("dates")
    if not isinstance(dates, dict):
        dates = {}
    for key in ("published_date", "effective_date", "deadline", "valid_until"):
        value = _safe_text(policy_data.get(key))
        if value and not dates.get(key):
            dates[key] = value
    if dates:
        policy_data["dates"] = dates


def _first_text(policy_data: dict[str, Any], keys: list[str]) -> str | None:
    for key in keys:
        value = _safe_text(policy_data.get(key))
        if value:
            return value
    return None


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        text = _safe_text(value)
        return [text] if text else []
    if not isinstance(value, list):
        return []
    return [text for item in value if (text := _safe_text(item))]


def _benefit_list(value: Any) -> list[Any]:
    if isinstance(value, dict):
        return [value]
    if isinstance(value, str):
        text = _safe_text(value)
        if text:
            return [{"type": "政策支持", "amount_text": text, "evidence": text}]
        return []
    if not isinstance(value, list):
        return []
    benefits: list[Any] = []
    for item in value:
        if isinstance(item, dict):
            benefits.append(item)
        elif isinstance(item, str) and (text := _safe_text(item)):
            benefits.append({"type": "政策支持", "amount_text": text, "evidence": text})
    return benefits


def _eligibility_list(value: Any) -> list[Any]:
    if isinstance(value, dict):
        return [value]
    if isinstance(value, str):
        text = _safe_text(value)
        if text:
            return [{"condition": text, "evidence": text}]
        return []
    if not isinstance(value, list):
        return []
    eligibility: list[Any] = []
    for item in value:
        if isinstance(item, dict):
            eligibility.append(item)
        elif isinstance(item, str) and (text := _safe_text(item)):
            eligibility.append({"condition": text, "evidence": text})
    return eligibility


def _score_value(value: Any, default: float) -> float:
    if isinstance(value, int | float):
        return _clamp_score(float(value))
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized.endswith("%"):
            try:
                return _clamp_score(float(normalized[:-1].strip()) / 100)
            except ValueError:
                return default
        label_scores = {
            "高": 0.85,
            "较高": 0.75,
            "中": 0.5,
            "中等": 0.5,
            "低": 0.25,
            "high": 0.85,
            "medium": 0.5,
            "low": 0.25,
        }
        if normalized in label_scores:
            return label_scores[normalized]
        try:
            return _clamp_score(float(normalized))
        except ValueError:
            return default
    return default


def _clamp_score(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _strings_from_list(value: Any, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    strings = [_safe_text(item) for item in value]
    return _dedupe_strings([item for item in strings if item])[:limit]


def _dedupe_entities(entities: Iterable[OfficialEntity]) -> list[OfficialEntity]:
    seen: set[str] = set()
    deduped: list[OfficialEntity] = []
    for entity in entities:
        if entity.id in seen:
            continue
        seen.add(entity.id)
        deduped.append(entity)
    return deduped


def _dedupe_strings(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _decode_page_text(page: FetchedPage) -> str:
    charset = _charset(page.content_type)
    try:
        return page.content.decode(charset or "utf-8", errors="replace")
    except LookupError:
        return page.content.decode("utf-8", errors="replace")


def _media_type(content_type: str) -> str:
    return content_type.split(";", 1)[0].strip().lower()


def _charset(content_type: str) -> str | None:
    for parameter in content_type.split(";")[1:]:
        key, separator, value = parameter.partition("=")
        if separator and key.strip().lower() == "charset":
            return value.strip().strip("\"'") or None
    return None


def _is_html_media(media_type: str) -> bool:
    return media_type in {"text/html", "application/xhtml+xml"}


def _is_feed_media(media_type: str, url: str) -> bool:
    if media_type in {"application/rss+xml", "application/atom+xml", "application/rdf+xml"}:
        return True
    path = urlparse(url).path.casefold()
    return path.endswith((".rss", ".atom", ".rdf"))


def _is_sitemap_media(media_type: str, url: str) -> bool:
    if media_type not in {"application/xml", "text/xml"}:
        return False
    return _looks_like_sitemap_url(url) or "sitemap" in urlparse(url).path.casefold()


def _looks_like_sitemap_url(url: str) -> bool:
    path = urlparse(url).path.casefold()
    return path.endswith(".xml") and "sitemap" in path


def _robots_url(seed_url: str) -> str | None:
    parsed = urlparse(seed_url)
    if not parsed.scheme or not parsed.netloc:
        return None
    return urlunparse((parsed.scheme, parsed.netloc, "/robots.txt", "", "", ""))


def _http_fallback_url(url: str, exc: Exception) -> str | None:
    if not _looks_like_bad_ecpoint_error(exc):
        return None
    parsed = urlparse(url)
    if parsed.scheme.casefold() != "https" or not parsed.netloc:
        return None
    return urlunparse(("http", parsed.netloc, parsed.path, parsed.params, parsed.query, ""))


def _looks_like_bad_ecpoint_error(exc: Exception) -> bool:
    message = str(exc).casefold()
    return (
        "bad_ecpoint" in message
        or "bad ecpoint" in message
        or "sslv3" in message
        or "handshake failure" in message
    )


def _canonical_url(url: str) -> str:
    parsed = urlparse(url)
    return urlunparse(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower().rstrip("."),
            parsed.path.rstrip("/") or "/",
            "",
            parsed.query,
            "",
        )
    )


def _safe_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = " ".join(value.split())
    return stripped or None


def _short_error(prefix: str, exc: Exception) -> str:
    return f"{prefix}: {type(exc).__name__}: {_trim(str(exc))}"


def _add_warning(warnings: list[SearchWarning], source: str, message: str) -> None:
    if len(warnings) >= MAX_WARNING_COUNT:
        return
    warnings.append(
        SearchWarning(
            source=_trim(source, MAX_WARNING_SOURCE),
            message=_trim(" ".join(message.split()), MAX_WARNING_TEXT),
        )
    )


def _trim(value: str, limit: int = MAX_WARNING_TEXT) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."
