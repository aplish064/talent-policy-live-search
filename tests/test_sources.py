from pathlib import Path

import pytest
from pydantic import ValidationError

from talent_policy_search.models import ApplicationInfo, PolicyCard, SearchRequest
from talent_policy_search.sources import SourceRegistry


def test_registry_loads_entities_and_matches_aliases():
    registry = SourceRegistry.from_yaml(Path("config/official_sources.yaml"))

    shenzhen = registry.match_entity("深圳市")[0]
    hkust = registry.match_entity("HKUST")[0]

    assert shenzhen.id == "shenzhen"
    assert shenzhen.type == "region"
    assert "hrss.sz.gov.cn" in shenzhen.official_domains
    assert hkust.id == "hkust"
    assert hkust.display_name == "香港科技大学"


def test_registry_returns_empty_list_for_unknown_query():
    registry = SourceRegistry.from_yaml(Path("config/official_sources.yaml"))

    assert registry.match_entity("不存在的大学") == []


@pytest.mark.parametrize("query", ["", "   "])
def test_registry_returns_empty_list_for_blank_query(query):
    registry = SourceRegistry.from_yaml(Path("config/official_sources.yaml"))

    assert registry.match_entity(query) == []


def test_search_request_rejects_blank_query():
    with pytest.raises(ValidationError):
        SearchRequest(query="   ")


def test_policy_card_rejects_invalid_source_type():
    with pytest.raises(ValidationError):
        PolicyCard(title="Policy", official_url="https://hrss.sz.gov.cn/policy", source_type="blog")


def test_policy_card_rejects_out_of_range_confidence():
    with pytest.raises(ValidationError):
        PolicyCard(title="Policy", official_url="https://hrss.sz.gov.cn/policy", confidence=1.1)


def test_policy_card_rejects_out_of_range_completeness():
    with pytest.raises(ValidationError):
        PolicyCard(title="Policy", official_url="https://hrss.sz.gov.cn/policy", completeness=-0.1)


@pytest.mark.parametrize(
    "official_url",
    [
        "javascript:alert(1)",
        "not a url",
        "https://evil.com\\@hrss.sz.gov.cn/path",
        "https://user:pass@hrss.sz.gov.cn/path",
        "https://hrss.sz.gov.cn/path\nHost:evil.com",
    ],
)
def test_policy_card_rejects_unsafe_official_url_shape(official_url):
    with pytest.raises(ValidationError):
        PolicyCard(title="Policy", official_url=official_url)


def test_policy_card_rejects_unsafe_application_entry_url_shape():
    with pytest.raises(ValidationError):
        PolicyCard(
            title="Policy",
            official_url="https://hrss.sz.gov.cn/policy",
            application={"entry_url": "https://user:pass@hrss.sz.gov.cn/apply"},
        )


def test_application_info_rejects_entry_url_with_raw_control_characters():
    with pytest.raises(ValidationError):
        ApplicationInfo(entry_url="https://hrss.sz.gov.cn/path\tquery")


def test_policy_card_accepts_safe_absolute_url_as_string():
    card = PolicyCard(title="Policy", official_url="https://ust.hk/")

    assert card.official_url == "https://ust.hk/"
    assert isinstance(card.official_url, str)
