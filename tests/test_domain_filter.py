from pathlib import Path

import pytest

from talent_policy_search.domain_filter import DomainFilter
from talent_policy_search.sources import SourceRegistry


def test_domain_filter_accepts_official_domains_and_suffixes():
    registry = SourceRegistry.from_yaml(Path("config/official_sources.yaml"))
    domain_filter = DomainFilter(registry)
    hkust = registry.match_entity("HKUST")[0]

    assert domain_filter.is_allowed("https://hrss.sz.gov.cn/x/y.html") is True
    assert domain_filter.is_allowed("https://www.gov.hk/en/residents/") is True
    assert domain_filter.is_allowed_for_entity("https://hr.hkust.edu.hk/jobs", hkust) is True


def test_domain_filter_rejects_unofficial_and_unsafe_urls():
    registry = SourceRegistry.from_yaml(Path("config/official_sources.yaml"))
    domain_filter = DomainFilter(registry)

    assert domain_filter.is_allowed("https://weixin.qq.com/s/example") is False
    assert domain_filter.is_allowed("https://evil.com/policy") is False
    assert domain_filter.is_allowed("https://example.com/policy") is False
    assert domain_filter.is_allowed("javascript:alert(1)") is False
    assert domain_filter.is_allowed("ftp://hrss.sz.gov.cn/file") is False


def test_domain_filter_rejects_suffix_spoofing_and_denied_subdomains():
    registry = SourceRegistry.from_yaml(Path("config/official_sources.yaml"))
    domain_filter = DomainFilter(registry)

    assert domain_filter.is_allowed("https://hrss.sz.gov.cn.evil.com/path") is False
    assert domain_filter.is_allowed("https://evilgov.cn/path") is False
    assert domain_filter.is_allowed("https://foo.weixin.qq.com/s/example") is False


def test_domain_filter_accepts_uppercase_official_url():
    registry = SourceRegistry.from_yaml(Path("config/official_sources.yaml"))
    domain_filter = DomainFilter(registry)

    assert domain_filter.is_allowed("HTTPS://HRSS.SZ.GOV.CN/x") is True


def test_domain_filter_rejects_malformed_netlocs_without_raising():
    registry = SourceRegistry.from_yaml(Path("config/official_sources.yaml"))
    domain_filter = DomainFilter(registry)

    assert domain_filter.is_allowed("https://[::1") is False
    assert domain_filter.is_allowed("https://hrss.sz.gov.cn:bad/path") is False


def test_domain_filter_rejects_backslash_and_userinfo_ambiguity():
    registry = SourceRegistry.from_yaml(Path("config/official_sources.yaml"))
    domain_filter = DomainFilter(registry)

    assert domain_filter.is_allowed("https://evil.com\\@hrss.sz.gov.cn/path") is False
    assert domain_filter.is_allowed("https://user:pass@hrss.sz.gov.cn/path") is False
    assert domain_filter.is_allowed("https://hrss.sz.gov.cn\\@evil.com/path") is False


@pytest.mark.parametrize(
    "url",
    [
        "https://hrss.sz.gov.cn/path\nHost:evil.com",
        "https://hrss.sz.gov.cn/path\r\nHost:evil.com",
        "https://hrss.sz.gov.cn/path\tquery",
    ],
)
def test_domain_filter_rejects_raw_control_characters(url):
    registry = SourceRegistry.from_yaml(Path("config/official_sources.yaml"))
    domain_filter = DomainFilter(registry)

    assert domain_filter.is_allowed(url) is False


def test_domain_filter_handles_default_and_non_default_ports():
    registry = SourceRegistry.from_yaml(Path("config/official_sources.yaml"))
    domain_filter = DomainFilter(registry)

    assert domain_filter.is_allowed("https://hrss.sz.gov.cn:443/x") is True
    assert domain_filter.is_allowed("http://hrss.sz.gov.cn:80/x") is True
    assert domain_filter.is_allowed("https://hrss.sz.gov.cn:444/x") is False
    assert domain_filter.is_allowed("https://hrss.sz.gov.cn:80/x") is False


def test_domain_filter_accepts_trailing_dot_official_fqdn():
    registry = SourceRegistry.from_yaml(Path("config/official_sources.yaml"))
    domain_filter = DomainFilter(registry)

    assert domain_filter.is_allowed("https://hrss.sz.gov.cn./path") is True


@pytest.mark.parametrize(
    "url",
    [
        "https://.gov.cn/path",
        "https://hrss..sz.gov.cn/path",
        "https://-foo.gov.cn/path",
        "https://foo-.gov.cn/path",
        "https://foo_bar.gov.cn/path",
    ],
)
def test_domain_filter_rejects_malformed_official_looking_host_labels(url):
    registry = SourceRegistry.from_yaml(Path("config/official_sources.yaml"))
    domain_filter = DomainFilter(registry)

    assert domain_filter.is_allowed(url) is False


def test_domain_filter_accepts_registry_httpurl_seed_urls():
    registry = SourceRegistry.from_yaml(Path("config/official_sources.yaml"))
    domain_filter = DomainFilter(registry)

    for entity in registry.entities:
        for seed_url in entity.seed_urls:
            assert domain_filter.is_allowed(seed_url) is True
            assert domain_filter.is_allowed_for_entity(seed_url, entity) is True
