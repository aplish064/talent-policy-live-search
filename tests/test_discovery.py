import feedparser.api
import feedparser.http

from talent_policy_search.discovery import (
    detect_get_search_forms,
    extract_feed_links,
    extract_links,
    parse_feed_urls,
    parse_robots_sitemaps,
    parse_sitemap_urls,
    rank_candidate_urls,
    score_url,
)


def test_parse_robots_sitemaps_extracts_sitemap_urls():
    robots = "User-agent: *\nAllow: /\nSitemap: https://example.gov.cn/sitemap.xml\n"

    assert parse_robots_sitemaps(robots) == ["https://example.gov.cn/sitemap.xml"]


def test_parse_sitemap_urls_handles_urlset_and_sitemapindex():
    xml = """
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url><loc>https://hrss.sz.gov.cn/talent.html</loc></url>
    </urlset>
    """
    index = """
    <sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <sitemap><loc>https://hrss.sz.gov.cn/a.xml</loc></sitemap>
    </sitemapindex>
    """

    assert parse_sitemap_urls(xml) == ["https://hrss.sz.gov.cn/talent.html"]
    assert parse_sitemap_urls(index) == ["https://hrss.sz.gov.cn/a.xml"]


def test_extract_links_and_feed_links_resolve_relative_urls():
    html = """
    <html>
      <head><link rel="alternate" type="application/rss+xml" href="/rss.xml"></head>
      <body><a href="/policy/talent.html">人才政策</a></body>
    </html>
    """

    assert extract_links(html, "https://hrss.sz.gov.cn/") == [
        "https://hrss.sz.gov.cn/policy/talent.html"
    ]
    assert extract_feed_links(html, "https://hrss.sz.gov.cn/") == [
        "https://hrss.sz.gov.cn/rss.xml"
    ]


def test_detect_get_search_forms_returns_safe_search_urls():
    html = """
    <form method="get" action="/search">
      <input type="text" name="q">
    </form>
    """

    urls = detect_get_search_forms(html, "https://www.nus.edu.sg/", ["faculty recruitment"])

    assert urls == ["https://www.nus.edu.sg/search?q=faculty+recruitment"]


def test_parse_feed_urls_extracts_entry_links():
    feed = """
    <rss version="2.0">
      <channel>
        <item><link>https://hrss.sz.gov.cn/policy/talent.html</link></item>
      </channel>
    </rss>
    """

    assert parse_feed_urls(feed) == ["https://hrss.sz.gov.cn/policy/talent.html"]


def test_parse_feed_urls_uses_in_memory_stream_without_network_or_file_io(monkeypatch):
    def fail_network(*args, **kwargs):
        raise RuntimeError("feedparser attempted network I/O")

    def fail_file(*args, **kwargs):
        raise RuntimeError("feedparser attempted file I/O")

    monkeypatch.setattr(feedparser.http, "get", fail_network)
    monkeypatch.setattr(feedparser.api, "open", fail_file, raising=False)

    assert parse_feed_urls("https://example.gov.cn/rss.xml") == []
    assert parse_feed_urls("/etc/passwd") == []


def test_parse_feed_urls_keeps_valid_rss_and_atom_parsing_without_file_io(monkeypatch):
    def fail_network(*args, **kwargs):
        raise RuntimeError("feedparser attempted network I/O")

    def fail_file(*args, **kwargs):
        raise RuntimeError("feedparser attempted file I/O")

    monkeypatch.setattr(feedparser.http, "get", fail_network)
    monkeypatch.setattr(feedparser.api, "open", fail_file, raising=False)

    rss = """
    <rss version="2.0">
      <channel>
        <item><link>https://hrss.sz.gov.cn/rss-policy.html</link></item>
      </channel>
    </rss>
    """
    atom = """
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry><link href="https://hrss.sz.gov.cn/atom-policy.html"/></entry>
    </feed>
    """

    assert parse_feed_urls(rss) == ["https://hrss.sz.gov.cn/rss-policy.html"]
    assert parse_feed_urls(atom) == ["https://hrss.sz.gov.cn/atom-policy.html"]


def test_rank_candidate_urls_prioritizes_terms_and_hints():
    urls = [
        "https://example.edu.sg/about",
        "https://example.edu.sg/faculty-recruitment/startup-fund",
    ]

    ranked = rank_candidate_urls(
        urls,
        terms=["startup fund"],
        path_hints=["faculty-recruitment"],
        limit=1,
    )

    assert ranked == ["https://example.edu.sg/faculty-recruitment/startup-fund"]


def test_discovery_rejects_unsafe_extracted_urls():
    html = """
    <html>
      <head>
        <link rel="alternate" type="application/rss+xml" href="https://hrss.sz.gov.cn:444/rss.xml">
        <link
          rel="alternate"
          type="application/atom+xml"
          href="https://user@hrss.sz.gov.cn/feed.xml"
        >
      </head>
      <body>
        <a href="javascript:alert(1)">bad</a>
        <a href="https://evil.com\\@hrss.sz.gov.cn/path">bad</a>
        <a href="https://hrss.sz.gov.cn/path\tquery">bad</a>
        <a href="https://hrss.sz.gov.cn:443/path#section">good</a>
      </body>
    </html>
    """

    assert extract_links(html, "https://hrss.sz.gov.cn/") == ["https://hrss.sz.gov.cn:443/path"]
    assert extract_feed_links(html, "https://hrss.sz.gov.cn/") == []


def test_extractors_reject_raw_whitespace_localhost_and_private_urls():
    html = """
    <html>
      <head>
        <link rel="alternate" type="application/rss+xml" href="http://127.0.0.1/rss.xml">
        <link rel="alternate" type="application/rss+xml" href="https://news.localhost/rss.xml">
        <link rel="alternate" type="application/rss+xml" href="https://example.gov.cn/rss bad.xml">
        <link rel="alternate" type="application/rss+xml" href="https://example.gov.cn/rss.xml">
      </head>
      <body>
        <a href="http://127.0.0.1/admin">loopback</a>
        <a href="http://169.254.169.254/latest">metadata</a>
        <a href="https://localhost/admin">localhost</a>
        <a href="https://example.gov.cn/a b">space</a>
        <a href="https://example.gov.cn/safe.html">safe</a>
      </body>
    </html>
    """

    assert extract_links(html, "https://example.gov.cn/") == [
        "https://example.gov.cn/safe.html"
    ]
    assert extract_feed_links(html, "https://example.gov.cn/") == [
        "https://example.gov.cn/rss.xml"
    ]


def test_extract_links_rejects_non_public_ip_literal_categories():
    html = """
    <html>
      <body>
        <a href="http://127.0.0.1/admin">loopback</a>
        <a href="http://169.254.169.254/latest">link local</a>
        <a href="http://10.0.0.1/private">private</a>
        <a href="http://224.0.0.1/multicast">multicast</a>
        <a href="http://240.0.0.1/reserved">reserved</a>
        <a href="http://0.0.0.0/unspecified">unspecified</a>
        <a href="http://8.8.8.8/public">public</a>
      </body>
    </html>
    """

    assert extract_links(html, "https://example.gov.cn/") == ["http://8.8.8.8/public"]


def test_extract_links_rejects_legacy_ipv4_literals_and_single_label_hosts():
    html = """
    <html>
      <body>
        <a href="http://2130706433/admin">integer loopback</a>
        <a href="http://0177.0.0.1/admin">octal loopback</a>
        <a href="http://0x7f.0.0.1/admin">hex loopback</a>
        <a href="http://127.1/admin">short loopback</a>
        <a href="http://internal/admin">single label</a>
        <a href="http://8.8.8.8/public">public ip</a>
        <a href="https://hrss.sz.gov.cn/policy.html">official domain</a>
      </body>
    </html>
    """

    assert extract_links(html, "https://example.gov.cn/") == [
        "http://8.8.8.8/public",
        "https://hrss.sz.gov.cn/policy.html",
    ]


def test_parse_sources_reject_raw_whitespace_localhost_and_private_urls():
    robots = """
    Sitemap: http://127.0.0.1/sitemap.xml
    Sitemap: https://example.gov.cn/a b.xml
    Sitemap: https://foo.localhost/sitemap.xml
    Sitemap: https://example.gov.cn/sitemap.xml
    """
    sitemap = """
    <urlset>
      <url><loc>http://169.254.169.254/latest</loc></url>
      <url><loc>https://localhost/admin</loc></url>
      <url><loc>https://example.gov.cn/a b.html</loc></url>
      <url><loc>https://example.gov.cn/safe.html</loc></url>
    </urlset>
    """

    assert parse_robots_sitemaps(robots) == ["https://example.gov.cn/sitemap.xml"]
    assert parse_sitemap_urls(sitemap) == ["https://example.gov.cn/safe.html"]


def test_robots_and_ranking_reject_legacy_ipv4_and_single_label_hosts():
    robots = """
    Sitemap: http://2130706433/sitemap.xml
    Sitemap: http://0177.0.0.1/sitemap.xml
    Sitemap: http://0x7f.0.0.1/sitemap.xml
    Sitemap: http://127.1/sitemap.xml
    Sitemap: http://internal/sitemap.xml
    Sitemap: https://example.gov.cn/sitemap.xml
    """
    urls = [
        "http://2130706433/faculty-recruitment/startup-fund",
        "http://0177.0.0.1/faculty-recruitment/startup-fund",
        "http://0x7f.0.0.1/faculty-recruitment/startup-fund",
        "http://127.1/faculty-recruitment/startup-fund",
        "http://internal/faculty-recruitment/startup-fund",
        "https://example.edu.sg/faculty-recruitment/startup-fund",
    ]

    assert parse_robots_sitemaps(robots) == ["https://example.gov.cn/sitemap.xml"]
    assert rank_candidate_urls(urls, ["startup fund"], ["faculty-recruitment"], limit=10) == [
        "https://example.edu.sg/faculty-recruitment/startup-fund"
    ]


def test_rank_candidate_urls_rejects_unsafe_candidates_before_scoring():
    urls = [
        "http://127.0.0.1/faculty-recruitment/startup-fund",
        "http://169.254.169.254/latest?term=startup+fund",
        "https://example.edu.sg/startup fund",
        "https://example.edu.sg/faculty-recruitment/startup-fund",
    ]

    assert rank_candidate_urls(urls, ["startup fund"], ["faculty-recruitment"], limit=10) == [
        "https://example.edu.sg/faculty-recruitment/startup-fund"
    ]


def test_extract_feed_links_accepts_realistic_xml_alternate_feed():
    html = """
    <html>
      <head>
        <link rel="alternate" type="application/xml" href="/feed">
        <link rel="alternate" type="text/html" href="/policy.html">
      </head>
    </html>
    """

    assert extract_feed_links(html, "https://hrss.sz.gov.cn/") == [
        "https://hrss.sz.gov.cn/feed"
    ]


def test_detect_get_search_forms_rejects_actions_with_raw_controls():
    html = """
    <form method="get" action="/search\tbad">
      <input type="text" name="q">
    </form>
    """

    assert detect_get_search_forms(html, "https://example.edu.sg/", ["startup fund"]) == []


def test_detect_get_search_forms_skips_unsafe_or_non_get_forms():
    html = """
    <form method="post" action="/search">
      <input type="text" name="q">
    </form>
    <form method="get" action="https://user:pass@example.edu.sg/search">
      <input type="search" name="q">
    </form>
    <form method="get" action="/upload">
      <input type="file" name="q">
    </form>
    <form method="get" action="/safe">
      <input type="hidden" name="q">
      <input type="search" name="keyword">
    </form>
    """

    urls = detect_get_search_forms(html, "https://example.edu.sg/", ["startup fund"])

    assert urls == ["https://example.edu.sg/safe?keyword=startup+fund"]


def test_parse_sources_dedupe_and_reject_unsafe_urls():
    robots = """
    Sitemap: https://example.gov.cn/sitemap.xml
    Sitemap: https://example.gov.cn/sitemap.xml
    Sitemap: ftp://example.gov.cn/sitemap.xml
    Sitemap: https://bad_host.gov.cn/sitemap.xml
    """
    sitemap = """
    <urlset>
      <url><loc>https://example.gov.cn/a#section</loc></url>
      <url><loc>https://example.gov.cn/a</loc></url>
      <url><loc>https://example.gov.cn:444/b</loc></url>
    </urlset>
    """
    feed = """
    <rss version="2.0">
      <channel>
        <item><link>https://example.gov.cn/a#section</link></item>
        <item><link>ftp://example.gov.cn/b</link></item>
      </channel>
    </rss>
    """

    assert parse_robots_sitemaps(robots) == ["https://example.gov.cn/sitemap.xml"]
    assert parse_sitemap_urls(sitemap) == ["https://example.gov.cn/a"]
    assert parse_feed_urls(feed) == ["https://example.gov.cn/a"]


def test_score_url_rewards_terms_and_path_hints():
    assert (
        score_url(
            "https://example.edu.sg/faculty-recruitment/startup-fund?topic=talent",
            terms=["startup fund", "talent"],
            path_hints=["faculty-recruitment"],
        )
        > score_url("https://example.edu.sg/about", terms=["startup fund"], path_hints=[])
    )


def test_score_url_matches_percent_encoded_chinese_terms():
    policy_url = "https://hrss.sz.gov.cn/%E4%BA%BA%E6%89%8D%E6%94%BF%E7%AD%96.html"

    assert score_url(policy_url, terms=["人才政策"], path_hints=[]) > score_url(
        "https://hrss.sz.gov.cn/about.html",
        terms=["人才政策"],
        path_hints=[],
    )


def test_rank_candidate_urls_retains_percent_encoded_chinese_policy_urls():
    urls = [
        "https://hrss.sz.gov.cn/about.html",
        "https://hrss.sz.gov.cn/%E4%BA%BA%E6%89%8D%E6%94%BF%E7%AD%96.html",
    ]

    assert rank_candidate_urls(urls, terms=["人才政策"], path_hints=[], limit=2) == [
        "https://hrss.sz.gov.cn/%E4%BA%BA%E6%89%8D%E6%94%BF%E7%AD%96.html"
    ]
