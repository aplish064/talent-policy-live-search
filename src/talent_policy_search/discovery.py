from __future__ import annotations

import io
import ipaddress
import re
import socket
import xml.etree.ElementTree as ET
from urllib.parse import parse_qsl, unquote, unquote_plus, urlencode, urljoin, urlparse, urlunparse

import feedparser
from bs4 import BeautifulSoup


_HOST_LABEL_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789-")
_TERM_RE = re.compile(r"[a-z0-9]+")
_FEED_TYPES = {"application/rss+xml", "application/atom+xml"}
_XML_FEED_TYPES = {"application/xml", "text/xml", "application/rdf+xml"}
_FEED_PATH_HINTS = {"rss", "atom", "feed"}
_TEXT_INPUT_TYPES = {"", "text", "search"}
_UNSAFE_FORM_INPUT_TYPES = {"file", "password"}


def parse_robots_sitemaps(text: str) -> list[str]:
    urls: list[str] = []
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip().lower() == "sitemap":
            url = _safe_url(value.strip())
            if url is not None:
                urls.append(url)
    return _dedupe(urls)


def parse_sitemap_urls(xml_text: str) -> list[str]:
    root = ET.fromstring(xml_text)
    urls: list[str] = []
    for element in root.iter():
        if _local_name(element.tag) != "loc" or element.text is None:
            continue
        url = _safe_url(element.text.strip())
        if url is not None:
            urls.append(url)
    return _dedupe(urls)


def extract_links(html_text: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html_text, "lxml")
    urls: list[str] = []
    for anchor in soup.find_all("a"):
        href = anchor.get("href")
        if not href:
            continue
        url = _safe_join(base_url, href)
        if url is not None:
            urls.append(url)
    return _dedupe(urls)


def extract_feed_links(html_text: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html_text, "lxml")
    urls: list[str] = []
    for link in soup.find_all("link"):
        if not _has_rel(link.get("rel"), "alternate"):
            continue
        media_type = str(link.get("type", "")).split(";", 1)[0].strip().lower()
        href = link.get("href")
        if not href:
            continue
        url = _safe_join(base_url, href)
        if url is not None:
            if not (_is_feed_media_type(media_type) or _looks_like_feed_href(url)):
                continue
            urls.append(url)
    return _dedupe(urls)


def parse_feed_urls(feed_text: str) -> list[str]:
    parsed = feedparser.parse(io.BytesIO(feed_text.encode("utf-8")))
    urls: list[str] = []
    for entry in parsed.entries:
        url = _safe_url(entry.get("link", ""))
        if url is not None:
            urls.append(url)
    return _dedupe(urls)


def detect_get_search_forms(html_text: str, base_url: str, terms: list[str]) -> list[str]:
    soup = BeautifulSoup(html_text, "lxml")
    urls: list[str] = []
    for form in soup.find_all("form"):
        method = str(form.get("method", "get")).strip().lower()
        if method and method != "get":
            continue
        if _has_unsafe_form_input(form):
            continue
        input_name = _first_search_input_name(form)
        if input_name is None:
            continue
        action = form.get("action") or base_url
        action_url = _safe_join(base_url, action)
        if action_url is None:
            continue
        for term in terms:
            generated = _safe_url(_with_query_param(action_url, input_name, term))
            if generated is not None:
                urls.append(generated)
    return _dedupe(urls)


def rank_candidate_urls(
    urls: list[str],
    terms: list[str],
    path_hints: list[str],
    limit: int,
) -> list[str]:
    scored: list[tuple[int, str]] = []
    safe_urls = [safe_url for url in urls if (safe_url := _safe_url(url)) is not None]
    for url in _dedupe(safe_urls):
        score = score_url(url, terms, path_hints)
        if score > 0:
            scored.append((score, url))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [url for _, url in scored[:limit]]


def score_url(url: str, terms: list[str], path_hints: list[str]) -> int:
    safe_url = _safe_url(url)
    if safe_url is None:
        return 0

    parsed = urlparse(safe_url)
    path = unquote(parsed.path).casefold()
    searchable = f"{path} {unquote_plus(parsed.query).casefold()}"
    searchable_words = set(_TERM_RE.findall(searchable))
    score = 0

    for hint in path_hints:
        normalized_hint = _normalize_search_text(hint)
        if normalized_hint and normalized_hint in path:
            score += 5

    for term in terms:
        normalized_term = _normalize_search_text(term)
        words = _term_words(normalized_term)
        if not words:
            if normalized_term and normalized_term in searchable:
                score += 4
            continue
        matched_words = sum(1 for word in words if word in searchable_words)
        score += matched_words
        if matched_words == len(words):
            score += 3

    return score


def _strip_fragment(url: str) -> str:
    parsed = urlparse(url)
    return urlunparse(parsed._replace(fragment=""))


def _with_query_param(url: str, name: str, value: str) -> str:
    parsed = urlparse(url)
    params = [
        (key, val)
        for key, val in parse_qsl(parsed.query, keep_blank_values=True)
        if key != name
    ]
    params.append((name, value))
    return urlunparse(parsed._replace(query=urlencode(params)))


def _dedupe(urls: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        deduped.append(url)
    return deduped


def _safe_url(url: object) -> str | None:
    url_text = _strict_url_text(url)
    if not url_text:
        return None
    stripped = _strip_fragment(url_text)
    if _safe_absolute_host(stripped) is None:
        return None
    return stripped


def _safe_join(base_url: str, value: object) -> str | None:
    base_text = _strict_url_text(base_url)
    value_text = _strict_url_text(value)
    if base_text is None or value_text is None:
        return None
    return _safe_url(urljoin(base_text, value_text))


def _safe_absolute_host(url: str) -> str | None:
    if _has_url_ambiguity(url):
        return None
    try:
        parsed = urlparse(url)
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        return None
    if scheme not in {"http", "https"}:
        return None
    if "@" in netloc:
        return None
    if not hostname:
        return None
    if port is not None and port != _default_port(scheme):
        return None
    host = hostname.lower().rstrip(".")
    if host == "localhost" or host.endswith(".localhost"):
        return None
    ip = _parse_ip_address(host)
    if ip is not None:
        if _is_disallowed_ip_address(ip):
            return None
        return host
    if _is_legacy_ipv4_literal(host):
        return None
    host = host.removeprefix("www.")
    if "." not in host:
        return None
    if not _has_valid_host_labels(host):
        return None
    return host


def _parse_ip_address(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        return None


def _is_disallowed_ip_address(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return (
        ip.is_loopback
        or ip.is_link_local
        or ip.is_private
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
        or not ip.is_global
    )


def _is_legacy_ipv4_literal(host: str) -> bool:
    try:
        socket.inet_aton(host)
    except OSError:
        return False
    return True


def _default_port(scheme: str) -> int:
    return 443 if scheme == "https" else 80


def _has_valid_host_labels(host: str) -> bool:
    labels = host.split(".")
    return all(
        label
        and not label.startswith("-")
        and not label.endswith("-")
        and all(char in _HOST_LABEL_CHARS for char in label)
        for label in labels
    )


def _has_url_ambiguity(value: str) -> bool:
    return "\\" in value or any(char.isspace() for char in value)


def _strict_url_text(value: object) -> str | None:
    raw = str(value)
    if not raw or raw != raw.strip():
        return None
    if _has_url_ambiguity(raw):
        return None
    return raw


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _has_rel(rel: object, expected: str) -> bool:
    if isinstance(rel, str):
        values = rel.split()
    elif isinstance(rel, list):
        values = [str(value) for value in rel]
    else:
        return False
    return expected in {value.lower() for value in values}


def _is_feed_media_type(media_type: str) -> bool:
    return (
        media_type in _FEED_TYPES
        or media_type in _XML_FEED_TYPES
        or any(token in media_type for token in ("rss", "atom", "rdf"))
    )


def _looks_like_feed_href(url: str) -> bool:
    path = urlparse(url).path.casefold().rstrip("/")
    if path.endswith((".rss", ".atom", ".rdf")):
        return True
    segments = [segment for segment in path.split("/") if segment]
    if any(segment in _FEED_PATH_HINTS for segment in segments):
        return True
    if not segments:
        return False
    filename = segments[-1]
    return any(filename.startswith(f"{hint}.") for hint in _FEED_PATH_HINTS)


def _has_unsafe_form_input(form: object) -> bool:
    for input_tag in form.find_all("input"):
        input_type = str(input_tag.get("type", "text")).strip().lower()
        if input_type in _UNSAFE_FORM_INPUT_TYPES:
            return True
    return False


def _first_search_input_name(form: object) -> str | None:
    for input_tag in form.find_all("input"):
        input_type = str(input_tag.get("type", "text")).strip().lower()
        if input_type not in _TEXT_INPUT_TYPES:
            continue
        name = str(input_tag.get("name", "")).strip()
        if name:
            return name
    return None


def _term_words(term: str) -> list[str]:
    return _TERM_RE.findall(term.casefold())


def _normalize_search_text(value: str) -> str:
    return unquote_plus(value.strip()).casefold()
