from __future__ import annotations

from urllib.parse import urlparse

from talent_activity_search.models import OfficialEntity
from talent_activity_search.sources import SourceRegistry


_HOST_LABEL_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789-")


class DomainFilter:
    def __init__(self, registry: SourceRegistry):
        self.registry = registry

    def is_allowed(self, url: object) -> bool:
        host = _host(url)
        if host is None:
            return False
        if self._is_denied(host):
            return False
        if any(_host_matches(host, domain) for domain in self._official_domains()):
            return True
        return any(_suffix_matches(host, suffix) for suffix in self.registry.all_allowed_suffixes())

    def is_allowed_for_entity(self, url: object, entity: OfficialEntity) -> bool:
        host = _host(url)
        if host is None:
            return False
        if self._is_denied(host):
            return False
        if any(_host_matches(host, domain) for domain in entity.official_domains):
            return True
        rule = self.registry.jurisdictions.get(entity.jurisdiction)
        if rule is None:
            return False
        return any(_suffix_matches(host, suffix) for suffix in rule.allow_suffixes)

    def _official_domains(self) -> set[str]:
        domains: set[str] = set()
        for entity in self.registry.entities:
            domains.update(entity.official_domains)
        return domains

    def _is_denied(self, host: str) -> bool:
        return any(_host_matches(host, domain) for domain in self.registry.all_denied_domains())


def _host(url: object) -> str | None:
    url_text = str(url)
    if "\\" in url_text or _has_control_chars(url_text):
        return None
    try:
        parsed = urlparse(url_text)
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
    host = hostname.lower().rstrip(".").removeprefix("www.")
    if not _has_valid_host_labels(host):
        return None
    return host


def _host_matches(host: str, domain: str) -> bool:
    clean = domain.lower().removeprefix("www.")
    return host == clean or host.endswith("." + clean)


def _suffix_matches(host: str, suffix: str) -> bool:
    clean = suffix.lower().lstrip(".")
    return host == clean or host.endswith("." + clean)


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


def _has_control_chars(value: str) -> bool:
    return any(ord(char) < 32 or ord(char) == 127 for char in value)
