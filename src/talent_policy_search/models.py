from __future__ import annotations

from datetime import datetime
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field, HttpUrl, field_validator


_HOST_LABEL_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789-")


EntityType = Literal["region", "university", "country_or_jurisdiction", "agency"]
SourceType = Literal[
    "government",
    "university",
    "research_agency",
    "immigration_agency",
    "public_service_portal",
]


class JurisdictionRule(BaseModel):
    allow_suffixes: list[str] = Field(default_factory=list)
    deny_domains: list[str] = Field(default_factory=list)


class OfficialEntity(BaseModel):
    id: str
    type: EntityType
    display_name: str
    aliases: list[str] = Field(default_factory=list)
    jurisdiction: str
    official_domains: list[str] = Field(default_factory=list)
    seed_urls: list[HttpUrl] = Field(default_factory=list)
    focus_terms: list[str] = Field(default_factory=list)

    @property
    def searchable_names(self) -> list[str]:
        return [self.display_name, *self.aliases]


class SourceConfig(BaseModel):
    jurisdictions: dict[str, JurisdictionRule]
    entities: list[OfficialEntity]


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=200)

    @field_validator("query", mode="before")
    @classmethod
    def strip_query(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value


class Benefit(BaseModel):
    type: str
    amount_text: str | None = None
    currency: str | None = None
    evidence: str | None = None


class EligibilityCondition(BaseModel):
    condition: str
    evidence: str | None = None


class ApplicationInfo(BaseModel):
    entry_url: str | None = None
    materials: list[str] = Field(default_factory=list)
    process: str | None = None
    evidence: str | None = None

    @field_validator("entry_url")
    @classmethod
    def validate_entry_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_safe_absolute_url(value)


class PolicyDates(BaseModel):
    published_date: str | None = None
    effective_date: str | None = None
    deadline: str | None = None
    valid_until: str | None = None


class PolicyCard(BaseModel):
    title: str
    source_name: str | None = None
    source_type: SourceType | None = None
    official_url: str
    matched_entity: str | None = None
    jurisdiction: str | None = None
    policy_types: list[str] = Field(default_factory=list)
    applicable_to: list[str] = Field(default_factory=list)
    benefits: list[Benefit] = Field(default_factory=list)
    eligibility: list[EligibilityCondition] = Field(default_factory=list)
    application: ApplicationInfo = Field(default_factory=ApplicationInfo)
    dates: PolicyDates = Field(default_factory=PolicyDates)
    evidence_snippets: list[str] = Field(default_factory=list)
    summary: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    completeness: float = Field(default=0.0, ge=0.0, le=1.0)
    score: float = 0.0

    @field_validator("official_url")
    @classmethod
    def validate_official_url(cls, value: str) -> str:
        return _validate_safe_absolute_url(value)


class SearchWarning(BaseModel):
    source: str
    message: str


class SearchResponse(BaseModel):
    query: str
    normalized_query: str
    mode: str = "deep_realtime_stateless"
    persistence: str = "none"
    searched_at: datetime
    duration_seconds: float
    official_sources_checked: int
    candidate_pages_seen: int
    results_returned: int
    summary_markdown: str = ""
    warnings: list[SearchWarning] = Field(default_factory=list)
    results: list[PolicyCard] = Field(default_factory=list)


def _validate_safe_absolute_url(value: str) -> str:
    if _safe_absolute_host(value) is None:
        raise ValueError("URL must be a safe absolute http(s) URL.")
    return value


def _safe_absolute_host(url: str) -> str | None:
    if "\\" in url or _has_control_chars(url):
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
    host = hostname.lower().rstrip(".").removeprefix("www.")
    if not _has_valid_host_labels(host):
        return None
    return host


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
