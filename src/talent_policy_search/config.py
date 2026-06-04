from functools import lru_cache

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from .env without storing search results."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
        hide_input_in_errors=True,
    )

    anthropic_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices("ANTHROPIC_API_KEY", "anthropic_api_key"),
    )
    anthropic_base_url: str = Field(
        default="https://api.minimaxi.com/anthropic",
        validation_alias=AliasChoices("ANTHROPIC_BASE_URL", "anthropic_base_url"),
    )
    anthropic_model: str = Field(
        default="MiniMax-M2.7-highspeed",
        validation_alias=AliasChoices("ANTHROPIC_MODEL", "anthropic_model"),
    )
    llm_timeout_seconds: int = Field(
        default=60,
        gt=0,
        le=120,
        validation_alias=AliasChoices("LLM_TIMEOUT_SECONDS", "llm_timeout_seconds"),
    )
    search_timeout_seconds: int = Field(
        default=240,
        gt=0,
        le=900,
        validation_alias=AliasChoices("SEARCH_TIMEOUT_SECONDS", "search_timeout_seconds"),
    )
    max_llm_planned_queries: int = Field(
        default=12,
        gt=0,
        le=100,
        validation_alias=AliasChoices("MAX_LLM_PLANNED_QUERIES", "max_llm_planned_queries"),
    )
    max_seed_urls_per_entity: int = Field(
        default=20,
        gt=0,
        le=200,
        validation_alias=AliasChoices("MAX_SEED_URLS_PER_ENTITY", "max_seed_urls_per_entity"),
    )
    max_sitemap_urls_per_entity: int = Field(
        default=800,
        gt=0,
        le=10000,
        validation_alias=AliasChoices("MAX_SITEMAP_URLS_PER_ENTITY", "max_sitemap_urls_per_entity"),
    )
    max_candidate_pages: int = Field(
        default=120,
        gt=0,
        le=1000,
        validation_alias=AliasChoices("MAX_CANDIDATE_PAGES", "max_candidate_pages"),
    )
    max_extracted_pages: int = Field(
        default=50,
        gt=0,
        le=500,
        validation_alias=AliasChoices("MAX_EXTRACTED_PAGES", "max_extracted_pages"),
    )
    max_concurrent_extractions: int = Field(
        default=3,
        gt=0,
        le=20,
        validation_alias=AliasChoices(
            "MAX_CONCURRENT_EXTRACTIONS",
            "max_concurrent_extractions",
        ),
    )
    max_fulltext_chars_per_page: int = Field(
        default=30000,
        gt=0,
        le=200000,
        validation_alias=AliasChoices("MAX_FULLTEXT_CHARS_PER_PAGE", "max_fulltext_chars_per_page"),
    )
    max_fetch_bytes_per_page: int = Field(
        default=10_485_760,
        gt=0,
        le=50_000_000,
        validation_alias=AliasChoices("MAX_FETCH_BYTES_PER_PAGE", "max_fetch_bytes_per_page"),
    )
    official_only: bool = Field(
        default=True,
        validation_alias=AliasChoices("OFFICIAL_ONLY", "official_only"),
    )
    persist_results: bool = Field(
        default=False,
        validation_alias=AliasChoices("PERSIST_RESULTS", "persist_results"),
    )
    enable_public_search_page_fallback: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "ENABLE_PUBLIC_SEARCH_PAGE_FALLBACK",
            "enable_public_search_page_fallback",
        ),
    )
    enable_web_search: bool = Field(
        default=True,
        validation_alias=AliasChoices("ENABLE_WEB_SEARCH", "enable_web_search"),
    )
    anysearch_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices("ANYSEARCH_API_KEY", "anysearch_api_key"),
    )
    anysearch_endpoint: str = Field(
        default="https://api.anysearch.com/v1/search",
        validation_alias=AliasChoices("ANYSEARCH_ENDPOINT", "anysearch_endpoint"),
    )
    anysearch_zone: str = Field(
        default="cn",
        validation_alias=AliasChoices("ANYSEARCH_ZONE", "anysearch_zone"),
    )
    max_web_search_results: int = Field(
        default=12,
        gt=0,
        le=50,
        validation_alias=AliasChoices("MAX_WEB_SEARCH_RESULTS", "max_web_search_results"),
    )
    max_web_search_queries: int = Field(
        default=4,
        gt=0,
        le=20,
        validation_alias=AliasChoices("MAX_WEB_SEARCH_QUERIES", "max_web_search_queries"),
    )
    web_search_timeout_seconds: int = Field(
        default=20,
        gt=0,
        le=120,
        validation_alias=AliasChoices("WEB_SEARCH_TIMEOUT_SECONDS", "web_search_timeout_seconds"),
    )
    enable_scrapling_fallback: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "ENABLE_SCRAPLING_FALLBACK",
            "enable_scrapling_fallback",
        ),
    )

    @model_validator(mode="after")
    def enforce_no_persistence(self) -> "Settings":
        self.assert_no_persistence()
        return self

    def assert_no_persistence(self) -> None:
        if self.persist_results:
            raise ValueError("PERSIST_RESULTS must remain false for this stateless app.")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
