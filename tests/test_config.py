import pytest
from pydantic import ValidationError

from talent_policy_search.config import Settings, get_settings


SETTINGS_ENV_KEYS = [
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_MODEL",
    "LLM_TIMEOUT_SECONDS",
    "SEARCH_TIMEOUT_SECONDS",
    "MAX_LLM_PLANNED_QUERIES",
    "MAX_SEED_URLS_PER_ENTITY",
    "MAX_SITEMAP_URLS_PER_ENTITY",
    "MAX_CANDIDATE_PAGES",
    "MAX_EXTRACTED_PAGES",
    "MAX_CONCURRENT_EXTRACTIONS",
    "MAX_FULLTEXT_CHARS_PER_PAGE",
    "MAX_FETCH_BYTES_PER_PAGE",
    "OFFICIAL_ONLY",
    "PERSIST_RESULTS",
    "ENABLE_PUBLIC_SEARCH_PAGE_FALLBACK",
    "ENABLE_WEB_SEARCH",
    "ANYSEARCH_API_KEY",
    "ANYSEARCH_ENDPOINT",
    "ANYSEARCH_ZONE",
    "MAX_WEB_SEARCH_RESULTS",
    "MAX_WEB_SEARCH_QUERIES",
    "WEB_SEARCH_TIMEOUT_SECONDS",
    "ENABLE_SCRAPLING_FALLBACK",
]


@pytest.fixture(autouse=True)
def isolate_settings_env(monkeypatch):
    get_settings.cache_clear()
    for key in SETTINGS_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    yield
    get_settings.cache_clear()


def test_settings_defaults_disable_persistence_and_search_api_fallback():
    settings = Settings(_env_file=None)

    assert settings.persist_results is False
    assert settings.enable_public_search_page_fallback is False
    assert settings.enable_web_search is True
    assert settings.anysearch_endpoint == "https://api.anysearch.com/v1/search"
    assert settings.anysearch_zone == "cn"
    assert settings.max_web_search_results == 12
    assert settings.max_web_search_queries == 4
    assert settings.enable_scrapling_fallback is True
    assert settings.max_concurrent_extractions == 3
    assert settings.official_only is True
    assert settings.anthropic_base_url == "https://api.minimaxi.com/anthropic"
    assert settings.anthropic_model == "MiniMax-M2.7-highspeed"
    assert settings.llm_timeout_seconds == 60
    assert settings.max_fetch_bytes_per_page == 10_485_760


def test_get_settings_returns_cached_settings_instance(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    first = get_settings()
    second = get_settings()

    assert first is second


def test_settings_accept_snake_case_constructor_overrides():
    settings = Settings(search_timeout_seconds=5, _env_file=None)

    assert settings.search_timeout_seconds == 5


def test_settings_accept_env_style_constructor_overrides():
    settings = Settings(SEARCH_TIMEOUT_SECONDS=5, _env_file=None)

    assert settings.search_timeout_seconds == 5


def test_settings_accept_max_fetch_bytes_constructor_overrides():
    snake_case_settings = Settings(max_fetch_bytes_per_page=1024, _env_file=None)
    env_style_settings = Settings(MAX_FETCH_BYTES_PER_PAGE=2048, _env_file=None)

    assert snake_case_settings.max_fetch_bytes_per_page == 1024
    assert env_style_settings.max_fetch_bytes_per_page == 2048


@pytest.mark.parametrize("kwargs", [{"PERSIST_RESULTS": True}, {"persist_results": True}])
def test_settings_reject_persistence_at_model_boundary(kwargs):
    with pytest.raises(ValidationError):
        Settings(**kwargs, _env_file=None)


def test_settings_reject_non_positive_numeric_values():
    with pytest.raises(ValidationError):
        Settings(SEARCH_TIMEOUT_SECONDS=0, _env_file=None)


def test_settings_reject_non_positive_fetch_byte_limit():
    with pytest.raises(ValidationError):
        Settings(MAX_FETCH_BYTES_PER_PAGE=0, _env_file=None)


def test_settings_reject_unreasonably_large_numeric_values():
    with pytest.raises(ValidationError):
        Settings(MAX_CANDIDATE_PAGES=1001, _env_file=None)


def test_settings_reject_unreasonably_large_fetch_byte_limit():
    with pytest.raises(ValidationError):
        Settings(MAX_FETCH_BYTES_PER_PAGE=50_000_001, _env_file=None)


def test_settings_are_immutable_after_validation():
    settings = Settings(_env_file=None)

    with pytest.raises(ValidationError):
        settings.persist_results = True


def test_api_key_is_secret_in_repr_but_accessible_as_secret_value():
    settings = Settings(
        ANTHROPIC_API_KEY="sk-test-secret",
        ANYSEARCH_API_KEY="anysearch-secret",
        _env_file=None,
    )

    assert "sk-test-secret" not in repr(settings)
    assert "anysearch-secret" not in repr(settings)
    assert settings.anthropic_api_key.get_secret_value() == "sk-test-secret"
    assert settings.anysearch_api_key.get_secret_value() == "anysearch-secret"


def test_settings_load_search_timeout_from_environment(monkeypatch):
    monkeypatch.setenv("SEARCH_TIMEOUT_SECONDS", "7")

    settings = Settings(_env_file=None)

    assert settings.search_timeout_seconds == 7


def test_settings_load_fetch_byte_limit_from_environment(monkeypatch):
    monkeypatch.setenv("MAX_FETCH_BYTES_PER_PAGE", "4096")

    settings = Settings(_env_file=None)

    assert settings.max_fetch_bytes_per_page == 4096


def test_settings_reject_persistence_from_environment(monkeypatch):
    monkeypatch.setenv("PERSIST_RESULTS", "true")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)
