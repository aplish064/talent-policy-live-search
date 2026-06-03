import json
import traceback

import httpx
import pytest

from talent_activity_search.config import Settings
from talent_activity_search.llm import (
    LLMClient,
    LLMUnavailableError,
    parse_json_object,
    parse_json_value,
    redact_secrets,
)


def test_parse_json_object_accepts_plain_and_fenced_json():
    assert parse_json_object('{"ok": true}') == {"ok": True}
    fenced = '```json\n{"value": 3}\n```'
    assert parse_json_object(fenced) == {"value": 3}


def test_parse_json_object_accepts_plain_json_containing_backticks():
    assert parse_json_object('{"snippet":"```quoted```"}') == {"snippet": "```quoted```"}


def test_parse_json_object_accepts_prose_wrapped_single_fenced_json():
    text = 'Here is the JSON:\n```json\n{"ok": true}\n```'

    assert parse_json_object(text) == {"ok": True}


def test_parse_json_object_accepts_prose_wrapped_unfenced_json_object():
    text = '结果如下：\n{"policies": [{"title": "广州市人才政策"}], "discard_reason": null}\n请查收。'

    assert parse_json_object(text) == {
        "policies": [{"title": "广州市人才政策"}],
        "discard_reason": None,
    }


def test_parse_json_value_accepts_prose_wrapped_array():
    text = '候选如下：\n[{"term": "广州 人才政策"}, {"term": "人才补贴"}]'

    assert parse_json_value(text) == [{"term": "广州 人才政策"}, {"term": "人才补贴"}]


@pytest.mark.parametrize("text", ["[]", '"value"', "not json"])
def test_parse_json_object_rejects_non_object_or_invalid_json(text):
    with pytest.raises(ValueError):
        parse_json_object(text)


def test_parse_json_object_rejects_multiple_fenced_blocks_without_raw_text():
    text = '```json\n{"first": true}\n```\n```json\n{"second": true}\n```'

    with pytest.raises(ValueError) as exc_info:
        parse_json_object(text)

    assert text not in str(exc_info.value)


def test_parse_json_object_invalid_json_error_does_not_include_raw_text():
    text = "not json secret-key-123"

    with pytest.raises(ValueError) as exc_info:
        parse_json_object(text)

    assert text not in str(exc_info.value)
    assert "secret-key-123" not in str(exc_info.value)


def test_redact_secrets_masks_api_keys():
    text = (
        "failed with sk-test-abc123 and ANTHROPIC_API_KEY=secret "
        "and x-api-key: secret-key-123 and X-Api-Key: other-secret "
        "and Authorization: Bearer bearer-secret"
    )

    redacted = redact_secrets(text, sensitive_values=["secret-key-123"])

    assert "sk-test-abc123" not in redacted
    assert "ANTHROPIC_API_KEY=secret" not in redacted
    assert "secret-key-123" not in redacted
    assert "other-secret" not in redacted
    assert "bearer-secret" not in redacted
    assert "ANTHROPIC_API_KEY=<redacted>" in redacted
    assert "x-api-key: <redacted>" in redacted
    assert "X-Api-Key: <redacted>" in redacted
    assert "Authorization: Bearer <redacted>" in redacted


@pytest.mark.asyncio
async def test_llm_client_posts_to_anthropic_messages_and_parses_json():
    system_prompt = "system"
    payload = {"query": "深圳"}

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/anthropic/v1/messages"
        assert request.headers["x-api-key"] == "test-key"
        assert "**********" not in request.headers["x-api-key"]
        assert request.headers["anthropic-version"] == "2023-06-01"
        assert "application/json" in request.headers["content-type"]

        body = json.loads(request.content.decode("utf-8"))
        assert body["model"] == settings.anthropic_model
        assert body["max_tokens"] == 123
        assert body["system"] == system_prompt
        assert body["messages"][0]["role"] == "user"
        assert json.loads(body["messages"][0]["content"]) == payload

        return httpx.Response(
            200,
            json={
                "content": [
                    {
                        "type": "text",
                        "text": '{"normalized_query":"深圳","query_type":"region"}',
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    settings = Settings(
        ANTHROPIC_API_KEY="test-key",
        ANTHROPIC_BASE_URL="https://api.minimaxi.com/anthropic",
        _env_file=None,
    )
    client = LLMClient(settings=settings, transport=transport)

    result = await client.complete_json(system_prompt, payload, max_tokens=123)

    assert result == {"normalized_query": "深圳", "query_type": "region"}


@pytest.mark.asyncio
async def test_llm_client_concatenates_text_content_blocks_before_parsing():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "content": [
                    {"type": "text", "text": '{"normalized_query":'},
                    {"type": "text", "text": '"深圳","query_type":"region"}'},
                ]
            },
        )

    settings = Settings(ANTHROPIC_API_KEY="secret", _env_file=None)
    client = LLMClient(settings=settings, transport=httpx.MockTransport(handler))

    result = await client.complete_json("system", {"query": "深圳"})

    assert result == {"normalized_query": "深圳", "query_type": "region"}


@pytest.mark.asyncio
async def test_llm_client_requires_api_key():
    client = LLMClient(settings=Settings(ANTHROPIC_API_KEY="", _env_file=None))

    with pytest.raises(LLMUnavailableError):
        await client.complete_json("system", {"query": "深圳"})


@pytest.mark.asyncio
async def test_llm_client_redacts_http_failures():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            request=request,
            text="invalid ANTHROPIC_API_KEY=secret-key-123 and sk-test-abc123",
        )

    settings = Settings(ANTHROPIC_API_KEY="secret-key-123", _env_file=None)
    client = LLMClient(settings=settings, transport=httpx.MockTransport(handler))

    with pytest.raises(LLMUnavailableError) as exc_info:
        await client.complete_json("system", {"query": "深圳"})

    message = str(exc_info.value)
    formatted = "".join(
        traceback.format_exception(
            type(exc_info.value),
            exc_info.value,
            exc_info.value.__traceback__,
        )
    )
    assert "sk-test-abc123" not in message
    assert "secret-key-123" not in message
    assert "secret-key-123" not in formatted
    assert exc_info.value.__cause__ is None
    assert "ANTHROPIC_API_KEY=" not in message
    assert "status_code=401" in message
    assert "response_length=" in message
    assert "response_sha256=" in message


@pytest.mark.asyncio
async def test_llm_client_omits_raw_http_error_response_body():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500,
            request=request,
            text="provider echoed user-secret-456",
        )

    settings = Settings(ANTHROPIC_API_KEY="secret-key-123", _env_file=None)
    client = LLMClient(settings=settings, transport=httpx.MockTransport(handler))

    with pytest.raises(LLMUnavailableError) as exc_info:
        await client.complete_json("system", {"query": "深圳"})

    message = str(exc_info.value)
    formatted = "".join(traceback.format_exception(exc_info.value))
    assert "provider echoed user-secret-456" not in message
    assert "provider echoed user-secret-456" not in formatted
    assert "user-secret-456" not in message
    assert "user-secret-456" not in formatted
    assert exc_info.value.__cause__ is None
    assert "status_code=500" in message
    assert "response_length=" in message
    assert "response_sha256=" in message


@pytest.mark.asyncio
async def test_llm_client_includes_exception_type_for_transport_failures():
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("", request=request)

    settings = Settings(ANTHROPIC_API_KEY="secret-key-123", _env_file=None)
    client = LLMClient(settings=settings, transport=httpx.MockTransport(handler))

    with pytest.raises(LLMUnavailableError) as exc_info:
        await client.complete_json("system", {"query": "深圳"})

    message = str(exc_info.value)
    assert "ReadTimeout" in message
    assert "LLM request failed:" in message
    assert "secret-key-123" not in message


@pytest.mark.asyncio
async def test_llm_client_omits_non_configured_tokens_from_parse_failure_messages():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "content": [
                    {
                        "type": "text",
                        "text": "not json customer-token-123",
                    }
                ]
            },
        )

    settings = Settings(ANTHROPIC_API_KEY="secret-key-123", _env_file=None)
    client = LLMClient(settings=settings, transport=httpx.MockTransport(handler))

    with pytest.raises(LLMUnavailableError) as exc_info:
        await client.complete_json("system", {"query": "深圳"})

    message = str(exc_info.value)
    formatted = "".join(traceback.format_exception(exc_info.value))
    assert "customer-token-123" not in message
    assert "customer-token-123" not in formatted
    assert exc_info.value.__cause__ is None
    assert "content_length=" in message


@pytest.mark.asyncio
async def test_llm_client_redacts_response_parsing_failures():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "content": [
                    {
                        "type": "text",
                        "text": "not json sk-test-abc123 ANTHROPIC_API_KEY=secret-key-123",
                    }
                ]
            },
        )

    settings = Settings(ANTHROPIC_API_KEY="secret-key-123", _env_file=None)
    client = LLMClient(settings=settings, transport=httpx.MockTransport(handler))

    with pytest.raises(LLMUnavailableError) as exc_info:
        await client.complete_json("system", {"query": "深圳"})

    message = str(exc_info.value)
    formatted = "".join(
        traceback.format_exception(
            type(exc_info.value),
            exc_info.value,
            exc_info.value.__traceback__,
        )
    )
    assert "sk-test-abc123" not in message
    assert "secret-key-123" not in message
    assert "secret-key-123" not in formatted
    assert exc_info.value.__cause__ is None
    assert "content_length=" in message
