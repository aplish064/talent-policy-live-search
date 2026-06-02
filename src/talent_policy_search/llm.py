import hashlib
import json
import re
from collections.abc import Iterable
from typing import Any

import httpx

from talent_policy_search.config import Settings, get_settings


class LLMUnavailableError(RuntimeError):
    """Raised when the configured LLM cannot return a valid JSON object."""


_SK_TOKEN_PATTERN = re.compile(r"\bsk-[A-Za-z0-9][A-Za-z0-9._-]*")
_ANTHROPIC_KEY_PATTERN = re.compile(r"\b(ANTHROPIC_API_KEY\s*=\s*)[^\s,;]+", re.IGNORECASE)
_API_KEY_HEADER_PATTERN = re.compile(r"\b(x-api-key\s*:\s*)[^\s,;]+", re.IGNORECASE)
_AUTHORIZATION_BEARER_PATTERN = re.compile(
    r"\b(authorization\s*:\s*bearer\s+)[^\s,;]+",
    re.IGNORECASE,
)
_JSON_FENCE_PATTERN = re.compile(r"\A```(?:json)?\s*(.*?)\s*```\Z", re.IGNORECASE | re.DOTALL)
_JSON_FENCE_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)


def redact_secrets(text: str, sensitive_values: Iterable[str] | None = None) -> str:
    """Remove API-key shaped values from an error string."""

    redacted = text
    if sensitive_values is not None:
        for value in sorted({value for value in sensitive_values if value}, key=len, reverse=True):
            redacted = redacted.replace(value, "<redacted>")

    redacted = _ANTHROPIC_KEY_PATTERN.sub(r"\1<redacted>", redacted)
    redacted = _API_KEY_HEADER_PATTERN.sub(r"\1<redacted>", redacted)
    redacted = _AUTHORIZATION_BEARER_PATTERN.sub(r"\1<redacted>", redacted)
    return _SK_TOKEN_PATTERN.sub("<redacted>", redacted)


def parse_json_object(text: str) -> dict[str, Any]:
    parsed = parse_json_value(text)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object, got {type(parsed).__name__}")
    return parsed


def parse_json_value(text: str) -> Any:
    stripped = text.strip()

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        pass
    else:
        return parsed

    fence_blocks = list(_JSON_FENCE_BLOCK_PATTERN.finditer(stripped))
    if len(fence_blocks) > 1:
        raise ValueError("Expected a single JSON value; found multiple fenced JSON blocks.")
    if len(fence_blocks) != 1:
        for candidate in _json_value_candidates(stripped):
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            return parsed
        raise ValueError("Expected JSON value text.")

    fenced = _JSON_FENCE_PATTERN.match(stripped)
    stripped = (fenced or fence_blocks[0]).group(1).strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        raise ValueError("Expected JSON value text.") from None

    return parsed


def _json_value_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    start: int | None = None
    stack: list[str] = []
    in_string = False
    escaped = False

    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char in "{[":
            if not stack:
                start = index
            stack.append(char)
            continue
        if char not in "}]" or not stack:
            continue
        expected_closer = "}" if stack[-1] == "{" else "]"
        if char != expected_closer:
            continue

        stack.pop()
        if not stack and start is not None:
            candidates.append(text[start : index + 1])
            start = None

    return candidates


class LLMClient:
    def __init__(
        self,
        settings: Settings | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._transport = transport

    async def complete_json(
        self,
        system_prompt: str,
        payload: dict[str, Any],
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        parsed = await self._complete_json_value(system_prompt, payload, max_tokens)
        if not isinstance(parsed, dict):
            raise LLMUnavailableError(f"Expected JSON object, got {type(parsed).__name__}")
        return parsed

    async def complete_json_value(
        self,
        system_prompt: str,
        payload: dict[str, Any],
        max_tokens: int = 4096,
    ) -> Any:
        return await self._complete_json_value(system_prompt, payload, max_tokens)

    async def complete_text(
        self,
        system_prompt: str,
        payload: dict[str, Any],
        max_tokens: int = 4096,
    ) -> str:
        return await self._complete_text(system_prompt, payload, max_tokens)

    async def _complete_json_value(
        self,
        system_prompt: str,
        payload: dict[str, Any],
        max_tokens: int,
    ) -> Any:
        content_text = await self._complete_text(system_prompt, payload, max_tokens)
        try:
            return parse_json_value(content_text)
        except Exception as exc:
            message = f"LLM response parsing failed: {exc}"
            content_digest = hashlib.sha256(content_text.encode("utf-8")).hexdigest()
            message = (
                f"{message}; content_length={len(content_text)}; "
                f"content_sha256={content_digest}"
            )
            api_key = self._settings.anthropic_api_key.get_secret_value()
            raise LLMUnavailableError(redact_secrets(message, [api_key])) from None

    async def _complete_text(
        self,
        system_prompt: str,
        payload: dict[str, Any],
        max_tokens: int,
    ) -> str:
        api_key = self._settings.anthropic_api_key.get_secret_value()
        if not api_key.strip():
            raise LLMUnavailableError("ANTHROPIC_API_KEY is required.")
        sensitive_values = [api_key]

        request_body = {
            "model": self._settings.anthropic_model,
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False),
                }
            ],
        }
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        url = f"{self._settings.anthropic_base_url.rstrip('/')}/v1/messages"

        try:
            async with httpx.AsyncClient(
                timeout=self._settings.llm_timeout_seconds,
                transport=self._transport,
            ) as client:
                response = await client.post(url, headers=headers, json=request_body)
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            response_body = exc.response.content
            response_digest = hashlib.sha256(response_body).hexdigest()
            message = (
                f"LLM request failed: {exc}; status_code={exc.response.status_code}; "
                f"response_length={len(response_body)}; response_sha256={response_digest}"
            )
            raise LLMUnavailableError(redact_secrets(message, sensitive_values)) from None
        except Exception as exc:
            error_text = str(exc)
            detail = f"{type(exc).__name__}: {error_text}" if error_text else type(exc).__name__
            message = f"LLM request failed: {detail}"
            raise LLMUnavailableError(redact_secrets(message, sensitive_values)) from None

        content_text: str | None = None
        try:
            response_json = response.json()
            content_text = self._extract_text_content(response_json)
            return content_text
        except Exception as exc:
            message = f"LLM response text extraction failed: {exc}"
            if content_text is not None:
                content_digest = hashlib.sha256(content_text.encode("utf-8")).hexdigest()
                message = (
                    f"{message}; content_length={len(content_text)}; "
                    f"content_sha256={content_digest}"
                )
            raise LLMUnavailableError(redact_secrets(message, sensitive_values)) from None

    @staticmethod
    def _extract_text_content(response_json: Any) -> str:
        if not isinstance(response_json, dict):
            raise LLMUnavailableError("LLM response was not a JSON object.")

        content = response_json.get("content")
        if not isinstance(content, list):
            raise LLMUnavailableError("LLM response did not contain content list.")

        text_blocks = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    text_blocks.append(text)

        if text_blocks:
            return "".join(text_blocks)

        raise LLMUnavailableError("LLM response did not contain text content.")
