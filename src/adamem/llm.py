"""Provider-agnostic LLM client.

This module exposes a tiny `LLMClient` Protocol so the rest of AdaMem can call
LLM judges and answer agents without depending on any specific SDK. We ship
three concrete clients:

* `OpenAIClient`     - uses OpenAI Chat Completions HTTP API.
* `GeminiClient`     - uses Google Generative Language `generateContent` API.
* `MockLLMClient`    - deterministic, scripted responses for tests / CI.

Real clients pull credentials from environment variables and never log them.
We intentionally avoid the official SDKs so the package keeps zero hard
dependencies. The wire calls are plain JSON over HTTPS via `urllib`.
"""
from __future__ import annotations

import json
import os
import ssl
import time
from dataclasses import dataclass
from typing import Protocol
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError


def _ssl_context_from_env() -> ssl.SSLContext | None:
    """Optional unverified SSL context for internal/corp endpoints."""
    if os.environ.get("ADAMEM_INSECURE_SSL", "").lower() in {"1", "true", "yes"}:
        return ssl._create_unverified_context()
    return None


class LLMClient(Protocol):
    """A minimal text-in / text-out LLM interface."""

    def complete(self, prompt: str, *, system: str | None = None, max_tokens: int = 256, temperature: float = 0.0) -> str:
        ...


@dataclass(slots=True)
class _RetrySpec:
    attempts: int = 3
    backoff_seconds: float = 1.5


class _HttpJSONClient:
    """Shared retry/transport for HTTP-JSON LLM endpoints."""

    def __init__(self, retry: _RetrySpec | None = None, timeout_seconds: float = 60.0) -> None:
        self.retry = retry or _RetrySpec()
        self.timeout_seconds = timeout_seconds

    def _post_json(self, url: str, payload: dict, headers: dict) -> dict:
        body = json.dumps(payload).encode("utf-8")
        last_error: Exception | None = None
        ssl_ctx = _ssl_context_from_env()
        for attempt in range(self.retry.attempts):
            req = urlrequest.Request(url=url, data=body, headers=headers, method="POST")
            try:
                with urlrequest.urlopen(req, timeout=self.timeout_seconds, context=ssl_ctx) as response:
                    raw = response.read().decode("utf-8")
                return json.loads(raw)
            except HTTPError as error:
                last_error = error
                if (error.code == 429 or 500 <= error.code < 600) and attempt + 1 < self.retry.attempts:
                    time.sleep(self.retry.backoff_seconds * (2 ** attempt))
                    continue
                raise
            except (URLError, TimeoutError) as error:
                last_error = error
                if attempt + 1 < self.retry.attempts:
                    time.sleep(self.retry.backoff_seconds * (2 ** attempt))
                    continue
                raise
        assert last_error is not None
        raise last_error


class OpenAIClient(_HttpJSONClient):
    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        retry: _RetrySpec | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        super().__init__(retry=retry, timeout_seconds=timeout_seconds)
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAIClient")
        self.base_url = base_url.rstrip("/")

    def complete(self, prompt: str, *, system: str | None = None, max_tokens: int = 256, temperature: float = 0.0) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = self._post_json(f"{self.base_url}/chat/completions", payload, headers)
        return data["choices"][0]["message"]["content"].strip()


class GeminiClient(_HttpJSONClient):
    def __init__(
        self,
        *,
        model: str = "gemini-1.5-flash",
        api_key: str | None = None,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        retry: _RetrySpec | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        super().__init__(retry=retry, timeout_seconds=timeout_seconds)
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is required for GeminiClient")
        self.base_url = base_url.rstrip("/")

    def complete(self, prompt: str, *, system: str | None = None, max_tokens: int = 256, temperature: float = 0.0) -> str:
        contents = [{"role": "user", "parts": [{"text": prompt}]}]
        payload: dict = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            },
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        url = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"
        data = self._post_json(url, payload, {"Content-Type": "application/json"})
        candidates = data.get("candidates") or []
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts") or []
        return "".join(part.get("text", "") for part in parts).strip()


class ModelHubClient(_HttpJSONClient):
    """ByteDance ModelHub (Azure OpenAI-compatible) client.

    The ModelHub gateway proxies many backends (Gemini, GPT, ...) behind an
    Azure-OpenAI shaped chat-completions endpoint. The `azure_endpoint` is the
    BASE; the actual call goes to
    `{azure_endpoint}/deployments/{model}/chat/completions?api-version=...`.
    Authentication uses an `api-key` header rather than `Authorization: Bearer`.
    """

    def __init__(
        self,
        *,
        model: str = "gemini-3.1-fi",
        api_key: str | None = None,
        azure_endpoint: str | None = None,
        api_version: str = "2024-03-01-preview",
        retry: _RetrySpec | None = None,
        timeout_seconds: float = 90.0,
    ) -> None:
        super().__init__(retry=retry, timeout_seconds=timeout_seconds)
        self.model = model
        self.api_key = api_key or os.environ.get("MODELHUB_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("MODELHUB_API_KEY is required for ModelHubClient")
        self.azure_endpoint = (
            azure_endpoint
            or os.environ.get("MODELHUB_ENDPOINT")
            or "https://aidp.bytedance.net/api/modelhub/online/multimodal/crawl"
        ).rstrip("/")
        self.api_version = api_version

    def complete(self, prompt: str, *, system: str | None = None, max_tokens: int = 256, temperature: float = 0.0) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        url = (
            f"{self.azure_endpoint}/openai/deployments/{self.model}/chat/completions"
            f"?api-version={self.api_version}"
        )
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json",
        }
        data = self._post_json(url, payload, headers)
        choices = data.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            # OpenAI-style content parts: collect text.
            return "".join(part.get("text", "") for part in content if isinstance(part, dict)).strip()
        return str(content or "").strip()


class MockLLMClient:
    """Deterministic mock for unit tests.

    Pass either a fixed response or a list of responses returned in order.
    """

    def __init__(self, responses: str | list[str]) -> None:
        if isinstance(responses, str):
            self._queue: list[str] = [responses]
            self._sticky: bool = True
        else:
            self._queue = list(responses)
            self._sticky = False
        self.calls: list[dict] = []

    def complete(self, prompt: str, *, system: str | None = None, max_tokens: int = 256, temperature: float = 0.0) -> str:
        self.calls.append({"prompt": prompt, "system": system})
        if self._sticky:
            return self._queue[0]
        if not self._queue:
            raise AssertionError("MockLLMClient ran out of scripted responses")
        return self._queue.pop(0)


def build_client(provider: str, **kwargs) -> LLMClient:
    """Construct an LLM client by name.

    Recognized providers: `openai`, `gemini`, `modelhub`, `mock`. Unknown
    names raise.
    """
    name = provider.lower()
    if name == "openai":
        return OpenAIClient(**kwargs)
    if name == "gemini":
        return GeminiClient(**kwargs)
    if name == "modelhub":
        return ModelHubClient(**kwargs)
    if name == "mock":
        kwargs.pop("model", None)
        kwargs.setdefault("responses", "CORRECT")
        return MockLLMClient(**kwargs)
    raise ValueError(f"Unknown LLM provider: {provider}")
