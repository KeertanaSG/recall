from types import SimpleNamespace

import anthropic
import httpx
import pytest

from recall.llm import FALLBACK_BETA, FAST_BETA, create_message


class FakeClient:
    def __init__(self, *failures):
        self.failures = list(failures)
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.failures:
            raise self.failures.pop(0)
        return "response"


def rate_limited():
    request = httpx.Request("POST", "https://x")
    return anthropic.RateLimitError("slow down", response=httpx.Response(429, request=request), body=None)


def test_fast_mode_sends_both_betas():
    client = FakeClient()
    assert create_message(client, "claude-opus-5-5", True, max_tokens=10) == "response"
    assert client.calls[0]["speed"] == "fast"
    assert client.calls[0]["betas"] == [FALLBACK_BETA, FAST_BETA]
    assert client.calls[0]["fallbacks"] == "default"


def test_standard_speed_omits_fast_mode():
    client = FakeClient()
    create_message(client, "claude-opus-5-5", False)
    assert "speed" not in client.calls[0]
    assert client.calls[0]["betas"] == [FALLBACK_BETA]


def test_rate_limited_fast_mode_falls_back_to_standard():
    client = FakeClient(rate_limited())
    assert create_message(client, "claude-opus-5-5", True) == "response"
    assert "speed" not in client.calls[1]


def test_rate_limit_at_standard_speed_is_raised():
    client = FakeClient(rate_limited())
    with pytest.raises(anthropic.RateLimitError):
        create_message(client, "claude-opus-5-5", False)


def test_haiku_gets_neither_fallback_nor_fast_mode():
    client = FakeClient()
    create_message(client, "claude-haiku-5-5", True)
    assert "speed" not in client.calls[0] and "fallbacks" not in client.calls[0]
