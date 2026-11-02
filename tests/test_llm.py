from types import SimpleNamespace

import anthropic
import httpx
import pytest

from recall.llm.claude import FALLBACK_BETA, create_message


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
    return anthropic.RateLimitError(
        "slow down", response=httpx.Response(429, request=request), body=None
    )


def test_sonnet_and_opus_get_fallback():
    client = FakeClient()
    create_message(client, "claude-opus-5-5")
    assert client.calls[0]["betas"] == [FALLBACK_BETA]


def test_rate_limit_is_raised():
    client = FakeClient(rate_limited())
    with pytest.raises(anthropic.RateLimitError):
        create_message(client, "claude-opus-5-5")


def test_haiku_gets_no_fallback():
    client = FakeClient()
    create_message(client, "claude-haiku-5-5")
    assert "fallbacks" not in client.calls[0]
