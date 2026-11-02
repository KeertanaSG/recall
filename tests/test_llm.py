import json
from types import SimpleNamespace as NS

import httpx
import openai
import pytest
from conftest import completion

from recall.llm.base import LLMError, Stop, ToolResult
from recall.llm.litellm_provider import LiteLLMProvider

SCHEMA = {"type": "object"}
TOOL = {"name": "save", "description": "Save", "input_schema": SCHEMA}


def provider(*answers):
    calls = []

    def complete(**kwargs):
        calls.append(kwargs)
        answer = answers[len(calls) - 1]
        if isinstance(answer, Exception):
            raise answer
        return answer

    return LiteLLMProvider(complete), calls


def test_chat_sends_tools_caching_effort_and_note():
    llm, calls = provider(completion("hi"))
    reply = llm.chat("anthropic/m", "system", [], [TOOL], "low", "note")
    sent = calls[0]
    assert sent["tools"][0]["function"] == {
        "name": "save",
        "description": "Save",
        "parameters": SCHEMA,
    }
    assert sent["reasoning_effort"] == "low" and sent["num_retries"] == 2
    system = sent["messages"][0]["content"]
    assert system[0]["cache_control"] == {"type": "ephemeral"} and system[1]["text"] == "note"
    assert reply.text == "hi" and reply.stop is Stop.DONE


@pytest.mark.parametrize(
    ("finish", "stop"),
    [
        ("stop", Stop.DONE),
        ("length", Stop.CUT_OFF),
        ("content_filter", Stop.REFUSED),
        ("tool_calls", Stop.TOOL_USE),
    ],
)
def test_finish_reasons_map_to_stops(finish, stop):
    llm, _ = provider(completion(finish=finish))
    assert llm.chat("m", "s", [], [], "low").stop is stop


def test_tool_calls_parse_and_results_become_tool_messages():
    llm, _ = provider(completion(calls=[("save", {"a": 1})]))
    reply = llm.chat("m", "s", [], [], "low")
    assert (reply.tool_calls[0].name, reply.tool_calls[0].input) == ("save", {"a": 1})
    assert llm.tool_results_message([ToolResult("1", "done")]) == [
        {"role": "tool", "tool_call_id": "1", "content": "done"}
    ]
    assert llm.assistant_message(reply)["role"] == "assistant"


def test_cached_tokens_are_not_counted_as_new_input(monkeypatch):
    counted = {}
    monkeypatch.setattr(
        "recall.llm.litellm_provider.count", lambda **numbers: counted.update(numbers)
    )
    answer = completion("hi")
    answer.usage = NS(prompt_tokens=100, completion_tokens=5, cache_read_input_tokens=80)
    provider(answer)[0].chat("m", "s", [], [], "low")
    assert counted["input_tokens"] == 20 and counted["cache_read_tokens"] == 80


def test_ask_json_requests_the_schema():
    llm, calls = provider(completion(json.dumps({"a": 1})))
    assert llm.ask_json("m", "s", "p", SCHEMA) == {"a": 1}
    assert calls[0]["response_format"]["json_schema"]["schema"] == SCHEMA


@pytest.mark.parametrize(
    "answer",
    [
        completion("{", finish="stop"),
        completion(None, finish="stop"),
        completion("{}", finish="length"),
        openai.APIConnectionError(request=httpx.Request("POST", "https://x")),
    ],
)
def test_ask_json_failures_are_llm_errors(answer):
    with pytest.raises(LLMError):
        provider(answer)[0].ask_json("m", "s", "p", SCHEMA)


def test_chat_failures_are_llm_errors():
    failure = openai.APIConnectionError(request=httpx.Request("POST", "https://x"))
    with pytest.raises(LLMError):
        provider(failure)[0].chat("m", "s", [], [], "low")


def test_photos_are_sent_as_data_urls():
    message = provider()[0].user_message("hi", b"jpeg")
    assert message["content"][0]["image_url"]["url"] == "data:image/jpeg;base64,anBlZw=="
    assert provider()[0].user_message("hi") == {"role": "user", "content": "hi"}
