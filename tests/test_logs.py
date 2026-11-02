import json
import logging

import pytest

from recall import logs
from recall.logs import bind, count, get_logger, measured, setup, user_ref

log = get_logger("recall.test")


@pytest.fixture
def events(tmp_path):
    setup(tmp_path, "test")

    def read():
        for handler in logging.getLogger().handlers:
            handler.flush()
        lines = (tmp_path / "logs" / "test.jsonl").read_text().splitlines()
        return [json.loads(line) for line in lines]

    yield read
    logging.basicConfig(force=True, handlers=[logging.NullHandler()])


def last(read, name):
    return [event for event in read() if event["event"] == name][-1]


def test_events_are_json_with_fields_and_context(events):
    with bind(user=user_ref(123), turn="t1"):
        log.info("reminder_sent", kind="promise")
    event = last(events, "reminder_sent")
    assert event["kind"] == "promise" and event["turn"] == "t1" and event["process"] == "test"
    assert event["level"] == "info" and "ts" in event and event["user"] == user_ref(123)
    assert "123" not in event["user"]


def test_measured_totals_add_up_through_nested_blocks(events):
    with measured(log, "turn") as stats:
        stats["saved"] = 1
        with measured(log, "llm_call"):
            count(input_tokens=10)
        with measured(log, "llm_call"):
            count(input_tokens=5)
    turn = last(events, "turn")
    assert turn["input_tokens"] == 15 and turn["saved"] == 1
    assert turn["outcome"] == "ok" and turn["duration_ms"] >= 0
    assert last(events, "llm_call")["input_tokens"] == 5


def test_expected_errors_are_warnings_without_a_stack(events):
    with pytest.raises(KeyError):
        with measured(log, "tool_call", expected=(KeyError,), tool="query"):
            raise KeyError("x")
    event = last(events, "tool_call")
    assert event["level"] == "warning" and event["outcome"] == "rejected"
    assert event["error_type"] == "KeyError" and "stack" not in event


def test_errors_keep_their_stack_but_never_their_message(events):
    with pytest.raises(ValueError):
        with measured(log, "turn"):
            raise ValueError("my secret note")
    event = last(events, "turn")
    assert event["outcome"] == "error" and event["error_type"] == "ValueError"
    assert event["stack"] and "my secret note" not in json.dumps(event)


def test_bot_tokens_are_redacted(events):
    logging.getLogger("telegram.ext").info("POST https://api.telegram.org/bot123456:AAbb-cc/getMe")
    assert "AAbb" not in json.dumps(events())


def test_log_files_are_private(events, tmp_path):
    assert (tmp_path / "logs" / "test.jsonl").stat().st_mode & 0o777 == 0o600
    assert logs.QUIET_LOGGERS


def test_a_full_turn_logs_metadata_and_no_memory_text(events, store):
    from test_agent import agent_with, response, text, tool

    store.create_table("notes", "Notes", [{"name": "body", "type": "text"}])
    store.add_row("notes", {"body": "secret diary line"})
    agent, _ = agent_with(
        response(tool("add_row", table="notes", values={"body": "secret diary line two"})),
        response(tool("query", sql="SELECT nothing FROM missing")),
        response(text("Saved secret diary line two.")),
    )
    agent.respond(store, "remember secret diary line two")
    turn = last(events, "turn")
    assert turn["kind"] == "chat" and turn["tools"] == ["add_row", "query"] and turn["saved"] == 1
    assert turn["llm_calls"] == 3 and turn["input_tokens"] == 3
    assert last(events, "tool_call")["outcome"] == "rejected"
    assert all(event["turn"] == turn["turn"] for event in events() if event["event"] == "llm_call")
    assert "secret" not in json.dumps(events())
