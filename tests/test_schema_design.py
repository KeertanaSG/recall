import json
from types import SimpleNamespace

import httpx
import openai
import pytest
from conftest import completion, fake_llm

from recall.core.tools import TableDesigner
from recall.storage.store import ToolError

DESIGN = {
    "name": "movies",
    "description": "Movies watched",
    "columns": [
        {"name": "title", "type": "text", "description": "", "required": True},
        {"name": "watched_with", "type": "text", "description": "", "required": False},
    ],
}


def client_returning(*responses):
    queue = list(responses)
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    return SimpleNamespace(complete=create, calls=calls)


def reply(text, finish="stop"):
    return completion(text, finish=finish)


def test_design_creates_table_and_reports_columns(store):
    client = client_returning(reply(json.dumps(DESIGN)))
    result = TableDesigner(fake_llm(client), "anthropic/claude-opus-5-5").create(
        store, "movies", "saw Dune"
    )
    assert result["created"] == "movies"
    assert result["columns"][0] == {"name": "title", "type": "text", "required": True}
    assert client.calls[0]["model"] == "anthropic/claude-opus-5-5"
    assert store.tables()[0]["columns"][0]["required"] == 1


def test_design_is_told_about_existing_tables(store, expenses):
    client = client_returning(reply(json.dumps(DESIGN)))
    TableDesigner(fake_llm(client), "m").create(store, "movies", "saw Dune")
    assert "expenses: Money the user spent" in client.calls[0]["messages"][1]["content"]


def test_duplicate_guard_still_applies_to_designed_tables(store):
    client = client_returning(
        reply(json.dumps(DESIGN)), reply(json.dumps({**DESIGN, "name": "movie_log"}))
    )
    designer = TableDesigner(fake_llm(client), "m")
    designer.create(store, "movies", "saw Dune")
    with pytest.raises(ToolError, match="already covers"):
        designer.create(store, "movies", "saw Up", confirmed_distinct=True)


@pytest.mark.parametrize(
    "failure",
    [
        openai.APIConnectionError(request=httpx.Request("POST", "https://x")),
        reply("not json"),
        reply("{", finish="length"),
    ],
)
def test_failed_design_is_a_tool_error_and_creates_nothing(store, failure):
    with pytest.raises(ToolError, match="Try once more"):
        TableDesigner(fake_llm(client_returning(failure)), "m").create(store, "movies", "saw Dune")
    assert store.tables() == []
