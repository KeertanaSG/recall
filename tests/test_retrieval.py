import json
from types import SimpleNamespace

import anthropic
import httpx
import pytest

from recall.retrieval import MAX_ATTEMPTS, Retriever, fuse
from recall.store import ToolError


def reply(payload, stop_reason="end_turn"):
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
    )


class FakeClient:
    """Answers plan and grade steps in order, recording each prompt."""

    def __init__(self, plan=None, grades=()):
        self.plan = plan or {"queries": [], "keywords": []}
        self.grades = list(grades)
        self.prompts = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        prompt = kwargs["messages"][0]["content"]
        self.prompts.append(prompt)
        if isinstance(self.plan, Exception):
            raise self.plan
        if "Entries:" not in prompt:
            return reply(self.plan)
        grade = self.grades.pop(0)
        if isinstance(grade, Exception):
            raise grade
        return reply(grade)


@pytest.fixture
def notes(store):
    store.create_table("notes", "Notes", [{"name": "body", "type": "text", "required": True}])
    for body in ("lent Arjun 2000 for tickets", "cab to airport", "watched dune"):
        store.add_row("notes", {"body": body}, body)
    store.turn_id = None
    return store


def row(table, row_id):
    return {"table": table, "row": {"id": row_id}}


def test_fusion_favours_rows_found_by_several_searches():
    merged = fuse([[row("a", 1), row("a", 2)], [row("a", 2), row("a", 3)]])
    assert [r["row"]["id"] for r in merged] == [2, 1, 3]


def test_grading_keeps_only_relevant_rows(notes):
    client = FakeClient(grades=[{"relevant": ["notes#1"]}])
    result = Retriever(client, "m").search(notes, "lent money")
    assert [r["row"]["id"] for r in result["results"]] == [1]
    assert result["filtered_for_relevance"] and not result["nothing_relevant"]


def test_keywords_find_rows_that_embeddings_miss(notes):
    client = FakeClient(plan={"queries": [], "keywords": ["arjun"]}, grades=[{"relevant": ["notes#1"]}])
    result = Retriever(client, "m").search(notes, "unrelated words entirely")
    assert result["results"][0]["row"]["body"] == "lent Arjun 2000 for tickets"


def test_nothing_relevant_retries_with_a_new_angle_then_says_so(notes):
    client = FakeClient(
        plan={"queries": ["cab"], "keywords": []},
        grades=[{"relevant": []}, {"relevant": []}],
    )
    result = Retriever(client, "m", max_attempts=2).search(notes, "passport number")
    assert result["nothing_relevant"] and result["results"] == []
    assert len(client.prompts) == 4
    assert "Already tried" in client.prompts[2]


def test_default_gives_up_after_one_pass(notes):
    client = FakeClient(grades=[{"relevant": []}])
    assert Retriever(client, "m").search(notes, "passport number")["nothing_relevant"]
    assert len(client.prompts) == 2 * MAX_ATTEMPTS


def test_a_retry_can_succeed(notes):
    client = FakeClient(grades=[{"relevant": []}, {"relevant": ["notes#2"]}])
    result = Retriever(client, "m", max_attempts=2).search(notes, "cab")
    assert [r["row"]["id"] for r in result["results"]] == [2]


def test_invented_keys_from_the_grader_are_ignored(notes):
    client = FakeClient(grades=[{"relevant": ["notes#99", "secrets#1"]}, {"relevant": []}])
    assert Retriever(client, "m").search(notes, "cab")["results"] == []


@pytest.mark.parametrize(
    "failure",
    [anthropic.APIConnectionError(request=httpx.Request("POST", "https://x")), "not json"],
)
def test_search_survives_a_failed_grade(notes, failure):
    client = FakeClient(grades=[failure])
    result = Retriever(client, "m").search(notes, "cab")
    assert result["results"] and not result["filtered_for_relevance"]


def test_search_survives_a_failed_plan(notes):
    client = FakeClient(plan=anthropic.APIConnectionError(request=httpx.Request("POST", "https://x")))
    assert Retriever(client, "m").search(notes, "cab")["results"]


def test_search_survives_an_embedding_outage(notes, embedder):
    embedder.failing = True
    client = FakeClient(plan={"queries": [], "keywords": ["cab"]}, grades=[{"relevant": ["notes#2"]}])
    assert Retriever(client, "m").search(notes, "cab")["results"][0]["row"]["id"] == 2


def test_table_filter_applies_to_both_searches(notes):
    notes.create_table("films", "Films", [{"name": "title", "type": "text"}])
    notes.add_row("films", {"title": "cab movie"})
    client = FakeClient(plan={"queries": [], "keywords": ["cab"]}, grades=[{"relevant": ["films#1"]}])
    result = Retriever(client, "m").search(notes, "cab", table="films")
    assert {r["table"] for r in result["results"]} == {"films"}
    assert "notes#" not in client.prompts[1]


def test_keyword_search_treats_wildcards_literally(notes):
    assert notes.keyword_search(["%"]) == []
    assert notes.keyword_search(["_"]) == []
    assert notes.keyword_search([" "]) == []


def test_keyword_search_covers_reminders(notes):
    from datetime import datetime, timezone

    notes.add_reminder("send Priya the file", datetime(2026, 12, 1, tzinfo=timezone.utc), "promise", False)
    assert notes.keyword_search(["priya"])[0]["table"] == "_reminders"


def test_required_columns_are_enforced(store):
    store.create_table(
        "movies",
        "Films",
        [{"name": "title", "type": "text", "required": True}, {"name": "rating", "type": "real"}],
    )
    with pytest.raises(ToolError, match="requires"):
        store.add_row("movies", {"rating": 9})
    row_id = store.add_row("movies", {"title": "Dune"})
    assert store.get_row("movies", row_id)["rating"] is None
    with pytest.raises(ToolError, match="can't be emptied"):
        store.update_row("movies", row_id, {"title": None}, "oops")
    store.add_column("movies", "genre", "text")
    assert store.get_row("movies", row_id)["genre"] is None
    assert any(c["required"] for c in store.tables()[0]["columns"])
