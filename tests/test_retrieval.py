import json
import sqlite3
from datetime import UTC
from types import SimpleNamespace

import anthropic
import httpx
import pytest

from recall.core.retrieval import Retriever, fuse, hybrid_search
from recall.llm.claude import ClaudeLLM
from recall.storage.base import EmbeddingError
from recall.storage.store import LocalEmbedder, Store, ToolError


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
    result = Retriever(ClaudeLLM(client), "m").search(notes, "lent money")
    assert [r["row"]["id"] for r in result["results"]] == [1]
    assert result["filtered_for_relevance"] and not result["nothing_relevant"]


def test_keywords_find_rows_that_embeddings_miss(notes):
    client = FakeClient(
        plan={"queries": [], "keywords": ["arjun"]}, grades=[{"relevant": ["notes#1"]}]
    )
    result = Retriever(ClaudeLLM(client), "m").search(notes, "unrelated words entirely")
    assert result["results"][0]["row"]["body"] == "lent Arjun 2000 for tickets"


def test_default_gives_up_after_one_pass(notes):
    client = FakeClient(grades=[{"relevant": []}])
    assert Retriever(ClaudeLLM(client), "m").search(notes, "passport number")["nothing_relevant"]
    assert len(client.prompts) == 2


def test_invented_keys_from_the_grader_are_ignored(notes):
    client = FakeClient(grades=[{"relevant": ["notes#99", "secrets#1"]}, {"relevant": []}])
    assert Retriever(ClaudeLLM(client), "m").search(notes, "cab")["results"] == []


@pytest.mark.parametrize(
    "failure",
    [anthropic.APIConnectionError(request=httpx.Request("POST", "https://x")), "not json"],
)
def test_search_survives_a_failed_grade(notes, failure):
    client = FakeClient(grades=[failure])
    result = Retriever(ClaudeLLM(client), "m").search(notes, "cab")
    assert result["results"] and not result["filtered_for_relevance"]


def test_search_survives_a_failed_plan(notes):
    client = FakeClient(
        plan=anthropic.APIConnectionError(request=httpx.Request("POST", "https://x"))
    )
    assert Retriever(ClaudeLLM(client), "m").search(notes, "cab")["results"]


def test_search_survives_an_embedding_outage(notes, embedder):
    embedder.failing = True
    client = FakeClient(
        plan={"queries": [], "keywords": ["cab"]}, grades=[{"relevant": ["notes#2"]}]
    )
    assert Retriever(ClaudeLLM(client), "m").search(notes, "cab")["results"][0]["row"]["id"] == 2


def test_table_filter_applies_to_both_searches(notes):
    notes.create_table("films", "Films", [{"name": "title", "type": "text"}])
    notes.add_row("films", {"title": "cab movie"})
    client = FakeClient(
        plan={"queries": [], "keywords": ["cab"]}, grades=[{"relevant": ["films#1"]}]
    )
    result = Retriever(ClaudeLLM(client), "m").search(notes, "cab", table="films")
    assert {r["table"] for r in result["results"]} == {"films"}
    assert "notes#" not in client.prompts[1]


def test_keyword_search_treats_wildcards_literally(notes):
    assert notes.keyword_search(["%"]) == []
    assert notes.keyword_search(["_"]) == []
    assert notes.keyword_search([" "]) == []


def test_keyword_search_covers_reminders(notes):
    from datetime import datetime

    notes.add_reminder("send Priya the file", datetime(2026, 12, 1, tzinfo=UTC), "promise", False)
    assert notes.keyword_search(["priya"])[0]["table"] == "_reminders"


def ids(results):
    return [(r["table"], r["row"]["id"]) for r in results]


def test_keyword_search_ranks_by_relevance_not_recency(notes):
    notes.add_row("notes", {"body": "cab again"}, "cab again")
    # The older row matches both words, so it beats the newer row that matches one.
    assert ids(notes.keyword_search(["arjun", "tickets", "cab"]))[0] == ("notes", 1)


def test_keyword_search_ranks_across_tables_together(notes):
    from datetime import datetime

    for n in range(5):
        notes.add_reminder(f"call the bank {n}", datetime(2026, 12, 1, tzinfo=UTC), "other", True)
    # Reminders used to fill the list before any other table was read.
    assert ("notes", 1) in ids(notes.keyword_search(["bank", "arjun"], limit=3))


def test_keyword_search_matches_inside_words_unless_whole_word(notes):
    assert ids(notes.keyword_search(["airp"])) == [("notes", 2)]
    assert ids(notes.keyword_search(["arj"])) == [("notes", 1)]
    assert notes.keyword_search(["arj"], whole_word=True) == []
    assert ids(notes.keyword_search(["arjun"], whole_word=True)) == [("notes", 1)]
    assert ids(notes.keyword_search(["cab"], table="notes")) == [("notes", 2)]
    assert notes.keyword_search(["cab"], table="_reminders") == []


def test_keyword_search_finds_terms_under_three_characters(notes):
    notes.add_row("notes", {"body": "coffee with Al"}, "coffee with Al")
    notes.add_row("notes", {"body": "50% off at the 2_for_1 sale"}, "")
    assert ids(notes.keyword_search(["Al"], whole_word=True)) == [("notes", 4)]
    assert ids(notes.keyword_search(["al"], table="notes")) == [("notes", 5), ("notes", 4)]
    assert notes.keyword_search(["al"], table="films") == []
    # Ranked matches lead, then short-term matches, each row once.
    assert ids(notes.keyword_search(["dune", "al"]))[0] == ("notes", 3)
    assert len(ids(notes.keyword_search(["coffee", "Al"]))) == len(
        set(ids(notes.keyword_search(["coffee", "Al"])))
    )
    assert ids(notes.keyword_search(["%"])) == [("notes", 5)]
    assert ids(notes.keyword_search(["_"])) == [("notes", 5)]


def test_keyword_index_follows_updates_undo_and_forget(notes):
    notes.update_row("notes", 2, {"body": "train to airport"}, "wrong mode")
    assert ids(notes.keyword_search(["train"])) == [("notes", 2)]
    notes.undo()
    assert notes.keyword_search(["train"]) == []
    notes.forget("notes", 1)
    assert notes.keyword_search(["arjun"]) == []
    assert notes.conn.execute("SELECT COUNT(*) FROM _fts").fetchone()[0] == 2


def test_rows_without_text_stay_out_of_the_keyword_index(store):
    store.create_table("steps", "Daily steps", [{"name": "count", "type": "integer"}])
    store.add_row("steps", {"count": 9000})
    assert store.keyword_search(["9000"]) == []
    assert store.keyword_search(["count"]) == []


def test_existing_databases_get_a_keyword_index(notes, embedder):
    notes.conn.execute("DROP TABLE _fts")
    notes.conn.execute("DELETE FROM notes WHERE id = 3")  # An entry left behind by old data.
    notes.close()
    with Store(notes.path, embedder, "Asia/Kolkata") as reopened:
        assert ids(reopened.keyword_search(["arjun"])) == [("notes", 1)]
        assert reopened.keyword_search(["dune"]) == []


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


def test_a_locked_database_is_a_retryable_tool_error(store):
    store.create_table("notes", "Notes", [{"name": "body", "type": "text"}])
    other = sqlite3.connect(store.path)
    other.execute("BEGIN IMMEDIATE")
    store.conn.execute("PRAGMA busy_timeout=20")
    with pytest.raises(ToolError, match="busy"):
        store.add_row("notes", {"body": "x"})
    other.rollback()
    assert store.add_row("notes", {"body": "x"}) == 1


def test_a_failed_embedding_load_is_an_embedding_error(tmp_path, monkeypatch):
    def fails(*_args, **_kwargs):
        raise OSError("no network")

    monkeypatch.setattr("recall.storage.store.TextEmbedding", fails)
    with pytest.raises(EmbeddingError):
        LocalEmbedder(tmp_path).embed(["x"], "query")


def test_hybrid_search_drops_weak_semantic_matches(notes):
    # Unrelated rows still get a similarity score; they must not pad the list to the limit.
    assert [r["row"]["body"] for r in hybrid_search(notes, "airport", 8)] == ["cab to airport"]
