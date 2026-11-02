import sqlite3
import zipfile
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path

import pytest

from recall.storage.store import (
    REMINDERS,
    Store,
    ToolError,
    delete_user_data,
    name_key,
    user_db_path,
)

NOW = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)


def new_turn(store):
    store.turn_id = None


def test_add_row_converts_types(store, expenses):
    row_id = store.add_row(
        expenses, {"item": "cab", "amount": "450", "spent_on": "2026-11-01"}, "cab 450"
    )
    row = store.get_row(expenses, row_id)
    assert row["amount"] == 450.0
    assert row["source_text"] == "cab 450"


@pytest.mark.parametrize(
    "values",
    [
        {"amount": "cheap"},
        {"spent_on": "next friday"},
        {"amount": float("nan")},
        {"nope": 1},
        {"item": None},
    ],
)
def test_add_row_rejects_bad_values(store, expenses, values):
    with pytest.raises(ToolError):
        store.add_row(expenses, values)


@pytest.mark.parametrize("name", ["coffees", "coffee_log", "my_coffee_tracker", "coffee_entries"])
def test_duplicate_table_names_are_blocked_even_when_forced(store, name):
    store.create_table("coffee", "Coffees drunk", [{"name": "kind", "type": "text"}])
    with pytest.raises(ToolError, match="already covers"):
        store.create_table(
            name, "Other", [{"name": "shop", "type": "text"}], confirmed_distinct=True
        )


@pytest.mark.parametrize(
    ("first", "second"), [("movies", "movie_log"), ("cities", "city"), ("books", "book_list")]
)
def test_name_key_matches_plurals(first, second):
    assert name_key(first) == name_key(second)


def test_name_key_keeps_distinct_subjects_apart():
    assert name_key("movies") != name_key("movie_reviews")
    assert name_key("glass") == {"glass"}


def test_similar_columns_need_confirmation(store):
    columns = [{"name": "title", "type": "text"}, {"name": "rating", "type": "real"}]
    store.create_table("movies", "Films watched", columns)
    with pytest.raises(ToolError, match="Similar tables"):
        store.create_table("shows", "TV series watched", columns)
    store.create_table("shows", "TV series watched", columns, confirmed_distinct=True)
    assert {t["name"] for t in store.tables()} == {"movies", "shows"}


def test_similar_description_needs_confirmation(store):
    store.create_table("cabs", "cab rides taken around town", [{"name": "fare", "type": "real"}])
    with pytest.raises(ToolError, match="Similar tables"):
        store.create_table(
            "taxi_trips", "cab rides taken around town", [{"name": "driver", "type": "text"}]
        )


def test_schema_check_survives_embedding_outage(store, embedder):
    store.create_table("cabs", "cab rides", [{"name": "fare", "type": "real"}])
    embedder.failing = True
    store.create_table("books", "books read", [{"name": "title", "type": "text"}])


@pytest.mark.parametrize(
    "name", ["_secret", "Drop", "a b", 'x"; DROP TABLE y', "sqlite_x", "id", ""]
)
def test_unsafe_identifiers_are_rejected(store, name):
    with pytest.raises(ToolError):
        store.create_table(name, "x", [{"name": "a", "type": "text"}])
    with pytest.raises(ToolError):
        store.create_table("safe", "x", [{"name": name, "type": "text"}])


def test_update_keeps_history(store):
    store.create_table(
        "profile", "Facts", [{"name": "fact", "type": "text"}, {"name": "value", "type": "text"}]
    )
    new_turn(store)
    row_id = store.add_row("profile", {"fact": "city", "value": "Bangalore"})
    new_turn(store)
    store.update_row("profile", row_id, {"value": "Pune"}, "moved")
    assert store.get_row("profile", row_id)["value"] == "Pune"
    change = store.row_history("profile", row_id)[-1]
    assert change["old_values"] == {"value": "Bangalore"}
    assert change["reason"] == "moved"


def test_undo_reverses_a_whole_turn_including_new_table(store):
    store.create_table("books", "Books", [{"name": "title", "type": "text"}])
    store.add_row("books", {"title": "Dune"})
    store.add_column("books", "rating", "real")
    assert store.undo()
    assert store.tables() == []
    assert store.conn.execute("SELECT COUNT(*) FROM _embeddings").fetchone()[0] == 0
    assert store.conn.execute("SELECT COUNT(*) FROM _vectors").fetchone()[0] == 0
    assert not store.undo()


def test_undo_restores_an_update_and_reindexes_it(store, expenses):
    row_id = store.add_row(expenses, {"item": "cab", "amount": 100})
    new_turn(store)
    store.update_row(expenses, row_id, {"item": "ferry"}, "fix")
    assert store.undo()
    assert store.get_row(expenses, row_id)["item"] == "cab"
    assert store.search("cab")[0]["row"]["id"] == row_id


def test_old_undo_button_is_refused(store, expenses):
    store.add_row(expenses, {"item": "first"})
    first_turn = store.turn_id
    new_turn(store)
    store.add_row(expenses, {"item": "second"})
    assert not store.undo(first_turn)
    assert store.tables()[0]["row_count"] == 2


def test_failed_write_rolls_back_cleanly(store, expenses, monkeypatch):
    monkeypatch.setattr(
        store, "_mark_stale", lambda *_: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    with pytest.raises(RuntimeError):
        store.add_row(expenses, {"item": "cab"})
    assert store.tables()[0]["row_count"] == 0
    assert (
        store.conn.execute("SELECT COUNT(*) FROM _history WHERE op = 'insert'").fetchone()[0] == 0
    )


def test_rows_save_during_embedding_outage_and_catch_up(store, embedder, expenses):
    embedder.failing = True
    store.add_row(expenses, {"item": "ferry ticket"})
    with pytest.raises(ToolError):
        store.search("ferry")
    embedder.failing = False
    assert store.search("ferry") == []
    store.sync_embeddings()
    assert store.search("ferry")[0]["row"]["item"] == "ferry ticket"


def test_exact_query_totals(store, expenses):
    for amount in (100.5, 200.25):
        store.add_row(expenses, {"item": "cab", "amount": amount})
    result = store.read_query("SELECT SUM(amount) AS total, group_concat(id) AS ids FROM expenses")
    assert result["rows"] == [{"total": 300.75, "ids": "1,2"}]


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE expenses",
        "DELETE FROM expenses",
        "UPDATE expenses SET amount = 0",
        "INSERT INTO expenses (item, created_at, updated_at) VALUES ('x', '', '')",
        "CREATE TABLE evil (a)",
        "ATTACH DATABASE '/tmp/other.sqlite' AS other",
        "PRAGMA writable_schema=ON",
        "SELECT * FROM _history",
        "SELECT * FROM _messages",
        "SELECT * FROM sqlite_master",
        "SELECT load_extension('x')",
        "SELECT 1; DROP TABLE expenses",
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT MAX(x) FROM c",
    ],
)
def test_unsafe_or_runaway_queries_fail(store, expenses, sql):
    store.add_row(expenses, {"item": "cab", "amount": 1})
    with pytest.raises(ToolError):
        store.read_query(sql)
    assert store.tables()[0]["row_count"] == 1


def test_reminders_are_queryable(store):
    store.add_reminder("Send file", NOW, "promise", False)
    assert store.read_query(f"SELECT text FROM {REMINDERS}")["rows"] == [{"text": "Send file"}]


def test_query_row_limit(store, expenses):
    rows = ",".join("('a','','')" for _ in range(205))
    store.conn.execute(f"INSERT INTO expenses (item, created_at, updated_at) VALUES {rows}")
    result = store.read_query("SELECT id FROM expenses")
    assert len(result["rows"]) == 200 and result["truncated"]


def test_search_finds_by_shared_words_and_filters_by_table(store, expenses):
    store.create_table("movies", "Films", [{"name": "title", "type": "text"}])
    store.add_row(expenses, {"item": "dune popcorn"})
    store.add_row("movies", {"title": "dune part two"})
    assert store.search("dune", table="movies")[0]["table"] == "movies"
    assert len(store.search("dune")) == 2


def test_forget_leaves_no_trace_anywhere_in_the_file(store, expenses):
    secret = "zebrasecretword"
    row_id = store.add_row(expenses, {"item": secret}, f"bought {secret}")
    store.add_reminder(f"pay {secret}", NOW, "other", False, expenses, row_id)
    store.log_exchange(f"bought {secret}", "saved")
    assert store.forget(expenses, row_id)
    assert store.search(secret) == []
    assert store.conn.execute(f"SELECT COUNT(*) FROM {REMINDERS}").fetchone()[0] == 0
    assert store.conn.execute("SELECT COUNT(*) FROM _messages").fetchone()[0] == 0
    store.close()
    for suffix in ("", "-wal"):
        path = Path(str(store.path) + suffix)
        if path.exists():
            assert secret.encode() not in path.read_bytes()
    store.conn = sqlite3.connect(store.path)


def test_undo_after_forget_never_restores_the_forgotten_row(store, expenses):
    kept = store.add_row(expenses, {"item": "kept"})
    new_turn(store)
    forgotten = store.add_row(expenses, {"item": "gone"})
    store.forget(expenses, forgotten)
    assert store.undo()
    assert store.get_row(expenses, kept) is None
    assert store.get_row(expenses, forgotten) is None


def test_export_contains_tables_reminders_and_history(store, expenses):
    store.add_row(expenses, {"item": "cab", "amount": 5})
    store.add_reminder("x", NOW, "other", False)
    with zipfile.ZipFile(BytesIO(store.export_zip())) as archive:
        assert set(archive.namelist()) == {"expenses.csv", "reminders.csv", "history.json"}
        assert "cab" in archive.read("expenses.csv").decode()


def test_users_are_separate_files(tmp_path, embedder):
    with Store(user_db_path(tmp_path, 1), embedder, "UTC") as first:
        first.create_table("a", "x", [{"name": "b", "type": "text"}])
    with Store(user_db_path(tmp_path, 2), embedder, "UTC") as second:
        assert second.tables() == []
    assert (user_db_path(tmp_path, 1).stat().st_mode & 0o777) == 0o600


def test_delete_user_data_removes_all_files(tmp_path, embedder):
    with Store(user_db_path(tmp_path, 1), embedder, "UTC") as store:
        store.create_table("a", "x", [{"name": "b", "type": "text"}])
    delete_user_data(tmp_path, 1)
    assert list(user_db_path(tmp_path, 1).parent.iterdir()) == []


def test_same_reminder_for_a_row_is_not_created_twice(store, expenses):
    row_id = store.add_row(expenses, {"item": "file"})
    add = lambda text, when, *link, source="msg": store.add_reminder(
        text, when, "promise", False, *link, source_text=source
    )
    first = add("a", NOW, expenses, row_id)
    assert add("b", NOW, expenses, row_id) == first
    assert add("c", NOW) == first
    assert add("d", NOW + timedelta(days=1), expenses, row_id) != first
    other_row = store.add_row(expenses, {"item": "slides"})
    assert add("e", NOW, expenses, other_row) != first
    assert add("f", NOW, source="another message") != first


def test_snooze_returns_reminder_to_pending(store):
    reminder = store.add_reminder("x", NOW, "other", False)
    store.mark_sent(reminder)
    later = NOW + timedelta(days=1)
    store.snooze(reminder, later)
    assert store.due_reminders(NOW) == []
    assert store.due_reminders(later)[0]["id"] == reminder


def test_mute_cancels_pending_proactive_only(store):
    proactive = store.add_reminder("a", NOW, "promise", False)
    requested = store.add_reminder("b", NOW, "promise", True)
    store.mute_kind("promise")
    assert [r["id"] for r in store.due_reminders(NOW)] == [requested]
    assert store.get_row(REMINDERS, proactive)["status"] == "cancelled"
    assert store.prefs().muted_kinds == ["promise"]
