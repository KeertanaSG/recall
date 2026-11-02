from datetime import UTC, datetime

import pytest

from recall.cli import back_up, demo, seed
from recall.config import Settings
from recall.core.nudges import MAX_AGE, MIN_AGE, find_candidates
from recall.core.retrieval import result_key
from recall.storage.store import Store, user_db_path

NOW = datetime(2026, 11, 2, 9, 0, tzinfo=UTC)


def test_seed_makes_open_items_for_nudges_and_leaves_resolved_ones_for_the_judge(store):
    seed(store, NOW)
    found = {result_key(c) for c in find_candidates(store, NOW)}
    assert "loans#1" in found
    assert "loans#4" not in found  # Sam repaid this one.


def test_seeded_rows_keep_their_backdated_times(store):
    seed(store, NOW)
    row = store.get_row("loans", 1)
    assert (NOW - datetime.fromisoformat(row["created_at"])).days == 34
    assert MIN_AGE <= NOW - datetime.fromisoformat(row["created_at"]) <= MAX_AGE


def test_undo_can_not_erase_the_seed(store):
    seed(store, NOW)
    assert store.undo() is False
    assert store.get_row("loans", 1)


def test_refuses_to_replace_memory_without_fresh(tmp_path, embedder, monkeypatch):
    with Store(user_db_path(tmp_path, 1), embedder, "Asia/Kolkata") as store:
        store.create_table("mine", "Mine", [{"name": "body", "type": "text"}])
    config = Settings("t", frozenset({1}), tmp_path)
    monkeypatch.setattr("recall.cli.Settings.from_env", lambda: config)
    monkeypatch.setattr("sys.argv", ["recall-demo"])
    with pytest.raises(SystemExit, match="isn't empty"):
        demo()
    with Store(user_db_path(tmp_path, 1), embedder, "Asia/Kolkata") as store:
        assert [table["name"] for table in store.tables()] == ["mine"]


def test_backup_is_private_and_readable(tmp_path, embedder):
    with Store(user_db_path(tmp_path, 1), embedder, "Asia/Kolkata") as store:
        store.create_table("mine", "Mine", [{"name": "body", "type": "text"}])
        store.add_row("mine", {"body": "kept"})
        path = back_up(Settings("t", frozenset({1}), tmp_path), 1, NOW)
    import pathlib
    import sqlite3

    assert pathlib.Path(path).stat().st_mode & 0o777 == 0o600
    assert sqlite3.connect(path).execute("SELECT body FROM mine").fetchone() == ("kept",)
