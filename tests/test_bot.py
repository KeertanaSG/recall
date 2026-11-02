from datetime import datetime, timedelta, timezone
from pathlib import Path

from recall.bot import collect_due, reminder_keyboard, split_message
from recall.config import Settings
from recall.service import LocalStoreProvider, RecallService
from recall.store import Store, user_db_path

NOON_IST = datetime(2026, 11, 1, 6, 30, tzinfo=timezone.utc)


def make_service(tmp_path: Path, embedder, limit: int = 5) -> RecallService:
    config = Settings("token", frozenset({1}), tmp_path, "Asia/Kolkata", max_reminders_per_day=limit)
    return RecallService(config, None, LocalStoreProvider(config, embedder), None, None)


def test_collect_due_applies_the_daily_limit_after_sends(tmp_path, embedder):
    service = make_service(tmp_path, embedder, limit=2)
    with Store(user_db_path(tmp_path, 1), embedder, "Asia/Kolkata") as store:
        ids = [store.add_reminder(f"r{i}", NOON_IST, "promise", False) for i in range(3)]
        store.mark_sent(ids[0])
        store.conn.execute(
            "UPDATE _reminders SET sent_at = ? WHERE id = ?", (NOON_IST.isoformat(), ids[0])
        )
    due = collect_due(service, 1, NOON_IST + timedelta(minutes=1))
    assert [r["id"] for r in due] == [ids[1]]


def test_overnight_reminder_waits_for_morning(tmp_path, embedder):
    service = make_service(tmp_path, embedder)
    night = datetime(2026, 11, 1, 18, 0, tzinfo=timezone.utc)  # 23:30 in India
    with Store(user_db_path(tmp_path, 1), embedder, "Asia/Kolkata") as store:
        store.add_reminder("x", night, "deadline", False)
    assert collect_due(service, 1, night) == []
    morning = night + timedelta(hours=9)  # 08:30 in India
    assert len(collect_due(service, 1, morning)) == 1


def test_collect_due_retries_failed_embeddings(tmp_path, embedder):
    service = make_service(tmp_path, embedder)
    with Store(user_db_path(tmp_path, 1), embedder, "Asia/Kolkata") as store:
        store.create_table("notes", "Notes", [{"name": "body", "type": "text"}])
        embedder.failing = True
        store.add_row("notes", {"body": "ferry"})
        embedder.failing = False
    collect_due(service, 1, NOON_IST)
    with Store(user_db_path(tmp_path, 1), embedder, "Asia/Kolkata") as store:
        assert store.search("ferry")[0]["row"]["body"] == "ferry"


def test_keyboard_only_offers_stop_on_proactive_reminders():
    proactive = reminder_keyboard({"id": 1, "kind": "follow_up", "explicit": 0})
    requested = reminder_keyboard({"id": 1, "kind": "follow_up", "explicit": 1})
    assert len(proactive.inline_keyboard) == 2
    assert proactive.inline_keyboard[1][0].text == "Stop follow up reminders"
    assert len(requested.inline_keyboard) == 1


def test_split_message_keeps_all_text():
    text = "x" * 9000
    assert "".join(split_message(text)) == text
    assert all(len(part) <= 4000 for part in split_message(text))
    assert split_message("") == [""]
