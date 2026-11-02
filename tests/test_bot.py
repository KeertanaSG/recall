import asyncio
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from recall.llm.claude import ClaudeLLM
from recall.channels.telegram import (
    collect_due,
    deliver_due,
    forget_page,
    handle_button,
    handle_nudge,
    handle_photo,
    reminder_keyboard,
    split_message,
)
from recall.core.agent import Reply
from recall.config import Settings
from recall.core.nudges import Nudger
from recall.service import RecallService
from recall.storage.store import SqliteStoreProvider
from recall.storage.store import Store, user_db_path
from recall.usage import Limit

NOON_IST = datetime(2026, 11, 1, 6, 30, tzinfo=timezone.utc)


def make_service(tmp_path: Path, embedder, limit: int = 5) -> RecallService:
    config = Settings("token", frozenset({1}), tmp_path, "Asia/Kolkata", max_reminders_per_day=limit)
    return RecallService(config, None, SqliteStoreProvider(config, embedder), None, None)


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


def fake_chat(service, **message):
    message = SimpleNamespace(
        chat=SimpleNamespace(send_action=AsyncMock()), reply_text=AsyncMock(), **message
    )
    update = SimpleNamespace(effective_user=SimpleNamespace(id=1), message=message)
    data = {"settings": service.settings, "service": service, "locks": defaultdict(asyncio.Lock)}
    bot = SimpleNamespace(send_message=AsyncMock())
    return update, SimpleNamespace(bot_data=data, bot=bot, user_data={}, args=[])


def test_photo_goes_through_the_service_with_its_caption(tmp_path, embedder):
    service = make_service(tmp_path, embedder)
    seen = {}

    def respond(store, text, image=None):
        seen.update(text=text, image=image)
        return Reply("Saved.")

    service.agent = SimpleNamespace(respond=respond)
    service.limiter = SimpleNamespace(spend=lambda _user: Limit.NONE)
    file = SimpleNamespace(download_as_bytearray=AsyncMock(return_value=bytearray(b"jpeg")))
    small = SimpleNamespace(get_file=AsyncMock())
    large = SimpleNamespace(get_file=AsyncMock(return_value=file))
    status = SimpleNamespace(delete=AsyncMock())
    update, context = fake_chat(service, photo=[small, large], caption="lunch")
    update.message.reply_text.return_value = status
    asyncio.run(handle_photo(update, context))
    assert seen == {"text": "[photo] lunch", "image": b"jpeg"}
    small.get_file.assert_not_called()
    update.message.reply_text.assert_any_await("Reading your photo…")
    update.message.reply_text.assert_awaited_with("Saved.")
    status.delete.assert_awaited_once()


def test_nudge_runs_from_scan_to_a_sent_message_with_stop_button(tmp_path, embedder):
    service = make_service(tmp_path, embedder)
    with service.stores.open(1) as store:
        store.create_table("loans", "Loans", [{"name": "person", "type": "text"}])
        row_id = store.add_row("loans", {"person": "Arjun"})
        saved = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat(timespec="seconds")
        store.conn.execute("UPDATE loans SET created_at = ? WHERE id = ?", (saved, row_id))
    verdict = json.dumps({"nudges": [{"key": "loans#1", "message": "Arjun owes you.", "importance": 3}]})
    response = SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=verdict)])
    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=lambda **_: response)))
    service.nudger = Nudger(ClaudeLLM(client), "haiku")
    with service.stores.open(1) as store:
        prefs = store.prefs()
        prefs.quiet_enabled = False
        store.save_prefs(prefs)

    update, context = fake_chat(service)
    asyncio.run(handle_nudge(update, context))
    (user_id, text), kwargs = context.bot.send_message.await_args
    assert user_id == 1 and text.startswith("Arjun owes you.\n\nFrom your notes\nArjun")
    assert kwargs["reply_markup"].inline_keyboard[1][0].text == "Stop nudge reminders"
    assert asyncio.run(deliver_due(context, 1, datetime.now(timezone.utc))) == 0


def seeded_notes(service, count):
    with service.stores.open(1) as store:
        store.create_table("notes", "Notes", [{"name": "body", "type": "text"}])
        return [("notes", store.add_row("notes", {"body": f"cab ride {i}"})) for i in range(count)]


def labels(keyboard):
    return [button.text for row in keyboard.inline_keyboard for button in row]


def test_forget_pages_through_every_match(tmp_path, embedder):
    service = make_service(tmp_path, embedder)
    _, context = fake_chat(service)
    context.user_data["forget"] = seeded_notes(service, 12)
    text, first = forget_page(context, 1, 0)
    assert text.startswith("12 matches.") and "narrows" not in text
    assert labels(first)[-2:] == ["Next", "Cancel"] and len(labels(first)) == 7
    _, last = forget_page(context, 1, 2)
    assert len(labels(last)) == 4 and "Back" in labels(last) and "Next" not in labels(last)


def test_forget_suggests_narrowing_a_long_list(tmp_path, embedder):
    service = make_service(tmp_path, embedder)
    _, context = fake_chat(service)
    context.user_data["forget"] = seeded_notes(service, 25)
    assert "narrows these down" in forget_page(context, 1, 0)[0]


def test_forget_paging_survives_a_restart(tmp_path, embedder):
    update, context = fake_chat(make_service(tmp_path, embedder))
    query = SimpleNamespace(data="fpage:1", answer=AsyncMock(), edit_message_text=AsyncMock())
    update.callback_query = query
    asyncio.run(handle_button(update, context))
    assert "expired" in query.edit_message_text.await_args.args[0]


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
