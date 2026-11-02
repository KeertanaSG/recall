"""Every Telegram command and button, against a real service and database with Telegram faked."""

import asyncio
import io
import zipfile
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest
from telegram import Update
from telegram.error import BadRequest, Forbidden, TelegramError

from recall.access import Limit
from recall.channels import telegram as tg
from recall.config import Settings
from recall.core.agent import Reply, TurnFailed
from recall.service import RecallService
from recall.storage.store import SqliteStoreProvider


class Bot:
    def __init__(self, tmp_path, embedder):
        self.settings = Settings("t", frozenset({1}), tmp_path, "Asia/Kolkata")
        self.service = RecallService(
            self.settings, None, SqliteStoreProvider(self.settings, embedder), None, None
        )
        self.service.limiter = NS(spend=lambda _user: Limit.NONE)
        self.query = NS(
            answer=AsyncMock(),
            edit_message_text=AsyncMock(),
            edit_message_reply_markup=AsyncMock(),
            message=NS(reply_text=AsyncMock()),
        )
        self.message = NS(
            reply_text=AsyncMock(),
            reply_document=AsyncMock(),
            chat=NS(send_action=AsyncMock()),
            caption=None,
        )
        self.user = NS(id=1)
        self.context = NS(
            bot_data={
                "settings": self.settings,
                "service": self.service,
                "locks": defaultdict(asyncio.Lock),
            },
            bot=NS(send_message=AsyncMock()),
            user_data={},
            args=[],
        )

    def run(self, handler, *args):
        self.context.args = list(args)
        update = NS(effective_user=self.user, message=self.message, callback_query=self.query)
        asyncio.run(handler(update, self.context))

    def tap(self, data):
        self.query.data = data
        self.run(tg.handle_button)

    @property
    def said(self):
        return self.message.reply_text.await_args.args[0]

    @property
    def edited(self):
        return self.query.edit_message_text.await_args.args[0]

    def store(self):
        return self.service.stores.open(1)

    def note(self, body="cab to airport"):
        with self.store() as store:
            if not store.tables():
                store.create_table("notes", "Notes", [{"name": "body", "type": "text"}])
            store.turn_id = None
            return store.add_row("notes", {"body": body})


@pytest.fixture
def bot(tmp_path, embedder):
    return Bot(tmp_path, embedder)


@pytest.mark.parametrize(
    ("handler", "args", "expected"),
    [
        (tg.handle_help, [], "Text me anything"),
        (tg.handle_non_text, [], "text messages and photos"),
        (tg.handle_undo, [], "nothing to undo"),
        (tg.handle_forget, [], "Tell me what to forget"),
        (tg.handle_forget, ["zebra"], "couldn't find"),
        (tg.handle_delete_all, [], "can't be undone"),
        (tg.handle_timezone, [], "Your time zone is Asia/Kolkata"),
        (tg.handle_timezone, ["Mars/Base"], "don't know that zone"),
        (tg.handle_timezone, ["Europe/Paris"], "Time zone set to Europe/Paris"),
        (tg.handle_quiet, [], "Quiet hours: 22:00 to 08:00"),
        (tg.handle_quiet, ["off"], "Quiet hours are off"),
        (tg.handle_quiet, ["nope", "x"], "24-hour times"),
        (tg.handle_quiet, ["23:00", "07:00"], "Quiet hours are 23:00 to 07:00"),
        (tg.handle_pause, [], "paused"),
        (tg.handle_resume, [], "back on"),
        (tg.handle_unmute, ["promise"], "Muted reminder types: none"),
        (tg.handle_nudge, [], "Nothing's gone quiet"),
    ],
)
def test_command_replies(bot, handler, args, expected):
    bot.service.nudger = NS(scan=lambda store, force: [])
    bot.run(handler, *args)
    assert expected in bot.said


def test_settings_changes_persist(bot):
    bot.run(tg.handle_timezone, "Europe/Paris")
    bot.run(tg.handle_quiet, "off")
    bot.run(tg.handle_pause)
    with bot.store() as store:
        prefs = store.prefs()
    assert (prefs.timezone, prefs.quiet_enabled, prefs.paused) == ("Europe/Paris", False, True)


def test_quiet_status_when_off(bot):
    bot.run(tg.handle_quiet, "off")
    bot.run(tg.handle_quiet)
    assert "Quiet hours: off" in bot.said


def test_people_off_the_allowlist_get_nothing(bot):
    bot.user = NS(id=99)
    bot.run(tg.handle_help)
    bot.user = None
    bot.run(tg.handle_help)
    bot.message.reply_text.assert_not_awaited()


def test_undo_reverses_the_last_save(bot):
    bot.note()
    with bot.store() as store:
        store.add_row("notes", {"body": "mistake"})
    bot.run(tg.handle_undo)
    assert bot.said == "Undone."
    with bot.store() as store:
        assert store.get_row("notes", 2) is None


def test_export_sends_a_real_zip(bot):
    bot.note()
    bot.run(tg.handle_export)
    data = bot.message.reply_document.await_args.args[0].read()
    assert zipfile.ZipFile(io.BytesIO(data)).namelist()


def test_forget_erases_only_the_tapped_entry(bot):
    bot.note()
    bot.note("cab to office")
    bot.run(tg.handle_forget, "airport")
    labels = [
        b.text
        for row in bot.message.reply_text.await_args.kwargs["reply_markup"].inline_keyboard
        for b in row
    ]
    assert "cab to airport" in " ".join(labels) and labels[-1] == "Cancel"
    bot.tap("forget:notes:1")
    assert bot.edited == "Erased."
    bot.tap("forget:notes:1")
    assert bot.edited == "That entry is already gone."
    with bot.store() as store:
        assert store.get_row("notes", 1) is None and store.get_row("notes", 2)


def test_forget_skips_rows_erased_while_the_list_was_open(bot):
    bot.note()
    bot.context.user_data["forget"] = [("notes", 1), ("notes", 9)]
    text, keyboard = tg.forget_page(bot.context, 1, 0)
    assert text.startswith("2 matches") and len(keyboard.inline_keyboard) == 2
    bot.context.user_data["forget"] = [("notes", 1)]
    assert tg.forget_page(bot.context, 1, 0)[0].startswith("1 match.")


def test_forget_paging_button(bot):
    bot.context.user_data["forget"] = [("notes", bot.note(f"cab {i}")) for i in range(7)]
    bot.tap("fpage:1")
    assert bot.edited.startswith("7 matches")


def test_delete_all_needs_the_second_tap(bot):
    bot.note()
    bot.run(tg.handle_delete_all)
    assert bot.service.users() == [1]
    bot.tap("cancel")
    assert bot.edited == "Cancelled." and bot.service.users() == [1]
    bot.tap("delete_all")
    assert bot.edited == "Everything has been erased." and bot.service.users() == []


def test_unknown_button_does_nothing(bot):
    bot.tap("mystery")
    bot.query.edit_message_text.assert_not_awaited()


@pytest.mark.parametrize(
    ("data", "status", "reply"),
    [
        ("rem:done:1", "done", "Done."),
        ("rem:snooze:1", "pending", "Snoozed until tomorrow."),
        ("rem:mute:promise", "cancelled", "/unmute promise"),
    ],
)
def test_reminder_buttons(bot, data, status, reply):
    now = datetime.now(UTC)
    with bot.store() as store:
        store.add_reminder("x", now, "promise", False)
    bot.tap(data)
    assert reply in bot.query.message.reply_text.await_args.args[0]
    with bot.store() as store:
        assert store.get_row("_reminders", 1)["status"] == status


def test_unmute_turns_a_kind_back_on(bot):
    with bot.store() as store:
        store.mute_kind("promise")
    bot.run(tg.handle_unmute, "promise")
    assert bot.said == "promise reminders are back on."
    with bot.store() as store:
        assert store.prefs().muted_kinds == []


def responding(bot, result):
    def respond(store, text, image=None):
        if isinstance(result, Exception):
            raise result
        return result

    bot.service.agent = NS(respond=respond)


def test_text_reply_is_split_to_telegram_size(bot):
    bot.message.text = "hi"
    responding(bot, Reply("x" * 5000))
    bot.run(tg.handle_text)
    assert bot.message.reply_text.await_count == 2


def test_failed_turn_says_nothing_was_saved(bot):
    bot.message.text = "hi"
    responding(bot, TurnFailed("down"))
    bot.run(tg.handle_text)
    assert "nothing was saved" in bot.said


@pytest.mark.parametrize(
    ("limit", "expected"), [(Limit.USER, "You've reached"), (Limit.TOTAL, "Recall has reached")]
)
def test_daily_limits_are_explained(bot, limit, expected):
    bot.message.text = "hi"
    bot.service.limiter = NS(spend=lambda _user: limit)
    bot.run(tg.handle_text)
    assert expected in bot.said


def test_a_photo_that_will_not_download_asks_for_a_resend(bot):
    bot.message.photo = [NS(get_file=AsyncMock(side_effect=TelegramError("x")))]
    bot.run(tg.handle_photo)
    assert "couldn't open that photo" in bot.said


def test_typing_survives_a_telegram_error_and_stops_when_cancelled(monkeypatch):
    chat = NS(send_action=AsyncMock(side_effect=TelegramError("x")))
    monkeypatch.setattr(tg.asyncio, "sleep", AsyncMock(side_effect=asyncio.CancelledError))
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(tg.keep_typing(chat))


@pytest.mark.parametrize(
    ("error", "sent"), [(None, 1), (Forbidden("blocked"), 0), (TelegramError("down"), 0)]
)
def test_reminder_tick_delivers_or_keeps_it_pending(bot, error, sent):
    with bot.store() as store:
        store.add_reminder("call", datetime.now(UTC) - timedelta(minutes=1), "other", True)
    bot.context.bot.send_message.side_effect = error
    asyncio.run(tg.reminder_tick(bot.context))
    with bot.store() as store:
        assert (store.get_row("_reminders", 1)["sent_at"] is not None) == bool(sent)


@pytest.mark.parametrize("tick", [tg.reminder_tick, tg.nudge_tick])
def test_ticks_skip_a_busy_user(bot, tick):
    bot.note()
    bot.service.scan_nudges = lambda *_: pytest.fail("A busy user was scanned")

    async def run():
        async with bot.context.bot_data["locks"][1]:
            await tick(bot.context)

    asyncio.run(run())
    bot.context.bot.send_message.assert_not_awaited()


def test_nudge_tick_scans_and_one_failure_does_not_stop_it(bot):
    bot.service.users = lambda: [1, 2]
    scanned = []

    def scan(identity):
        scanned.append(identity.user_id)
        if identity.user_id == 1:
            raise RuntimeError("corrupt")

    bot.service.scan_nudges = scan
    asyncio.run(tg.nudge_tick(bot.context))
    assert scanned == [1, 2]


def test_nudge_held_by_quiet_rules_says_so(bot):
    bot.service.scan_nudges = lambda identity, force: [1]
    bot.run(tg.handle_nudge)
    assert "when it's allowed" in bot.said


def test_error_handler_copes_with_updates_that_have_no_message():
    asyncio.run(tg.on_error(object(), NS(error=RuntimeError("x"))))
    asyncio.run(tg.on_error(Update(1), NS(error=BadRequest("Something else"))))


def test_error_handler_tells_the_user():
    update = Update(1)
    update._effective_message = NS(reply_text=AsyncMock())
    asyncio.run(tg.on_error(update, NS(error=RuntimeError("x"))))
    assert "Something went wrong" in update._effective_message.reply_text.await_args.args[0]


def test_bot_starts_with_commands_and_handlers(tmp_path, monkeypatch):
    settings = Settings("1:fake", frozenset({1}), tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    application = tg.build_application(settings)
    assert application.handlers[0] and application.error_handlers
    bot = NS(set_my_commands=AsyncMock())
    asyncio.run(tg.on_start(NS(bot=bot)))
    assert bot.set_my_commands.await_args.args[0] == tg.COMMANDS

    started = []
    monkeypatch.setattr(tg.Settings, "from_env", lambda: settings)
    monkeypatch.setattr(tg, "setup", lambda *_: started.append("logs"))
    monkeypatch.setattr(
        tg, "build_application", lambda _s: NS(run_polling=lambda: started.append("polling"))
    )
    tg.main()
    assert started == ["logs", "polling"]
