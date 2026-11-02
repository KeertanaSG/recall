"""Telegram interface: commands, buttons and the reminder tick."""

import asyncio
import io
from collections import defaultdict
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, time, timedelta
from functools import wraps
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from telegram import BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.error import BadRequest, Forbidden, TelegramError
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from recall.access import OWNER, telegram_identity
from recall.config import Settings
from recall.core.agent import TurnFailed, row_date, short_date, summarize_row
from recall.core.nudges import select_sendable, start_of_local_day
from recall.core.retrieval import find_entries
from recall.factory import build_service
from recall.logs import get_logger, setup, user_ref
from recall.prompts import render
from recall.service import RateLimited, RecallService

log = get_logger(__name__)

TICK_SECONDS = 30
NUDGE_TICK_SECONDS = 3600
TELEGRAM_TIMEOUT_SECONDS = 20
TYPING_SECONDS = 4
TELEGRAM_LIMIT = 4000
FORGET_PAGE = 5
FORGET_MATCHES = 50
FORGET_NARROW_AT = 20
WELCOME = render("agent.welcome")

COMMANDS = [
    BotCommand("nudge", "Check for unfinished business"),
    BotCommand("undo", "Reverse the last save"),
    BotCommand("forget", "Erase one entry"),
    BotCommand("export", "Download everything"),
    BotCommand("delete_all", "Erase everything"),
    BotCommand("timezone", "Set your time zone"),
    BotCommand("quiet", "Set quiet hours"),
    BotCommand("pause", "Pause proactive messages"),
    BotCommand("resume", "Resume proactive messages"),
    BotCommand("unmute", "Turn a reminder type back on"),
    BotCommand("help", "Show help"),
]

Handler = Callable[[Update, ContextTypes.DEFAULT_TYPE], Awaitable[None]]


def allowed_only(handler: Handler) -> Handler:
    @wraps(handler)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user = update.effective_user
        if user is None or not context.bot_data["settings"].allows(user.id):
            log.info("user_ignored")
            return
        await handler(update, context)

    return wrapper


def open_store(service: RecallService, user_id: int):
    return service.access(telegram_identity(user_id), "use the bot", OWNER)


def user_store(context: ContextTypes.DEFAULT_TYPE, user_id: int):
    return open_store(context.bot_data["service"], user_id)


def split_message(text: str) -> list[str]:
    return [text[i : i + TELEGRAM_LIMIT] for i in range(0, len(text), TELEGRAM_LIMIT)] or [""]


async def keep_typing(chat) -> None:
    """Telegram clears the typing indicator after about five seconds, so keep renewing it."""
    while True:
        try:
            await chat.send_action(ChatAction.TYPING)
        except TelegramError:
            log.warning("typing_indicator_failed")
        await asyncio.sleep(TYPING_SECONDS)


@allowed_only
async def handle_help(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(WELCOME)


@allowed_only
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await run_turn(update, context, update.message.text)


@allowed_only
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        file = await update.message.photo[-1].get_file()  # The last size is the largest.
        image = bytes(await file.download_as_bytearray())
    except TelegramError:
        log.warning("photo_download_failed")
        await update.message.reply_text("I couldn't open that photo. Please send it again.")
        return
    status = await update.message.reply_text("Reading your photo…")
    try:
        await run_turn(update, context, f"[photo] {update.message.caption or ''}".strip(), image)
    finally:
        try:
            await status.delete()
        except TelegramError:
            log.warning("photo_status_delete_failed")


async def run_turn(
    update: Update, context: ContextTypes.DEFAULT_TYPE, text: str, image: bytes | None = None
) -> None:
    user_id = update.effective_user.id
    identity = telegram_identity(user_id)
    service: RecallService = context.bot_data["service"]

    typing = asyncio.create_task(keep_typing(update.message.chat))
    try:
        async with context.bot_data["locks"][user_id]:
            reply = await asyncio.to_thread(service.chat, identity, text, image)
    except RateLimited as exc:
        await update.message.reply_text(str(exc))
        return
    except TurnFailed:
        await update.message.reply_text(
            "Something went wrong, so nothing was saved. Please resend that."
        )
        return
    finally:
        typing.cancel()

    for part in split_message(reply.render()):
        await update.message.reply_text(part)


@allowed_only
async def handle_non_text(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("I can only read text messages and photos for now.")


@allowed_only
async def handle_undo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id

    def undo():
        with user_store(context, user_id) as store:
            return store.undo()

    async with context.bot_data["locks"][user_id]:
        undone = await asyncio.to_thread(undo)
    await update.message.reply_text("Undone." if undone else "There's nothing to undo.")


@allowed_only
async def handle_forget(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    words = " ".join(context.args)
    if not words:
        await update.message.reply_text("Tell me what to forget, like /forget cab to airport")
        return

    def find():
        with user_store(context, update.effective_user.id) as store:
            return find_entries(store, words, FORGET_MATCHES)

    matches = await asyncio.to_thread(find)
    if not matches:
        await update.message.reply_text("I couldn't find anything like that.")
        return

    context.user_data["forget"] = [(match["table"], match["row"]["id"]) for match in matches]
    text, keyboard = forget_page(context, update.effective_user.id, 0)
    await update.message.reply_text(text, reply_markup=keyboard)


def forget_page(
    context: ContextTypes.DEFAULT_TYPE, user_id: int, page: int
) -> tuple[str, InlineKeyboardMarkup]:
    matches = context.user_data["forget"]
    start = page * FORGET_PAGE
    buttons = []
    with user_store(context, user_id) as store:
        for table, row_id in matches[start : start + FORGET_PAGE]:
            row = store.get_row(table, row_id)
            if row is None:
                continue
            details = summarize_row(row, store.boolean_columns(table), with_date=False)
            label = f"{short_date(row_date(row))} · {details}"[:60]
            buttons.append([InlineKeyboardButton(label, callback_data=f"forget:{table}:{row_id}")])
    pages = []
    if page > 0:
        pages.append(InlineKeyboardButton("Back", callback_data=f"fpage:{page - 1}"))
    if start + FORGET_PAGE < len(matches):
        pages.append(InlineKeyboardButton("Next", callback_data=f"fpage:{page + 1}"))
    if pages:
        buttons.append(pages)
    buttons.append([InlineKeyboardButton("Cancel", callback_data="cancel")])

    count = "1 match" if len(matches) == 1 else f"{len(matches)} matches"
    hint = " Adding a word or date narrows these down." if len(matches) > FORGET_NARROW_AT else ""
    text = f"{count}.{hint} Which one should I erase for good? This also removes its history and can't be undone."
    return text, InlineKeyboardMarkup(buttons)


@allowed_only
async def handle_export(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    with user_store(context, update.effective_user.id) as store:
        data = store.export_zip()
    await update.message.reply_document(io.BytesIO(data), filename="recall-export.zip")


@allowed_only
async def handle_delete_all(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("Yes, erase everything", callback_data="delete_all")],
            [InlineKeyboardButton("Cancel", callback_data="cancel")],
        ]
    )
    await update.message.reply_text(
        "This erases everything saved here and can't be undone. Are you sure?",
        reply_markup=keyboard,
    )


@allowed_only
async def handle_timezone(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    with user_store(context, update.effective_user.id) as store:
        prefs = store.prefs()
        if not context.args:
            await update.message.reply_text(f"Your time zone is {prefs.timezone}.")
            return
        try:
            ZoneInfo(context.args[0])
        except (ZoneInfoNotFoundError, ValueError):
            await update.message.reply_text("I don't know that zone. Try one like Asia/Kolkata.")
            return
        prefs.timezone = context.args[0]
        store.save_prefs(prefs)
    await update.message.reply_text(f"Time zone set to {context.args[0]}.")


@allowed_only
async def handle_quiet(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    with user_store(context, update.effective_user.id) as store:
        prefs = store.prefs()
        if context.args == ["off"]:
            prefs.quiet_enabled = False
            message = "Quiet hours are off."
        elif len(context.args) == 2:
            try:
                start, end = (time.fromisoformat(arg) for arg in context.args)
            except ValueError:
                await update.message.reply_text("Use 24-hour times, like /quiet 22:00 08:00")
                return
            prefs.quiet_enabled = True
            prefs.quiet_start = start.strftime("%H:%M")
            prefs.quiet_end = end.strftime("%H:%M")
            message = f"Quiet hours are {prefs.quiet_start} to {prefs.quiet_end}."
        else:
            state = f"{prefs.quiet_start} to {prefs.quiet_end}" if prefs.quiet_enabled else "off"
            await update.message.reply_text(
                f"Quiet hours: {state}. Use /quiet 22:00 08:00 or /quiet off."
            )
            return
        store.save_prefs(prefs)
    await update.message.reply_text(message)


async def _set_paused(update: Update, context: ContextTypes.DEFAULT_TYPE, paused: bool) -> None:
    with user_store(context, update.effective_user.id) as store:
        prefs = store.prefs()
        prefs.paused = paused
        store.save_prefs(prefs)
    await update.message.reply_text(
        "Proactive messages paused. Reminders you asked for still arrive."
        if paused
        else "Proactive messages are back on."
    )


@allowed_only
async def handle_pause(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _set_paused(update, context, True)


@allowed_only
async def handle_resume(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _set_paused(update, context, False)


@allowed_only
async def handle_unmute(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    with user_store(context, update.effective_user.id) as store:
        prefs = store.prefs()
        if not context.args or context.args[0] not in prefs.muted_kinds:
            muted = ", ".join(prefs.muted_kinds) or "none"
            await update.message.reply_text(f"Muted reminder types: {muted}.")
            return
        prefs.muted_kinds.remove(context.args[0])
        store.save_prefs(prefs)
    await update.message.reply_text(f"{context.args[0]} reminders are back on.")


@allowed_only
async def handle_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    user_id = update.effective_user.id
    action, *args = query.data.split(":")

    if action == "cancel":
        await query.edit_message_text("Cancelled.")
    elif action == "delete_all":
        async with context.bot_data["locks"][user_id]:
            context.bot_data["service"].delete_all(telegram_identity(user_id))
        log.info("memory_erased", user=user_ref(user_id))
        await query.edit_message_text("Everything has been erased.")
    elif action == "forget":
        table, row_id = args[0], int(args[1])
        async with context.bot_data["locks"][user_id]:
            with user_store(context, user_id) as store:
                forgotten = store.forget(table, row_id)
        log.info("entry_erased", user=user_ref(user_id), table=table, found=forgotten)
        await query.edit_message_text("Erased." if forgotten else "That entry is already gone.")
    elif action == "fpage":
        if "forget" not in context.user_data:
            await query.edit_message_text(
                "That list has expired. Please search again with /forget."
            )
            return
        text, keyboard = forget_page(context, user_id, int(args[0]))
        await query.edit_message_text(text, reply_markup=keyboard)
    elif action == "rem":
        await _handle_reminder_button(query, context, user_id, args)


async def _handle_reminder_button(query, context, user_id: int, args: list[str]) -> None:
    verb, value = args
    with user_store(context, user_id) as store:
        if verb == "done":
            store.mark_done(int(value))
            status = "Done."
        elif verb == "snooze":
            store.snooze(int(value), datetime.now(UTC) + timedelta(days=1))
            status = "Snoozed until tomorrow."
        else:
            store.mute_kind(value)
            status = f"I'll stop sending {value} reminders. /unmute {value} turns them back on."
    await query.edit_message_reply_markup(None)
    await query.message.reply_text(status)


def reminder_keyboard(reminder: dict) -> InlineKeyboardMarkup:
    row = [
        InlineKeyboardButton("Done", callback_data=f"rem:done:{reminder['id']}"),
        InlineKeyboardButton("Snooze 1 day", callback_data=f"rem:snooze:{reminder['id']}"),
    ]
    rows = [row]
    if not reminder["explicit"]:
        label = f"Stop {reminder['kind'].replace('_', ' ')} reminders"
        rows.append([InlineKeyboardButton(label, callback_data=f"rem:mute:{reminder['kind']}")])
    return InlineKeyboardMarkup(rows)


def collect_due(service: RecallService, user_id: int, now: datetime) -> list[dict]:
    with open_store(service, user_id) as store:
        store.sync_embeddings()
        prefs = store.prefs()
        return select_sendable(
            store.due_reminders(now),
            now,
            prefs,
            store.sent_since(start_of_local_day(now, prefs)),
            service.settings.max_reminders_per_day,
        )


async def deliver_due(context: ContextTypes.DEFAULT_TYPE, user_id: int, now: datetime) -> int:
    """Send what's allowed now. The caller holds the user's lock."""
    service: RecallService = context.bot_data["service"]
    due = await asyncio.to_thread(collect_due, service, user_id, now)
    for reminder in due:
        await context.bot.send_message(
            user_id, reminder["text"], reply_markup=reminder_keyboard(reminder)
        )
        with user_store(context, user_id) as store:
            store.mark_sent(reminder["id"])
        log.info(
            "reminder_sent",
            user=user_ref(user_id),
            kind=reminder["kind"],
            explicit=bool(reminder["explicit"]),
        )
    return len(due)


async def reminder_tick(context: ContextTypes.DEFAULT_TYPE) -> None:
    service: RecallService = context.bot_data["service"]
    now = datetime.now(UTC)
    for user_id in service.users():
        lock = context.bot_data["locks"][user_id]
        if lock.locked():
            continue  # One busy user shouldn't delay everyone else. The next tick retries.
        try:
            async with lock:
                await deliver_due(context, user_id, now)
        except Forbidden:
            log.info("reminder_held", reason="bot_blocked", user=user_ref(user_id))
        except TelegramError:
            log.exception("reminder_send_failed", user=user_ref(user_id))
        except Exception:
            # One user's broken memory mustn't stop everyone else's reminders.
            log.exception("reminder_tick_failed", user=user_ref(user_id))


async def nudge_tick(context: ContextTypes.DEFAULT_TYPE) -> None:
    service: RecallService = context.bot_data["service"]
    for user_id in service.users():
        lock = context.bot_data["locks"][user_id]
        if lock.locked():
            continue
        try:
            async with lock:
                await asyncio.to_thread(service.scan_nudges, telegram_identity(user_id))
        except Exception:
            log.exception("nudge_scan_failed", user=user_ref(user_id))


@allowed_only
async def handle_nudge(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    service: RecallService = context.bot_data["service"]
    typing = asyncio.create_task(keep_typing(update.message.chat))
    try:
        async with context.bot_data["locks"][user_id]:
            found = await asyncio.to_thread(service.scan_nudges, telegram_identity(user_id), True)
            sent = await deliver_due(context, user_id, datetime.now(UTC))
    finally:
        typing.cancel()
    if not found:
        await update.message.reply_text("Nothing's gone quiet right now.")
    elif not sent:
        await update.message.reply_text("I found something, and I'll send it when it's allowed.")


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    if isinstance(context.error, BadRequest) and "not modified" in str(context.error):
        return  # A double tap on a button that already did its job.
    log.error("handler_failed", exc_info=context.error)
    if isinstance(update, Update) and update.effective_message:
        try:
            await update.effective_message.reply_text("Something went wrong. Please try again.")
        except TelegramError:
            log.warning("error_reply_failed")


async def on_start(application: Application) -> None:
    await application.bot.set_my_commands(COMMANDS)


def build_application(settings: Settings) -> Application:
    application = (
        Application.builder()
        .token(settings.telegram_token)
        .connect_timeout(TELEGRAM_TIMEOUT_SECONDS)
        .read_timeout(TELEGRAM_TIMEOUT_SECONDS)
        .get_updates_connect_timeout(TELEGRAM_TIMEOUT_SECONDS)
        .concurrent_updates(True)
        .post_init(on_start)
        .build()
    )
    application.bot_data.update(
        settings=settings,
        service=build_service(settings),
        locks=defaultdict(asyncio.Lock),
    )

    private = filters.ChatType.PRIVATE
    for command, handler in {
        "start": handle_help,
        "help": handle_help,
        "nudge": handle_nudge,
        "undo": handle_undo,
        "forget": handle_forget,
        "export": handle_export,
        "delete_all": handle_delete_all,
        "timezone": handle_timezone,
        "quiet": handle_quiet,
        "pause": handle_pause,
        "resume": handle_resume,
        "unmute": handle_unmute,
    }.items():
        application.add_handler(CommandHandler(command, handler, filters=private))
    application.add_error_handler(on_error)
    application.add_handler(CallbackQueryHandler(handle_button))
    application.add_handler(MessageHandler(private & filters.TEXT & ~filters.COMMAND, handle_text))
    application.add_handler(MessageHandler(private & filters.PHOTO, handle_photo))
    application.add_handler(
        MessageHandler(private & ~filters.TEXT & ~filters.PHOTO & ~filters.COMMAND, handle_non_text)
    )
    application.job_queue.run_repeating(reminder_tick, interval=TICK_SECONDS, first=5)
    application.job_queue.run_repeating(nudge_tick, interval=NUDGE_TICK_SECONDS, first=60)
    return application


def main() -> None:
    settings = Settings.from_env()
    setup(settings.data_dir, "bot")
    build_application(settings).run_polling()
