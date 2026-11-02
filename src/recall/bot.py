"""Telegram interface: commands, buttons and the reminder tick."""

import asyncio
import io
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable
from datetime import datetime, time, timedelta, timezone
from functools import wraps
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from telegram import BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.error import Forbidden, TelegramError
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from recall.agent import Agent, TurnFailed, summarize_row
from recall.config import Settings
from recall.embeddings import LocalEmbedder
from recall.service import build_agent
from recall.reminders import select_sendable, start_of_local_day
from recall.store import Store, ToolError, delete_user_data, known_user_ids, user_db_path
from recall.usage import Limit, UsageLimiter

log = logging.getLogger(__name__)

TICK_SECONDS = 30
TELEGRAM_TIMEOUT_SECONDS = 20
TYPING_SECONDS = 4
TELEGRAM_LIMIT = 4000
WELCOME = """\
Text me anything and I'll remember it.

Try:
- Watched Dune 2 with Arjun, 9/10
- Cab to airport 450
- Told Priya I'd send the file by Friday
- How much did I spend on cabs this month?

This is an early test project. Your memories are kept separate from other users', but they \
aren't encrypted on the server, so the person running this bot could read them. Messages also \
pass through Telegram and Anthropic. Please don't send passwords, bank details or medical records.

Recall isn't a mental health service or for emergencies. If you're struggling, please reach out \
to someone you trust or a local crisis line.

/undo reverses the last save
/forget <words> erases one entry
/export downloads everything
/delete_all erases everything
/timezone <Area/City> sets your time zone
/quiet 22:00 08:00 or /quiet off sets quiet hours
/pause and /resume control proactive messages
/unmute <kind> turns a reminder type back on"""

COMMANDS = [
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
            log.info("Ignored a user who isn't on the allowlist")
            return
        await handler(update, context)

    return wrapper


def open_store(context: ContextTypes.DEFAULT_TYPE, user_id: int) -> Store:
    settings: Settings = context.bot_data["settings"]
    return Store(
        user_db_path(settings.data_dir, user_id),
        context.bot_data["embedder"],
        settings.default_timezone,
    )


def split_message(text: str) -> list[str]:
    return [text[i : i + TELEGRAM_LIMIT] for i in range(0, len(text), TELEGRAM_LIMIT)] or [""]


async def keep_typing(chat) -> None:
    """Telegram clears the typing indicator after about five seconds, so keep renewing it."""
    while True:
        try:
            await chat.send_action(ChatAction.TYPING)
        except TelegramError:
            log.warning("Couldn't send the typing indicator")
        await asyncio.sleep(TYPING_SECONDS)


@allowed_only
async def handle_help(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(WELCOME)


@allowed_only
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    text = update.message.text
    agent: Agent = context.bot_data["agent"]

    limit = context.bot_data["limiter"].spend(user_id)
    if limit is not Limit.NONE:
        await update.message.reply_text(
            "You've reached today's message limit. Please try again tomorrow."
            if limit is Limit.USER
            else "Recall has reached its limit for today. Please try again tomorrow."
        )
        return

    def respond():
        with open_store(context, user_id) as store:
            return agent.respond(store, text)

    typing = asyncio.create_task(keep_typing(update.message.chat))
    try:
        async with context.bot_data["locks"][user_id]:
            reply = await asyncio.to_thread(respond)
    except TurnFailed:
        log.exception("Turn failed")
        await update.message.reply_text("Something went wrong, so nothing was saved. Please resend that.")
        return
    finally:
        typing.cancel()

    for part in split_message(reply.render()):
        await update.message.reply_text(part)


@allowed_only
async def handle_non_text(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("I can only read text messages for now.")


@allowed_only
async def handle_undo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id

    def undo():
        with open_store(context, user_id) as store:
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
        with open_store(context, update.effective_user.id) as store:
            return store.search(words, limit=5)

    try:
        matches = await asyncio.to_thread(find)
    except ToolError:
        await update.message.reply_text("Search isn't available right now. Please try again shortly.")
        return
    if not matches:
        await update.message.reply_text("I couldn't find anything like that.")
        return

    buttons = []
    with open_store(context, update.effective_user.id) as store:
        for match in matches:
            row = store.get_row(match["table"], match["row"]["id"])
            label = f"{match['table'].lstrip('_')}: {summarize_row(row)}"[:60]
            data = f"forget:{match['table']}:{row['id']}"
            buttons.append([InlineKeyboardButton(label, callback_data=data)])
    buttons.append([InlineKeyboardButton("Cancel", callback_data="cancel")])
    await update.message.reply_text(
        "Which one should I erase for good? This also removes its history and can't be undone.",
        reply_markup=InlineKeyboardMarkup(buttons),
    )


@allowed_only
async def handle_export(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    with open_store(context, update.effective_user.id) as store:
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
    with open_store(context, update.effective_user.id) as store:
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
    with open_store(context, update.effective_user.id) as store:
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
            await update.message.reply_text(f"Quiet hours: {state}. Use /quiet 22:00 08:00 or /quiet off.")
            return
        store.save_prefs(prefs)
    await update.message.reply_text(message)


async def _set_paused(update: Update, context: ContextTypes.DEFAULT_TYPE, paused: bool) -> None:
    with open_store(context, update.effective_user.id) as store:
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
    with open_store(context, update.effective_user.id) as store:
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
    settings: Settings = context.bot_data["settings"]
    action, *args = query.data.split(":")

    if action == "cancel":
        await query.edit_message_text("Cancelled.")
    elif action == "delete_all":
        async with context.bot_data["locks"][user_id]:
            delete_user_data(settings.data_dir, user_id)
        await query.edit_message_text("Everything has been erased.")
    elif action == "forget":
        table, row_id = args[0], int(args[1])
        async with context.bot_data["locks"][user_id]:
            with open_store(context, user_id) as store:
                forgotten = store.forget(table, row_id)
        await query.edit_message_text("Erased." if forgotten else "That entry is already gone.")
    elif action == "rem":
        await _handle_reminder_button(query, context, user_id, args)


async def _handle_reminder_button(query, context, user_id: int, args: list[str]) -> None:
    verb, value = args
    with open_store(context, user_id) as store:
        if verb == "done":
            store.mark_done(int(value))
            status = "Done."
        elif verb == "snooze":
            store.snooze(int(value), datetime.now(timezone.utc) + timedelta(days=1))
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


def collect_due(settings: Settings, embedder, user_id: int, now: datetime) -> list[dict]:
    with Store(user_db_path(settings.data_dir, user_id), embedder, settings.default_timezone) as store:
        store.sync_embeddings()
        prefs = store.prefs()
        return select_sendable(
            store.due_reminders(now),
            now,
            prefs,
            store.sent_since(start_of_local_day(now, prefs)),
            settings.max_reminders_per_day,
        )


async def reminder_tick(context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.bot_data["settings"]
    embedder = context.bot_data["embedder"]
    now = datetime.now(timezone.utc)
    for user_id in known_user_ids(settings.data_dir):
        if not settings.allows(user_id):
            continue
        try:
            async with context.bot_data["locks"][user_id]:
                due = await asyncio.to_thread(collect_due, settings, embedder, user_id, now)
                for reminder in due:
                    await context.bot.send_message(
                        user_id, reminder["text"], reply_markup=reminder_keyboard(reminder)
                    )
                    with open_store(context, user_id) as store:
                        store.mark_sent(reminder["id"])
        except Forbidden:
            log.info("A user has blocked the bot, so their reminder stays pending")
        except TelegramError:
            log.exception("Couldn't send a reminder. It stays pending and will retry.")


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
    embedder = LocalEmbedder(settings.data_dir / "models")
    application.bot_data.update(
        settings=settings,
        embedder=embedder,
        agent=build_agent(settings),
        limiter=UsageLimiter(
            settings.data_dir / "usage.sqlite",
            settings.max_messages_per_user_per_day,
            settings.max_messages_per_day,
        ),
        locks=defaultdict(asyncio.Lock),
    )

    private = filters.ChatType.PRIVATE
    for command, handler in {
        "start": handle_help,
        "help": handle_help,
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
    application.add_handler(CallbackQueryHandler(handle_button))
    application.add_handler(MessageHandler(private & filters.TEXT & ~filters.COMMAND, handle_text))
    application.add_handler(MessageHandler(private & ~filters.TEXT & ~filters.COMMAND, handle_non_text))
    application.job_queue.run_repeating(reminder_tick, interval=TICK_SECONDS, first=5)
    return application
