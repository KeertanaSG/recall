"""Smart nudges: Recall speaks up about unfinished business without being asked.

detect -> judge -> policy -> deliver

Detection is free: it uses the schema and local search, and works on any table. Only its few
candidates reach the model, which leaves out anything finished or not worth a message. Picks
become ordinary reminders, so quiet hours, daily limits, mutes and buttons all apply unchanged.

Reminders the user asked for (`explicit`) always fire on time. Proactive ones respect pause, quiet
hours, muted kinds and the daily limit. Held reminders stay pending, so the next allowed tick sends
them.
"""

import json
import logging
from datetime import UTC, datetime, time, timedelta
from zoneinfo import ZoneInfo

from recall.core.agent import summarize_row
from recall.core.retrieval import fuse, hybrid_search, result_key
from recall.llm.base import LLM, LLMError
from recall.prompts import render
from recall.storage.base import ROW_METADATA, MemoryStore, Prefs, ToolError

log = logging.getLogger(__name__)

NUDGE = "nudge"
SCAN_EVERY = timedelta(days=1)
MIN_AGE = timedelta(days=3)  # Anything newer is still fresh in the user's mind.
MAX_AGE = timedelta(days=120)  # Anything older has probably stopped mattering.
POOL = 30
MAX_CANDIDATES = 12
MAX_NUDGES = 2
MAX_LATER_ENTRIES = 3

# Kinds of unfinished business, not subjects, so these fit whatever tables a user has.
OPEN_LOOPS = [
    "someone owes me money",
    "I owe someone money",
    "I lent something to someone",
    "I borrowed something and need to return it",
    "I promised someone I would do something",
    "I need to follow up or get back to someone",
    "I'm waiting to hear back",
    "something I still need to do",
]

JUDGE_SYSTEM = render("nudge_judge_system")

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "nudges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "message": {"type": "string"},
                    "importance": {"type": "integer"},
                },
                "required": ["key", "message", "importance"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["nudges"],
    "additionalProperties": False,
}


def is_quiet(local: datetime, prefs: Prefs) -> bool:
    if not prefs.quiet_enabled:
        return False
    start = time.fromisoformat(prefs.quiet_start)
    end = time.fromisoformat(prefs.quiet_end)
    now = local.time()
    if start == end:
        return False
    if start < end:
        return start <= now < end
    return now >= start or now < end


def start_of_local_day(now: datetime, prefs: Prefs) -> datetime:
    local = now.astimezone(ZoneInfo(prefs.timezone))
    return local.replace(hour=0, minute=0, second=0, microsecond=0)


def select_sendable(
    due: list[dict], now: datetime, prefs: Prefs, sent_today: int, daily_limit: int
) -> list[dict]:
    local = now.astimezone(ZoneInfo(prefs.timezone))
    quiet = is_quiet(local, prefs)
    allowance = max(daily_limit - sent_today, 0)
    sendable = []
    for reminder in due:
        if reminder["explicit"]:
            sendable.append(reminder)
        elif (
            not prefs.paused
            and not quiet
            and reminder["kind"] not in prefs.muted_kinds
            and allowance > 0
        ):
            sendable.append(reminder)
            allowance -= 1
    return sendable


def _saved_at(result: dict) -> datetime:
    return datetime.fromisoformat(result["row"]["created_at"])


def find_candidates(store: MemoryStore, now: datetime) -> list[dict]:
    """Rows that look unfinished, gone quiet, and not already followed up. Uses no model."""
    rankings = store.rows_not_marked(POOL)
    try:
        rankings.extend(store.search(phrase, POOL) for phrase in OPEN_LOOPS)
    except ToolError:
        log.warning("Semantic search is unavailable, so nudges use yes/no columns only")
    handled = store.linked_rows()

    def eligible(result: dict) -> bool:
        return (
            not result["table"].startswith("_")
            and (result["table"], result["row"]["id"]) not in handled
            and MIN_AGE <= now - _saved_at(result) <= MAX_AGE
        )

    return fuse([[r for r in ranking if eligible(r)] for ranking in rankings], MAX_CANDIDATES)


def later_entries(store: MemoryStore, candidate: dict) -> list[str]:
    """Newer entries on the same thing, which can show it's already resolved."""
    details = " ".join(
        str(value) for name, value in candidate["row"].items() if name not in ROW_METADATA
    )
    matches = [
        match
        for match in hybrid_search(store, details, MAX_CANDIDATES)
        if not match["table"].startswith("_") and _saved_at(match) > _saved_at(candidate)
    ]
    return [summarize_row(match["row"]) for match in matches[:MAX_LATER_ENTRIES]]


class Nudger:
    def __init__(self, llm: LLM, model: str, max_nudges: int = MAX_NUDGES):
        self.llm = llm
        self.model = model
        self.max_nudges = max_nudges

    def judge(self, store: MemoryStore, candidates: list[dict], now: datetime) -> list[dict] | None:
        """Return the worthwhile candidates, most important first, or None if the model failed."""
        if not candidates:
            return []
        by_key = {result_key(c): c for c in candidates}
        entries = {
            key: {
                "entry": c["row"],
                "days_ago": (now - _saved_at(c)).days,
                "later_entries": later_entries(store, c),
            }
            for key, c in by_key.items()
        }
        prompt = render(
            "nudge_judge",
            today=now.date().isoformat(),
            entries=json.dumps(entries, ensure_ascii=False, default=str),
        )
        try:
            verdict = self.llm.ask_json(self.model, JUDGE_SYSTEM, prompt, JUDGE_SCHEMA)
        except LLMError:
            log.exception("The nudge judge failed")
            return None
        # Keys the model made up are dropped, so a nudge always points at a real entry.
        picks = [pick for pick in verdict["nudges"] if pick["key"] in by_key]
        picks.sort(key=lambda pick: pick["importance"], reverse=True)
        return [{**pick, "result": by_key[pick["key"]]} for pick in picks[: self.max_nudges]]

    def scan(
        self, store: MemoryStore, now: datetime | None = None, force: bool = False
    ) -> list[int]:
        """Queue any nudges as due reminders, and return their IDs."""
        now = now or datetime.now(UTC)
        prefs = store.prefs()
        if not force and not self._is_due(prefs, now):
            return []
        picks = self.judge(store, find_candidates(store, now), now)
        if picks is None:
            return []  # Left unstamped, so the next tick tries again.
        ids = []
        for pick in picks:
            table, row = pick["result"]["table"], pick["result"]["row"]
            source = summarize_row(row, store.boolean_columns(table))
            text = f"{pick['message']}\n\nFrom your notes\n{source}"
            ids.append(store.add_reminder(text, now, NUDGE, False, table, row["id"]))
        prefs.last_nudge_scan = now.isoformat(timespec="seconds")
        store.save_prefs(prefs)
        return ids

    @staticmethod
    def _is_due(prefs: Prefs, now: datetime) -> bool:
        if prefs.paused or NUDGE in prefs.muted_kinds:
            return False
        last = prefs.last_nudge_scan
        return last is None or now - datetime.fromisoformat(last) >= SCAN_EVERY
