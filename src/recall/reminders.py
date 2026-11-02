"""Rules for which due reminders may go out right now.

Reminders the user asked for (`explicit`) always fire on time. Proactive ones
respect pause, quiet hours, muted kinds and the daily limit. Held reminders stay
pending, so the next allowed tick sends them.
"""

from datetime import datetime, time
from zoneinfo import ZoneInfo

from recall.store import Prefs


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
