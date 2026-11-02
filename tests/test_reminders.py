from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from recall.core.reminders import is_quiet, select_sendable, start_of_local_day
from recall.storage.store import Prefs

IST = ZoneInfo("Asia/Kolkata")


def at(hour, minute=0):
    return datetime(2026, 11, 1, hour, minute, tzinfo=IST)


def reminder(explicit=0, kind="promise"):
    return {"explicit": explicit, "kind": kind}


@pytest.mark.parametrize(
    ("hour", "quiet"), [(21, False), (22, True), (23, True), (3, True), (7, True), (8, False), (12, False)]
)
def test_overnight_quiet_hours(hour, quiet):
    assert is_quiet(at(hour), Prefs("Asia/Kolkata")) is quiet


def test_daytime_and_disabled_quiet_hours():
    assert is_quiet(at(14), Prefs("Asia/Kolkata", quiet_start="13:00", quiet_end="15:00"))
    assert not is_quiet(at(16), Prefs("Asia/Kolkata", quiet_start="13:00", quiet_end="15:00"))
    assert not is_quiet(at(23), Prefs("Asia/Kolkata", quiet_enabled=False))


def test_quiet_hours_use_the_users_time_zone():
    # 18:00 UTC is 23:30 in India, so this is quiet there.
    now = datetime(2026, 11, 1, 18, 0, tzinfo=timezone.utc)
    assert select_sendable([reminder()], now, Prefs("Asia/Kolkata"), 0, 5) == []
    assert select_sendable([reminder()], now, Prefs("UTC"), 0, 5) == [reminder()]


def test_explicit_reminders_ignore_quiet_pause_and_limit():
    prefs = Prefs("Asia/Kolkata", paused=True)
    assert select_sendable([reminder(explicit=1)], at(23), prefs, 99, 5) == [reminder(explicit=1)]


def test_proactive_respects_pause_mute_and_limit():
    assert select_sendable([reminder()], at(12), Prefs("Asia/Kolkata", paused=True), 0, 5) == []
    assert select_sendable([reminder()], at(12), Prefs("Asia/Kolkata", muted_kinds=["promise"]), 0, 5) == []
    assert len(select_sendable([reminder()] * 4, at(12), Prefs("Asia/Kolkata"), 3, 5)) == 2
    assert select_sendable([reminder()], at(12), Prefs("Asia/Kolkata"), 7, 5) == []


def test_daily_limit_resets_at_local_midnight():
    now = datetime(2026, 11, 1, 20, 0, tzinfo=timezone.utc)  # 01:30 on Nov 2 in India
    start = start_of_local_day(now, Prefs("Asia/Kolkata"))
    assert start.isoformat() == "2026-11-02T00:00:00+05:30"
