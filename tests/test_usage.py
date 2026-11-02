from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest

from recall.config import Settings, parse_allowed_users
from recall.store import known_user_ids
from recall.usage import Limit, UsageLimiter

NOW = datetime(2026, 11, 1, 12, tzinfo=timezone.utc)


def limiter(tmp_path, per_user=2, total=3):
    return UsageLimiter(tmp_path / "usage.sqlite", per_user, total)


def test_user_limit_stops_one_user_only(tmp_path):
    usage = limiter(tmp_path)
    assert [usage.spend(1, NOW) for _ in range(3)] == [Limit.NONE, Limit.NONE, Limit.USER]
    assert usage.spend(2, NOW) is Limit.NONE


def test_total_limit_stops_everyone(tmp_path):
    usage = limiter(tmp_path, per_user=10, total=2)
    usage.spend(1, NOW)
    usage.spend(2, NOW)
    assert usage.spend(3, NOW) is Limit.TOTAL


def test_refused_messages_do_not_use_up_the_total(tmp_path):
    usage = limiter(tmp_path, per_user=1, total=2)
    usage.spend(1, NOW)
    usage.spend(1, NOW)
    usage.spend(1, NOW)
    assert usage.spend(2, NOW) is Limit.NONE


def test_limits_reset_the_next_day_and_survive_restarts(tmp_path):
    limiter(tmp_path, per_user=1).spend(1, NOW)
    assert limiter(tmp_path, per_user=1).spend(1, NOW) is Limit.USER
    assert limiter(tmp_path, per_user=1).spend(1, NOW + timedelta(days=1)) is Limit.NONE


def test_old_days_are_not_kept(tmp_path):
    usage = limiter(tmp_path)
    usage.spend(1, NOW)
    usage.spend(2, NOW + timedelta(days=1))
    assert usage.conn.execute("SELECT COUNT(*) FROM usage").fetchone()[0] == 1


def test_total_limit_holds_across_separate_connections(tmp_path):
    """Each limiter stands in for a process: the bot, the API and the MCP server."""
    limiters = [limiter(tmp_path, per_user=1000, total=40) for _ in range(4)]

    def spend_many(usage):
        return sum(usage.spend(1, NOW) is Limit.NONE for _ in range(100))

    with ThreadPoolExecutor(len(limiters)) as pool:
        assert sum(pool.map(spend_many, limiters)) == 40


@pytest.mark.parametrize(("value", "expected"), [("*", None), (" * ", None), ("1, 2,", frozenset({1, 2}))])
def test_parse_allowed_users(value, expected):
    assert parse_allowed_users(value) == expected


def test_open_and_closed_settings():
    assert Settings("t", None).allows(999)
    closed = Settings("t", frozenset({1}))
    assert closed.allows(1) and not closed.allows(999)


def test_known_user_ids_ignores_other_files(tmp_path):
    users = tmp_path / "users"
    users.mkdir()
    (users / "111.sqlite").touch()
    (users / "111.sqlite-wal").touch()
    (users / "notes.sqlite").touch()
    assert known_user_ids(tmp_path) == [111]
    assert known_user_ids(tmp_path / "missing") == []
