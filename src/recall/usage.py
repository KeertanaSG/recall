"""Daily message limits, which cap what an open bot can spend on the Anthropic API."""

import sqlite3
import threading
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path


class Limit(Enum):
    NONE = "none"
    USER = "user"
    TOTAL = "total"


class UsageLimiter:
    """Counts persist across restarts, and only today's are kept."""

    def __init__(self, path: Path, per_user: int, total: int):
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.per_user = per_user
        self.total = total
        self.lock = threading.Lock()
        self.conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False, timeout=10)
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS usage (
                day TEXT NOT NULL, user_id INTEGER NOT NULL, messages INTEGER NOT NULL,
                PRIMARY KEY (day, user_id))"""
        )

    def spend(self, user_id: int, now: datetime | None = None) -> Limit:
        """Record one message, unless a limit has been reached."""
        with self.lock:
            return self._spend(user_id, now)

    def _spend(self, user_id: int, now: datetime | None) -> Limit:
        day = (now or datetime.now(timezone.utc)).date().isoformat()
        self.conn.execute("DELETE FROM usage WHERE day < ?", (day,))
        total = self.conn.execute(
            "SELECT COALESCE(SUM(messages), 0) FROM usage WHERE day = ?", (day,)
        ).fetchone()[0]
        if total >= self.total:
            return Limit.TOTAL
        row = self.conn.execute(
            "SELECT messages FROM usage WHERE day = ? AND user_id = ?", (day, user_id)
        ).fetchone()
        if row and row[0] >= self.per_user:
            return Limit.USER
        self.conn.execute(
            """INSERT INTO usage (day, user_id, messages) VALUES (?, ?, 1)
               ON CONFLICT (day, user_id) DO UPDATE SET messages = messages + 1""",
            (day, user_id),
        )
        return Limit.NONE
