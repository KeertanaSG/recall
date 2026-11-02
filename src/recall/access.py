"""Who can do what: API tokens, scopes, rate limits, the audit log and daily message limits.

Only a hash of each token is stored, so a leaked database reveals no usable token. The audit
log records what was done and by which token, never the text of a memory. Daily limits cap what
an open bot can spend on the model.
"""

import hashlib
import secrets
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path

TOKEN_PREFIX = "rcl_"
READ = "read"
WRITE = "write"
OWNER = "owner"  # Telegram only: tokens can never hold this scope.
AUDIT_DAYS = 30

SCHEMA = """
CREATE TABLE IF NOT EXISTS tokens (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    token_hash TEXT NOT NULL UNIQUE,
    user_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    scopes TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    revoked_at TEXT
);
CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    token_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    action TEXT NOT NULL,
    outcome TEXT NOT NULL,
    at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS audit_by_token ON audit (token_id, at);
CREATE INDEX IF NOT EXISTS audit_by_time ON audit (at);
"""


@dataclass(frozen=True)
class Identity:
    token_id: int | None
    user_id: int
    name: str
    scopes: frozenset[str]
    expires_at: datetime | None = None


def telegram_identity(user_id: int) -> Identity:
    """Telegram has already verified the sender, so the chat needs no token."""
    return Identity(None, user_id, "telegram", frozenset({READ, WRITE, OWNER}))


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class TokenStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = path
        with self._connect() as conn:
            conn.executescript(SCHEMA)
        path.chmod(0o600)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.path, isolation_level=None, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def create(
        self, user_id: int, name: str, scopes: set[str], days: int, now: datetime | None = None
    ) -> str:
        """Issue a token. This is the only time its value is available."""
        unknown = scopes - {READ, WRITE}
        if unknown or not scopes:
            raise ValueError(f"Scopes must be {READ} and/or {WRITE}.")
        now = now or datetime.now(UTC)
        token = TOKEN_PREFIX + secrets.token_urlsafe(32)
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO tokens (token_hash, user_id, name, scopes, created_at, expires_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    _hash(token),
                    user_id,
                    name,
                    ",".join(sorted(scopes)),
                    now.isoformat(timespec="seconds"),
                    (now + timedelta(days=days)).isoformat(timespec="seconds"),
                ),
            )
        return token

    def verify(self, token: str, now: datetime | None = None) -> Identity | None:
        """Return who a token belongs to, or None if it's unknown, expired or revoked."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM tokens WHERE token_hash = ? AND revoked_at IS NULL", (_hash(token),)
            ).fetchone()
        if row is None:
            return None
        expires_at = datetime.fromisoformat(row["expires_at"])
        if (now or datetime.now(UTC)) >= expires_at:
            return None
        return Identity(
            row["id"], row["user_id"], row["name"], frozenset(row["scopes"].split(",")), expires_at
        )

    def revoke(self, token_id: int) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE tokens SET revoked_at = ? WHERE id = ? AND revoked_at IS NULL",
                (datetime.now(UTC).isoformat(timespec="seconds"), token_id),
            )
        return cursor.rowcount == 1

    def list_tokens(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT id, user_id, name, scopes, created_at, expires_at, revoked_at
                   FROM tokens ORDER BY id"""
            ).fetchall()
        return [dict(row) for row in rows]

    def within_rate_limit(
        self, token_id: int, per_minute: int, now: datetime | None = None
    ) -> bool:
        since = (now or datetime.now(UTC)) - timedelta(minutes=1)
        with self._connect() as conn:
            used = conn.execute(
                "SELECT COUNT(*) FROM audit WHERE token_id = ? AND at >= ?",
                (token_id, since.isoformat(timespec="seconds")),
            ).fetchone()[0]
        return used < per_minute

    def record(
        self, identity: Identity, action: str, outcome: str, now: datetime | None = None
    ) -> None:
        now = now or datetime.now(UTC)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO audit (token_id, user_id, action, outcome, at) VALUES (?, ?, ?, ?, ?)",
                (
                    identity.token_id,
                    identity.user_id,
                    action,
                    outcome,
                    now.isoformat(timespec="seconds"),
                ),
            )
            conn.execute(
                "DELETE FROM audit WHERE at < ?",
                ((now - timedelta(days=AUDIT_DAYS)).isoformat(timespec="seconds"),),
            )


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
        # The bot, API and MCP server each have a connection, so the check and the count must
        # share one write transaction, or together they could spend past the limit.
        with self.lock, self.conn:
            self.conn.execute("BEGIN IMMEDIATE")
            return self._spend(user_id, now)

    def _spend(self, user_id: int, now: datetime | None) -> Limit:
        day = (now or datetime.now(UTC)).date().isoformat()
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
