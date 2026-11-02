"""What Recall needs from storage. A new backend implements these two protocols."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

REMINDERS = "_reminders"
ROW_METADATA = ("id", "created_at", "updated_at", "source_text")
COLUMN_TYPES = {
    "text": "TEXT",
    "integer": "INTEGER",
    "real": "REAL",
    "boolean": "INTEGER",
    "date": "TEXT",
    "datetime": "TEXT",
}


class ToolError(Exception):
    """A problem the agent can read and fix on its next step."""


@dataclass
class Prefs:
    timezone: str
    quiet_enabled: bool = True
    quiet_start: str = "22:00"
    quiet_end: str = "08:00"
    paused: bool = False
    muted_kinds: list[str] = field(default_factory=list)
    last_nudge_scan: str | None = None


class MemoryStore(Protocol):
    """One user's memory. Writes made during a turn are tracked under turn_id for undo."""

    turn_id: int | None

    def __enter__(self) -> "MemoryStore": ...

    def __exit__(self, *_exc) -> None: ...

    # Tables and rows
    def tables(self) -> list[dict]: ...

    def create_table(
        self, name: str, description: str, columns: list[dict], confirmed_distinct: bool = False
    ) -> None: ...

    def add_column(self, table: str, name: str, column_type: str, description: str = "") -> None: ...

    def boolean_columns(self, table: str) -> frozenset[str]: ...

    def add_row(self, table: str, values: dict[str, Any], source_text: str = "") -> int: ...

    def update_row(self, table: str, row_id: int, changes: dict[str, Any], reason: str) -> None: ...

    def get_row(self, table: str, row_id: int) -> dict | None: ...

    def row_history(self, table: str, row_id: int) -> list[dict]: ...

    def undo(self, turn_id: int | None = None) -> bool: ...

    def forget(self, table: str, row_id: int) -> bool: ...

    # Finding things
    def read_query(self, sql: str) -> dict: ...

    def search(self, text: str, limit: int = 8, table: str | None = None) -> list[dict]: ...

    def keyword_search(
        self, terms: list[str], limit: int = 15, table: str | None = None, whole_word: bool = False
    ) -> list[dict]: ...

    def mentions(self, name: str, limit: int = 40) -> list[dict]: ...

    def rows_not_marked(self, limit: int) -> list[list[dict]]: ...

    def sync_embeddings(self) -> None: ...

    # Reminders
    def add_reminder(
        self,
        text: str,
        remind_at: datetime,
        kind: str,
        explicit: bool,
        linked_table: str | None = None,
        linked_id: int | None = None,
        source_text: str = "",
    ) -> int: ...

    def cancel_reminder(self, reminder_id: int, reason: str) -> None: ...

    def pending_reminders(self) -> list[dict]: ...

    def linked_rows(self) -> set[tuple[str, int]]: ...

    def due_reminders(self, now: datetime) -> list[dict]: ...

    def sent_since(self, moment: datetime) -> int: ...

    def mark_sent(self, reminder_id: int) -> None: ...

    def mark_done(self, reminder_id: int) -> None: ...

    def snooze(self, reminder_id: int, until: datetime) -> None: ...

    def mute_kind(self, kind: str) -> None: ...

    # Settings and conversation
    def prefs(self) -> Prefs: ...

    def save_prefs(self, prefs: Prefs) -> None: ...

    def recent_messages(self) -> list[dict]: ...

    def log_exchange(self, user_text: str, reply: str) -> None: ...

    def export_zip(self) -> bytes: ...


class StoreProvider(Protocol):
    """Where user memories live. Choosing a backend only means replacing this."""

    def open(self, user_id: int) -> MemoryStore: ...

    def user_ids(self) -> list[int]: ...

    def delete(self, user_id: int) -> None: ...
