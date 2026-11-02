"""The agent's tools. There is deliberately no delete tool: deleting always takes a user tap."""

import json
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from recall.retrieval import Retriever
from recall.schema_design import TableDesigner
from recall.store import COLUMN_TYPES, Store, ToolError

REMINDER_KINDS = ["deadline", "promise", "follow_up", "other"]
ADD_ONLY_TOOLS = frozenset({"create_table", "add_column", "add_row", "set_reminder"})

TOOLS: list[dict] = [
    {
        "name": "list_tables",
        "description": (
            "List the user's tables with columns, row counts and the two latest rows. "
            "Check the latest rows to match how existing values are written."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "create_table",
        "description": (
            "Create a table for a new subject. Only do this when no existing table fits. A "
            "designer plans the schema, and this returns the table's name and columns."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "subject": {"type": "string", "description": "The broad subject, such as movies or loans."},
                "example": {"type": "string", "description": "The user's message that needs this table."},
                "confirmed_distinct": {
                    "type": "boolean",
                    "description": "Set true only after a similar-table warning, if the subject truly differs.",
                },
            },
            "required": ["subject", "example"],
        },
    },
    {
        "name": "add_column",
        "description": (
            "Add an optional column to an existing table. Use this whenever a new detail "
            "appears, instead of creating a table or burying the detail in a text column."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "table": {"type": "string"},
                "name": {"type": "string"},
                "type": {"type": "string", "enum": sorted(COLUMN_TYPES)},
                "description": {"type": "string"},
            },
            "required": ["table", "name", "type"],
        },
    },
    {
        "name": "add_row",
        "description": "Save one entry. Dates are YYYY-MM-DD. Returns the new row's id.",
        "input_schema": {
            "type": "object",
            "properties": {
                "table": {"type": "string"},
                "values": {"type": "object", "description": "Column names mapped to values."},
            },
            "required": ["table", "values"],
        },
    },
    {
        "name": "update_row",
        "description": "Change a saved entry. The old values are kept as history.",
        "input_schema": {
            "type": "object",
            "properties": {
                "table": {"type": "string"},
                "id": {"type": "integer"},
                "changes": {"type": "object", "description": "Column names mapped to new values."},
                "reason": {"type": "string", "description": "Why it changed, for the history."},
            },
            "required": ["table", "id", "changes", "reason"],
        },
    },
    {
        "name": "row_history",
        "description": "Show how a saved entry has changed over time, for questions about past values.",
        "input_schema": {
            "type": "object",
            "properties": {"table": {"type": "string"}, "id": {"type": "integer"}},
            "required": ["table", "id"],
        },
    },
    {
        "name": "query",
        "description": (
            "Run one read-only SQLite SELECT on the user's tables and _reminders. Use this for "
            "totals, counts, filters and dates. Do arithmetic in SQL, never yourself. For a total, "
            "also select group_concat(id) so the answer can cite its rows. Returns up to 200 rows."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"sql": {"type": "string"}},
            "required": ["sql"],
        },
    },
    {
        "name": "search",
        "description": (
            "Find entries for vague questions. This expands the question, searches by meaning and "
            "by keyword, and filters for relevance. If nothing_relevant is true, tell the user nothing "
            "is saved, without searching again. Use the whole question as text."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "table": {"type": "string", "description": "Optional: search one table only."},
                "limit": {"type": "integer", "description": "1 to 20. Defaults to 8."},
            },
            "required": ["text"],
        },
    },
    {
        "name": "set_reminder",
        "description": (
            "Schedule a reminder. remind_at is local time as YYYY-MM-DDTHH:MM. Set explicit=true "
            "only if the user asked to be reminded. Link the saved entry when there is one."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": (
                        "The message to send, as a friend would say it, like "
                        "\"You told Maya you'd send the photos by Monday.\" No 'Reminder:' prefix."
                    ),
                },
                "remind_at": {"type": "string"},
                "kind": {"type": "string", "enum": REMINDER_KINDS},
                "explicit": {"type": "boolean"},
                "linked_table": {"type": "string"},
                "linked_id": {"type": "integer"},
            },
            "required": ["text", "remind_at", "kind", "explicit"],
        },
    },
    {
        "name": "cancel_reminder",
        "description": "Cancel a reminder. To reschedule, cancel it and set a new one.",
        "input_schema": {
            "type": "object",
            "properties": {"id": {"type": "integer"}, "reason": {"type": "string"}},
            "required": ["id", "reason"],
        },
    },
]


class Tools:
    def __init__(
        self,
        store: Store,
        source_text: str,
        now: datetime,
        retriever: Retriever | None,
        designer: TableDesigner | None,
        allowed: frozenset[str] | None = None,
    ):
        self.store = store
        self.source_text = source_text
        self.now = now
        self.retriever = retriever
        self.designer = designer
        self.schemas = [tool for tool in TOOLS if allowed is None or tool["name"] in allowed]
        self.last_saved: tuple[str, int] | None = None
        self.saved: list[str] = []

    def run(self, name: str, tool_input: dict[str, Any]) -> str:
        handler = getattr(self, f"_{name}", None)
        if handler is None or not any(tool["name"] == name for tool in self.schemas):
            raise ToolError(f"Unknown tool {name!r}.")
        try:
            return json.dumps(handler(**tool_input), ensure_ascii=False, default=str)
        except TypeError as exc:
            raise ToolError(f"Bad arguments for {name}: {exc}") from exc

    def _list_tables(self) -> list[dict]:
        return self.store.tables()

    def _create_table(self, subject: str, example: str, confirmed_distinct: bool = False) -> dict:
        return self.designer.create(self.store, subject, example, confirmed_distinct)

    def _add_column(self, table: str, name: str, type: str, description: str = "") -> dict:
        self.store.add_column(table, name, type, description)
        return {"added": name}

    def _add_row(self, table: str, values: dict) -> dict:
        row_id = self.store.add_row(table, values, self.source_text)
        self.last_saved = (table, row_id)
        self.saved.append(f"{table}#{row_id}")
        return {"table": table, "id": row_id}

    def _update_row(self, table: str, id: int, changes: dict, reason: str) -> dict:
        self.store.update_row(table, id, changes, reason)
        return {"updated": f"{table}#{id}"}

    def _row_history(self, table: str, id: int) -> list[dict]:
        return self.store.row_history(table, id)

    def _query(self, sql: str) -> dict:
        return self.store.read_query(sql)

    def _search(self, text: str, table: str | None = None, limit: int = 8) -> dict:
        return self.retriever.search(self.store, text, table, min(max(limit, 1), 20))

    def _set_reminder(
        self,
        text: str,
        remind_at: str,
        kind: str,
        explicit: bool,
        linked_table: str | None = None,
        linked_id: int | None = None,
    ) -> dict:
        if kind not in REMINDER_KINDS:
            raise ToolError(f"kind must be one of {REMINDER_KINDS}.")
        prefs = self.store.prefs()
        if not explicit and kind in prefs.muted_kinds:
            raise ToolError(f"The user turned off {kind} reminders.")
        try:
            moment = datetime.fromisoformat(remind_at)
        except ValueError as exc:
            raise ToolError("remind_at must look like 2026-11-06T09:00.") from exc
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=ZoneInfo(prefs.timezone))
        if moment < self.now - timedelta(minutes=1):
            raise ToolError(f"{remind_at} is in the past. Pick a future time.")
        if linked_table is None and self.last_saved:
            # Keeps /forget able to remove the reminder along with its entry.
            linked_table, linked_id = self.last_saved
        reminder_id = self.store.add_reminder(
            text, moment, kind, explicit, linked_table, linked_id, self.source_text
        )
        self.saved.append(f"_reminders#{reminder_id}")
        return {"table": "_reminders", "id": reminder_id, "remind_at": remind_at}

    def _cancel_reminder(self, id: int, reason: str) -> dict:
        self.store.cancel_reminder(id, reason)
        return {"cancelled": id}
