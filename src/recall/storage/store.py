"""Per-user storage: agent-created tables, history, undo, and embeddings.

Each user gets one SQLite file. Internal tables start with an underscore, and
agent-created tables can't. Every write records a `_history` entry under the
current turn, which is what makes corrections and undo work.
"""

import csv
import io
import json
import logging
import math
import os
import re
import sqlite3
import threading
import time
import zipfile
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import sqlite_vec
from fastembed import TextEmbedding

from recall.config import Settings
from recall.storage.base import (
    COLUMN_TYPES,
    EMBEDDING_DIM,
    REMINDERS,
    ROW_METADATA,
    Embedder,
    EmbeddingError,
    InputType,
    Prefs,
    ToolError,
)

log = logging.getLogger(__name__)

RESERVED_NAMES = {*ROW_METADATA, "rowid", "oid"}
IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
GENERIC_NAME_TOKENS = {"log", "list", "tracker", "entry", "record", "history", "data", "my"}
SIMILAR_DESCRIPTION = 0.75  # Uncalibrated. Tune with the schema evals.
SIMILAR_COLUMNS = 0.6
QUERY_ROW_LIMIT = 200
QUERY_SECONDS = 2.0
MESSAGE_WINDOW = 12

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS _tables (
    name TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS _columns (
    table_name TEXT NOT NULL,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    required INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (table_name, name)
);
CREATE TABLE IF NOT EXISTS _turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS _history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    turn_id INTEGER NOT NULL,
    table_name TEXT NOT NULL,
    row_id INTEGER NOT NULL,
    op TEXT NOT NULL,
    old_values TEXT,
    new_values TEXT,
    reason TEXT NOT NULL DEFAULT '',
    changed_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS _embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    table_name TEXT NOT NULL,
    row_id INTEGER NOT NULL,
    embedded INTEGER NOT NULL DEFAULT 0,
    UNIQUE (table_name, row_id)
);
CREATE TABLE IF NOT EXISTS _messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS _settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS {REMINDERS} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    remind_at TEXT NOT NULL,
    kind TEXT NOT NULL,
    explicit INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending',
    sent_at TEXT,
    linked_table TEXT,
    linked_id INTEGER,
    source_text TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS _vectors
    USING vec0(embedding float[{EMBEDDING_DIM}] distance_metric=cosine);
"""


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def validate_identifier(name: str, kind: str) -> str:
    if not IDENTIFIER.match(name) or name.startswith("sqlite") or name in RESERVED_NAMES:
        raise ToolError(
            f"Invalid {kind} name {name!r}. Use lowercase letters, digits and underscores, "
            f"starting with a letter, and avoid {sorted(RESERVED_NAMES)}."
        )
    return name


def coerce_value(column: str, column_type: str, value: Any) -> Any:
    """Convert a value to its column type, so SQL totals can be trusted."""
    if value is None:
        return None
    try:
        if column_type == "text":
            return str(value).strip()
        if column_type == "boolean":
            if value in (True, False, 0, 1):
                return int(value)
            raise ValueError
        if column_type in ("integer", "real"):
            if isinstance(value, bool):
                raise ValueError
            number = float(value)
            if not math.isfinite(number):
                raise ValueError
            if column_type == "real":
                return number
            if not number.is_integer():
                raise ValueError
            return int(number)
        if column_type == "date":
            return date.fromisoformat(str(value)[:10]).isoformat()
        if column_type == "datetime":
            return datetime.fromisoformat(str(value)).isoformat()
    except ValueError:
        pass
    raise ToolError(f"{value!r} isn't a valid {column_type} for column {column!r}.")


def stem(word: str) -> str:
    """Movie and movies, and city and cities, each reduce to the same stem."""
    if word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    return word.removesuffix("ie").removesuffix("y")


def name_key(name: str) -> frozenset[str]:
    """Reduce a table name to its meaning, so coffee and coffee_logs compare equal."""
    tokens = {stem(token) for token in name.split("_")}
    return frozenset(tokens - {stem(token) for token in GENERIC_NAME_TOKENS} or tokens)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def _read_only_authorizer(action, arg1, arg2, _db, _trigger):
    if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_RECURSIVE):
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_READ:
        internal = arg1.startswith(("_", "sqlite_")) and arg1 != REMINDERS
        return sqlite3.SQLITE_DENY if internal else sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_FUNCTION:
        return sqlite3.SQLITE_DENY if arg2 == "load_extension" else sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


class LocalEmbedder:
    """Runs on this machine, so no key is needed and text never leaves it."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.lock = threading.Lock()
        self.loaded: TextEmbedding | None = None

    def _model(self) -> TextEmbedding:
        """Load on first use, so a process that never searches saves the memory."""
        with self.lock:
            if self.loaded is None:
                self.loaded = TextEmbedding(EMBEDDING_MODEL, cache_dir=str(self.cache_dir))
            return self.loaded

    def embed(self, texts: list[str], input_type: InputType) -> list[list[float]]:
        model = self._model()
        embed = model.query_embed if input_type == "query" else model.passage_embed
        try:
            return [vector.tolist() for vector in embed(texts)]
        except Exception as exc:
            raise EmbeddingError(str(exc)) from exc


class Store:
    def __init__(self, path: Path, embedder: Embedder, default_timezone: str):
        self.path = path
        self.embedder = embedder
        self.default_timezone = default_timezone
        self.turn_id: int | None = None
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.conn = sqlite3.connect(path, isolation_level=None, timeout=10)
        os.chmod(path, 0o600)
        self.conn.row_factory = sqlite3.Row
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA secure_delete=ON")
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    @contextmanager
    def _transaction(self):
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        self.conn.execute("COMMIT")

    # Schema

    def tables(self) -> list[dict]:
        tables = []
        for row in self.conn.execute(
            "SELECT name, description FROM _tables ORDER BY name"
        ).fetchall():
            columns = self.conn.execute(
                """SELECT name, type, description, required FROM _columns
                   WHERE table_name = ? ORDER BY rowid""",
                (row["name"],),
            ).fetchall()
            table = quote_identifier(row["name"])
            tables.append(
                {
                    "name": row["name"],
                    "description": row["description"],
                    "columns": [dict(column) for column in columns],
                    "row_count": self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0],
                    "latest_rows": [
                        self._public_row(latest)
                        for latest in self.conn.execute(
                            f"SELECT * FROM {table} ORDER BY id DESC LIMIT 2"
                        ).fetchall()
                    ],
                }
            )
        return tables

    def create_table(
        self,
        name: str,
        description: str,
        columns: list[dict],
        confirmed_distinct: bool = False,
    ) -> None:
        validate_identifier(name, "table")
        if not columns:
            raise ToolError("A table needs at least one column.")
        names = [validate_identifier(column["name"], "column") for column in columns]
        if len(set(names)) != len(names):
            raise ToolError("Column names must be unique.")
        for column in columns:
            if column["type"] not in COLUMN_TYPES:
                raise ToolError(f"Column type must be one of {sorted(COLUMN_TYPES)}.")

        same_name, similar = self._similar_tables(name, description, names)
        if same_name:
            raise ToolError(f"Table {same_name!r} already covers this. Add to it instead.")
        if similar and not confirmed_distinct:
            raise ToolError(
                f"Similar tables exist: {similar}. Use one of them, or add a column to it. "
                "Only if this is truly a different subject, retry with confirmed_distinct=true."
            )

        definitions = ", ".join(
            f"{quote_identifier(column['name'])} {COLUMN_TYPES[column['type']]}"
            + (" NOT NULL" if column.get("required") else "")
            for column in columns
        )
        with self._transaction():
            self.conn.execute(
                f"""CREATE TABLE {quote_identifier(name)} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, {definitions},
                    source_text TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"""
            )
            self.conn.execute(
                "INSERT INTO _tables (name, description, created_at) VALUES (?, ?, ?)",
                (name, description, utc_now()),
            )
            self.conn.executemany(
                """INSERT INTO _columns (table_name, name, type, description, required)
                   VALUES (?, ?, ?, ?, ?)""",
                [
                    (
                        name,
                        c["name"],
                        c["type"],
                        c.get("description", ""),
                        int(bool(c.get("required"))),
                    )
                    for c in columns
                ],
            )
            self._log(name, 0, "create_table")

    def add_column(self, table: str, name: str, column_type: str, description: str = "") -> None:
        columns = self._columns(table)
        validate_identifier(name, "column")
        if name in columns:
            raise ToolError(f"Column {name!r} already exists on {table!r}.")
        if column_type not in COLUMN_TYPES:
            raise ToolError(f"Column type must be one of {sorted(COLUMN_TYPES)}.")
        with self._transaction():
            self.conn.execute(
                f"ALTER TABLE {quote_identifier(table)} "
                f"ADD COLUMN {quote_identifier(name)} {COLUMN_TYPES[column_type]}"
            )
            self.conn.execute(
                "INSERT INTO _columns (table_name, name, type, description) VALUES (?, ?, ?, ?)",
                (table, name, column_type, description),
            )
            self._log(table, 0, "add_column", new={"column": name})

    def _similar_tables(
        self, name: str, description: str, column_names: list[str]
    ) -> tuple[str | None, list[str]]:
        existing = self.conn.execute("SELECT name, description FROM _tables").fetchall()
        if not existing:
            return None, []
        key = name_key(name)
        for row in existing:
            if name_key(row["name"]) == key:
                return row["name"], []

        similar = set()
        proposed = set(column_names)
        for row in existing:
            current = set(self._columns(row["name"]))
            if len(proposed & current) / len(proposed | current) >= SIMILAR_COLUMNS:
                similar.add(row["name"])
        texts = [f"{name}: {description}"] + [f"{r['name']}: {r['description']}" for r in existing]
        try:
            vectors = self.embedder.embed(texts, "document")
        except EmbeddingError:
            log.warning("Skipped the semantic schema check because embeddings are unavailable")
            return None, sorted(similar)
        for row, vector in zip(existing, vectors[1:]):
            if cosine(vectors[0], vector) >= SIMILAR_DESCRIPTION:
                similar.add(row["name"])
        return None, sorted(similar)

    def _columns(self, table: str) -> dict[str, str]:
        rows = self.conn.execute(
            "SELECT name, type FROM _columns WHERE table_name = ? ORDER BY rowid", (table,)
        ).fetchall()
        if not rows:
            raise ToolError(f"No table named {table!r}.")
        return {row["name"]: row["type"] for row in rows}

    def boolean_columns(self, table: str) -> frozenset[str]:
        rows = self.conn.execute(
            "SELECT name FROM _columns WHERE table_name = ? AND type = 'boolean'", (table,)
        ).fetchall()
        return frozenset(row["name"] for row in rows)

    def rows_not_marked(self, limit: int) -> list[list[dict]]:
        """One ranking per yes/no column, oldest first, of rows it isn't yet true for."""
        rankings = []
        for table in [
            row["name"] for row in self.conn.execute("SELECT name FROM _tables").fetchall()
        ]:
            for column in self.boolean_columns(table):
                name = quote_identifier(column)
                rows = self.conn.execute(
                    f"""SELECT * FROM {quote_identifier(table)}
                        WHERE {name} IS NULL OR {name} = 0 ORDER BY id LIMIT ?""",
                    (limit,),
                ).fetchall()
                rankings.append([{"table": table, "row": self._public_row(row)} for row in rows])
        return rankings

    def _required_columns(self, table: str) -> list[str]:
        rows = self.conn.execute(
            "SELECT name FROM _columns WHERE table_name = ? AND required = 1 ORDER BY rowid",
            (table,),
        ).fetchall()
        return [row["name"] for row in rows]

    def _coerce_row(self, table: str, values: dict[str, Any]) -> dict[str, Any]:
        columns = self._columns(table)
        unknown = set(values) - set(columns)
        if unknown:
            raise ToolError(
                f"Unknown columns {sorted(unknown)} on {table!r}. It has {list(columns)}. "
                "Use add_column if a new field is needed."
            )
        return {name: coerce_value(name, columns[name], value) for name, value in values.items()}

    # Rows

    def add_row(self, table: str, values: dict[str, Any], source_text: str = "") -> int:
        clean = self._coerce_row(table, values)
        if all(value is None for value in clean.values()):
            raise ToolError("A row needs at least one value.")
        missing = [name for name in self._required_columns(table) if clean.get(name) is None]
        if missing:
            raise ToolError(f"{table!r} requires {missing}. If the user didn't say, ask them.")
        now = utc_now()
        with self._transaction():
            row_id = self._insert(
                table, {**clean, "source_text": source_text, "created_at": now, "updated_at": now}
            )
        self.sync_embeddings()
        return row_id

    def update_row(self, table: str, row_id: int, changes: dict[str, Any], reason: str) -> None:
        clean = self._coerce_row(table, changes)
        if not clean:
            raise ToolError("No changes given.")
        blanked = [
            name for name in self._required_columns(table) if name in clean and clean[name] is None
        ]
        if blanked:
            raise ToolError(f"{blanked} are required on {table!r} and can't be emptied.")
        current = self.get_row(table, row_id)
        if current is None:
            raise ToolError(f"No row {table}#{row_id}.")
        with self._transaction():
            self._update(table, row_id, clean, reason)
        self.sync_embeddings()

    def get_row(self, table: str, row_id: int) -> dict | None:
        if (
            table != REMINDERS
            and not self.conn.execute("SELECT 1 FROM _tables WHERE name = ?", (table,)).fetchone()
        ):
            return None
        row = self.conn.execute(
            f"SELECT * FROM {quote_identifier(table)} WHERE id = ?", (row_id,)
        ).fetchone()
        return dict(row) if row else None

    def row_history(self, table: str, row_id: int) -> list[dict]:
        rows = self.conn.execute(
            """SELECT op, old_values, new_values, reason, changed_at FROM _history
               WHERE table_name = ? AND row_id = ? ORDER BY id""",
            (table, row_id),
        ).fetchall()
        return [
            {
                **dict(row),
                "old_values": json.loads(row["old_values"] or "null"),
                "new_values": json.loads(row["new_values"] or "null"),
            }
            for row in rows
        ]

    def _insert(self, table: str, row: dict[str, Any]) -> int:
        columns = ", ".join(quote_identifier(name) for name in row)
        placeholders = ", ".join("?" for _ in row)
        cursor = self.conn.execute(
            f"INSERT INTO {quote_identifier(table)} ({columns}) VALUES ({placeholders})",
            tuple(row.values()),
        )
        row_id = cursor.lastrowid
        self._log(table, row_id, "insert", new=row)
        self._mark_stale(table, row_id)
        return row_id

    def _update(self, table: str, row_id: int, changes: dict[str, Any], reason: str) -> None:
        current = self.get_row(table, row_id)
        assignments = ", ".join(f"{quote_identifier(name)} = ?" for name in changes)
        self.conn.execute(
            f"UPDATE {quote_identifier(table)} SET {assignments}, updated_at = ? WHERE id = ?",
            (*changes.values(), utc_now(), row_id),
        )
        self._log(
            table,
            row_id,
            "update",
            old={name: current[name] for name in changes},
            new=changes,
            reason=reason,
        )
        self._mark_stale(table, row_id)

    def _log(
        self,
        table: str,
        row_id: int,
        op: str,
        old: dict | None = None,
        new: dict | None = None,
        reason: str = "",
    ) -> None:
        if self.turn_id is None:
            self.turn_id = self.conn.execute(
                "INSERT INTO _turns (created_at) VALUES (?)", (utc_now(),)
            ).lastrowid
        self.conn.execute(
            """INSERT INTO _history
               (turn_id, table_name, row_id, op, old_values, new_values, reason, changed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                self.turn_id,
                table,
                row_id,
                op,
                json.dumps(old) if old is not None else None,
                json.dumps(new) if new is not None else None,
                reason,
                utc_now(),
            ),
        )

    # Undo and forget

    def undo(self, turn_id: int | None = None) -> bool:
        """Reverse the latest turn. An older turn_id is refused, to keep history linear."""
        latest = self.conn.execute("SELECT MAX(id) FROM _turns").fetchone()[0]
        if latest is None or (turn_id is not None and turn_id != latest):
            return False
        with self._transaction():
            entries = self.conn.execute(
                "SELECT * FROM _history WHERE turn_id = ? ORDER BY id DESC", (latest,)
            ).fetchall()
            for entry in entries:
                self._reverse(entry)
            self.conn.execute("DELETE FROM _history WHERE turn_id = ?", (latest,))
            self.conn.execute("DELETE FROM _turns WHERE id = ?", (latest,))
        self.turn_id = None
        self.sync_embeddings()
        return True

    def _reverse(self, entry: sqlite3.Row) -> None:
        table, row_id, op = entry["table_name"], entry["row_id"], entry["op"]
        name = quote_identifier(table)
        if op == "insert":
            self.conn.execute(f"DELETE FROM {name} WHERE id = ?", (row_id,))
            self._drop_embedding(table, row_id)
        elif op == "update":
            old = json.loads(entry["old_values"])
            assignments = ", ".join(f"{quote_identifier(column)} = ?" for column in old)
            self.conn.execute(
                f"UPDATE {name} SET {assignments} WHERE id = ?", (*old.values(), row_id)
            )
            self._mark_stale(table, row_id)
        elif op == "add_column":
            column = json.loads(entry["new_values"])["column"]
            self.conn.execute(f"ALTER TABLE {name} DROP COLUMN {quote_identifier(column)}")
            self.conn.execute(
                "DELETE FROM _columns WHERE table_name = ? AND name = ?", (table, column)
            )
        elif op == "create_table":
            self.conn.execute(f"DROP TABLE {name}")
            self.conn.execute("DELETE FROM _tables WHERE name = ?", (table,))
            self.conn.execute("DELETE FROM _columns WHERE table_name = ?", (table,))

    def forget(self, table: str, row_id: int) -> bool:
        """Erase a row for good, including its history, linked reminders and chat context."""
        if self.get_row(table, row_id) is None:
            return False
        linked = self.conn.execute(
            f"SELECT id FROM {REMINDERS} WHERE linked_table = ? AND linked_id = ?", (table, row_id)
        ).fetchall()
        targets = [(table, row_id)] + [(REMINDERS, row["id"]) for row in linked]
        with self._transaction():
            for target_table, target_id in targets:
                self.conn.execute(
                    f"DELETE FROM {quote_identifier(target_table)} WHERE id = ?", (target_id,)
                )
                self.conn.execute(
                    "DELETE FROM _history WHERE table_name = ? AND row_id = ?",
                    (target_table, target_id),
                )
                self._drop_embedding(target_table, target_id)
            self.conn.execute("DELETE FROM _messages")
            self.conn.execute("DELETE FROM _turns WHERE id NOT IN (SELECT turn_id FROM _history)")
        self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return True

    # Retrieval

    def read_query(self, sql: str) -> dict:
        """Run one read-only SELECT on a separate connection that can't write."""
        uri = f"file:{quote(str(self.path.resolve()))}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=5)
        deadline = time.monotonic() + QUERY_SECONDS
        try:
            conn.execute("PRAGMA query_only=ON")
            conn.set_authorizer(_read_only_authorizer)
            conn.set_progress_handler(lambda: time.monotonic() > deadline, 10_000)
            cursor = conn.execute(sql)
            if cursor.description is None:
                raise ToolError("Only SELECT queries are allowed.")
            columns = [column[0] for column in cursor.description]
            rows = cursor.fetchmany(QUERY_ROW_LIMIT + 1)
        except (sqlite3.Error, sqlite3.Warning) as exc:
            raise ToolError(f"Query failed: {exc}") from exc
        finally:
            conn.close()
        return {
            "rows": [dict(zip(columns, row)) for row in rows[:QUERY_ROW_LIMIT]],
            "truncated": len(rows) > QUERY_ROW_LIMIT,
        }

    def search(self, text: str, limit: int = 8, table: str | None = None) -> list[dict]:
        try:
            query_vector = self.embedder.embed([text], "query")[0]
        except EmbeddingError as exc:
            raise ToolError("Semantic search is unavailable right now.") from exc
        matches = self.conn.execute(
            """SELECT e.table_name, e.row_id, v.distance
               FROM _vectors v JOIN _embeddings e ON e.id = v.rowid
               WHERE v.embedding MATCH ? AND k = ? ORDER BY v.distance""",
            (sqlite_vec.serialize_float32(query_vector), limit * 4 if table else limit),
        ).fetchall()
        results = []
        for match in matches:
            if table and match["table_name"] != table:
                continue
            row = self.get_row(match["table_name"], match["row_id"])
            if row:
                results.append(
                    {
                        "table": match["table_name"],
                        "score": round(1 - match["distance"], 3),
                        "row": self._public_row(row),
                    }
                )
        return results[:limit]

    def mentions(self, name: str, limit: int = 40) -> list[dict]:
        """Every entry that names something as a whole word, so Sam doesn't match sample.

        Each comes with its table's purpose and the user's own words, which show who owes whom.
        """
        purposes = {table["name"]: table["description"] for table in self.tables()}
        results = []
        for result in self.keyword_search([name], limit, whole_word=True):
            full = self.get_row(result["table"], result["row"]["id"]) or {}
            results.append(
                {
                    **result,
                    "about": purposes.get(result["table"], ""),
                    "said": full.get("source_text", ""),
                }
            )
        return results

    def _text_columns(self) -> dict[str, list[str]]:
        searchable = {REMINDERS: ["text", "source_text"]}
        for row in self.conn.execute("SELECT name FROM _tables").fetchall():
            columns = self._columns(row["name"])
            text = [name for name, kind in columns.items() if kind == "text"]
            searchable[row["name"]] = ["source_text", *text]
        return searchable

    def keyword_search(
        self, terms: list[str], limit: int = 15, table: str | None = None, whole_word: bool = False
    ) -> list[dict]:
        """Find literal words, which embeddings can miss, such as names."""
        patterns = [
            "%" + term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
            for term in terms
            if term.strip()
        ]
        if not patterns:
            return []
        searchable = self._text_columns()
        results = []
        for name, columns in searchable.items():
            if table and name != table:
                continue
            clause = " OR ".join(
                f"{quote_identifier(column)} LIKE ? ESCAPE '\\'"
                for column in columns
                for _ in patterns
            )
            rows = self.conn.execute(
                f"SELECT * FROM {quote_identifier(name)} WHERE {clause} ORDER BY id DESC LIMIT ?",
                (
                    *(pattern for _ in columns for pattern in patterns),
                    limit * 5 if whole_word else limit,
                ),
            ).fetchall()
            if whole_word:
                words = re.compile("|".join(rf"\b{re.escape(term)}\b" for term in terms), re.I)
                rows = [
                    row
                    for row in rows
                    if any(words.search(str(value)) for value in dict(row).values())
                ]
            results.extend({"table": name, "row": self._public_row(row)} for row in rows)
        return results[:limit]

    def sync_embeddings(self) -> None:
        """Embed anything stale. Failures stay marked and are retried on the next tick."""
        stale = self.conn.execute(
            "SELECT id, table_name, row_id FROM _embeddings WHERE embedded = 0"
        ).fetchall()
        rows = [(entry, self.get_row(entry["table_name"], entry["row_id"])) for entry in stale]
        rows = [(entry, row) for entry, row in rows if row]
        if not rows:
            return
        texts = [self._embedding_text(entry["table_name"], row) for entry, row in rows]
        try:
            vectors = self.embedder.embed(texts, "document")
        except EmbeddingError:
            log.warning("Embedding failed for %d rows. Will retry.", len(rows))
            return
        with self._transaction():
            for (entry, _row), vector in zip(rows, vectors):
                self.conn.execute("DELETE FROM _vectors WHERE rowid = ?", (entry["id"],))
                self.conn.execute(
                    "INSERT INTO _vectors (rowid, embedding) VALUES (?, ?)",
                    (entry["id"], sqlite_vec.serialize_float32(vector)),
                )
                self.conn.execute(
                    "UPDATE _embeddings SET embedded = 1 WHERE id = ?", (entry["id"],)
                )

    def _mark_stale(self, table: str, row_id: int) -> None:
        self.conn.execute(
            """INSERT INTO _embeddings (table_name, row_id, embedded) VALUES (?, ?, 0)
               ON CONFLICT (table_name, row_id) DO UPDATE SET embedded = 0""",
            (table, row_id),
        )

    def _drop_embedding(self, table: str, row_id: int) -> None:
        entry = self.conn.execute(
            "SELECT id FROM _embeddings WHERE table_name = ? AND row_id = ?", (table, row_id)
        ).fetchone()
        if entry:
            self.conn.execute("DELETE FROM _vectors WHERE rowid = ?", (entry["id"],))
            self.conn.execute("DELETE FROM _embeddings WHERE id = ?", (entry["id"],))

    @staticmethod
    def _embedding_text(table: str, row: dict) -> str:
        fields = "; ".join(
            f"{name}: {value}"
            for name, value in row.items()
            if value is not None and name not in ROW_METADATA
        )
        return f"{table.lstrip('_')}: {fields}\n{row.get('source_text', '')}".strip()

    @staticmethod
    def _public_row(row: sqlite3.Row | dict) -> dict:
        row = dict(row)
        return {
            name: value
            for name, value in row.items()
            if value is not None and name not in ("updated_at", "source_text")
        }

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
    ) -> int:
        if linked_table and self.get_row(linked_table, linked_id) is None:
            raise ToolError(f"No row {linked_table}#{linked_id} to link to.")
        when = remind_at.astimezone(UTC).isoformat(timespec="seconds")
        existing = self._same_reminder(when, kind, source_text, linked_table, linked_id)
        if existing:
            return existing
        now = utc_now()
        with self._transaction():
            row_id = self._insert(
                REMINDERS,
                {
                    "text": text,
                    "remind_at": when,
                    "kind": kind,
                    "explicit": int(explicit),
                    "linked_table": linked_table,
                    "linked_id": linked_id,
                    "source_text": source_text,
                    "created_at": now,
                    "updated_at": now,
                },
            )
        self.sync_embeddings()
        return row_id

    def _same_reminder(
        self, when: str, kind: str, source_text: str, table: str | None, row_id: int | None
    ) -> int | None:
        """The model sometimes sets the same reminder twice for one message."""
        if not source_text:
            return None
        row = self.conn.execute(
            f"""SELECT id FROM {REMINDERS}
                WHERE status = 'pending' AND remind_at = ? AND kind = ? AND source_text = ?
                AND (linked_id IS NULL OR ? IS NULL OR (linked_table = ? AND linked_id = ?))""",
            (when, kind, source_text, row_id, table, row_id),
        ).fetchone()
        return row["id"] if row else None

    def cancel_reminder(self, reminder_id: int, reason: str) -> None:
        reminder = self.get_row(REMINDERS, reminder_id)
        if reminder is None or reminder["status"] not in ("pending", "sent"):
            raise ToolError(f"No active reminder #{reminder_id}.")
        with self._transaction():
            self._update(REMINDERS, reminder_id, {"status": "cancelled"}, reason)

    def pending_reminders(self) -> list[dict]:
        rows = self.conn.execute(
            f"""SELECT id, text, remind_at, kind, explicit, linked_table, linked_id
                FROM {REMINDERS} WHERE status = 'pending' ORDER BY remind_at, id"""
        ).fetchall()
        return [dict(row) for row in rows]

    def linked_rows(self) -> set[tuple[str, int]]:
        """Rows with a reminder in any state, so they're never nudged about again."""
        rows = self.conn.execute(
            f"SELECT linked_table, linked_id FROM {REMINDERS} WHERE linked_id IS NOT NULL"
        ).fetchall()
        return {(row["linked_table"], row["linked_id"]) for row in rows}

    def due_reminders(self, now: datetime) -> list[dict]:
        rows = self.conn.execute(
            f"""SELECT * FROM {REMINDERS} WHERE status = 'pending' AND remind_at <= ?
                ORDER BY remind_at, id""",
            (now.astimezone(UTC).isoformat(timespec="seconds"),),
        ).fetchall()
        return [dict(row) for row in rows]

    def sent_since(self, moment: datetime) -> int:
        """Count proactive reminders sent since a moment. Requested ones don't count."""
        return self.conn.execute(
            f"SELECT COUNT(*) FROM {REMINDERS} WHERE explicit = 0 AND sent_at >= ?",
            (moment.astimezone(UTC).isoformat(timespec="seconds"),),
        ).fetchone()[0]

    def mark_sent(self, reminder_id: int) -> None:
        now = utc_now()
        self.conn.execute(
            f"UPDATE {REMINDERS} SET status = 'sent', sent_at = ?, updated_at = ? WHERE id = ?",
            (now, now, reminder_id),
        )

    def mark_done(self, reminder_id: int) -> None:
        self.conn.execute(
            f"UPDATE {REMINDERS} SET status = 'done', updated_at = ? WHERE id = ?",
            (utc_now(), reminder_id),
        )

    def snooze(self, reminder_id: int, until: datetime) -> None:
        self.conn.execute(
            f"""UPDATE {REMINDERS} SET status = 'pending', sent_at = NULL, remind_at = ?,
                updated_at = ? WHERE id = ?""",
            (until.astimezone(UTC).isoformat(timespec="seconds"), utc_now(), reminder_id),
        )

    def mute_kind(self, kind: str) -> None:
        prefs = self.prefs()
        if kind not in prefs.muted_kinds:
            prefs.muted_kinds.append(kind)
            self.save_prefs(prefs)
        self.conn.execute(
            f"""UPDATE {REMINDERS} SET status = 'cancelled', updated_at = ?
                WHERE status = 'pending' AND explicit = 0 AND kind = ?""",
            (utc_now(), kind),
        )

    # Settings and conversation

    def prefs(self) -> Prefs:
        row = self.conn.execute("SELECT value FROM _settings WHERE key = 'prefs'").fetchone()
        return Prefs(**json.loads(row["value"])) if row else Prefs(self.default_timezone)

    def save_prefs(self, prefs: Prefs) -> None:
        self.conn.execute(
            """INSERT INTO _settings (key, value) VALUES ('prefs', ?)
               ON CONFLICT (key) DO UPDATE SET value = excluded.value""",
            (json.dumps(asdict(prefs)),),
        )

    def recent_messages(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT role, content FROM _messages ORDER BY id DESC LIMIT ?", (MESSAGE_WINDOW,)
        ).fetchall()
        messages = [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]
        while messages and messages[0]["role"] != "user":
            messages.pop(0)
        return messages

    def log_exchange(self, user_text: str, reply: str) -> None:
        now = utc_now()
        self.conn.executemany(
            "INSERT INTO _messages (role, content, created_at) VALUES (?, ?, ?)",
            [("user", user_text, now), ("assistant", reply, now)],
        )
        self.conn.execute(
            """DELETE FROM _messages WHERE id NOT IN
               (SELECT id FROM _messages ORDER BY id DESC LIMIT ?)""",
            (MESSAGE_WINDOW,),
        )

    # Export

    def export_zip(self) -> bytes:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            names = [row["name"] for row in self.conn.execute("SELECT name FROM _tables")]
            for name in [*names, REMINDERS]:
                cursor = self.conn.execute(f"SELECT * FROM {quote_identifier(name)}")
                output = io.StringIO()
                writer = csv.writer(output)
                writer.writerow(column[0] for column in cursor.description)
                writer.writerows(tuple(row) for row in cursor)
                archive.writestr(f"{name.lstrip('_')}.csv", output.getvalue())
            history = [dict(row) for row in self.conn.execute("SELECT * FROM _history ORDER BY id")]
            archive.writestr("history.json", json.dumps(history, indent=2))
        return buffer.getvalue()


def user_db_path(data_dir: Path, user_id: int) -> Path:
    return data_dir / "users" / f"{user_id}.sqlite"


def known_user_ids(data_dir: Path) -> list[int]:
    return [int(path.stem) for path in (data_dir / "users").glob("*.sqlite") if path.stem.isdigit()]


def delete_user_data(data_dir: Path, user_id: int) -> None:
    path = user_db_path(data_dir, user_id)
    for suffix in ("", "-wal", "-shm"):
        path.with_name(path.name + suffix).unlink(missing_ok=True)


class SqliteStoreProvider:
    """One SQLite file per user."""

    def __init__(self, settings: Settings, embedder: Embedder):
        self.settings = settings
        self.embedder = embedder

    def open(self, user_id: int) -> Store:
        path = user_db_path(self.settings.data_dir, user_id)
        return Store(path, self.embedder, self.settings.default_timezone)

    def user_ids(self) -> list[int]:
        return known_user_ids(self.settings.data_dir)

    def delete(self, user_id: int) -> None:
        delete_user_data(self.settings.data_dir, user_id)
