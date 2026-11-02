"""Structured logs: every line is one event with metadata, written as JSON to data/logs/.

    log = get_logger(__name__)
    log.info("reminder_sent", kind="promise")

    with bind(user=user_ref(user_id)):                  # Added to every event inside the block.
        with measured(log, "search") as stats:          # Logs duration_ms and outcome at the end.
            stats["results"] = 3
            count(input_tokens=120)                     # Also adds up in any outer block.

Logs carry IDs, counts and timings, never memory text. Exceptions keep their type and stack but
not their message, because messages can quote saved values.
"""

import json
import logging
import re
import time
import traceback
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from hashlib import sha256
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

LOG_BYTES = 5_000_000
LOG_BACKUPS = 3
BOT_TOKEN = re.compile(r"bot\d+:[\w-]+")
QUIET_LOGGERS = (
    "apscheduler",  # Reminder ticks would be most of the log.
    "httpx",  # Logs request URLs, and Telegram URLs contain the bot token.
    "httpx2",
    "uvicorn.access",  # Logs paths, and /v1/about/{name} contains a name.
)

_context: ContextVar[dict] = ContextVar("log_context", default={})
_open_blocks: ContextVar[tuple[dict, ...]] = ContextVar("log_blocks", default=())
_process = "recall"


class EventLogger(logging.LoggerAdapter):
    """Keyword arguments become fields of the event."""

    def process(self, msg, kwargs):
        reserved = {key: kwargs.pop(key) for key in ("exc_info", "stack_info") if key in kwargs}
        return msg, {**reserved, "extra": {"fields": kwargs}}


def get_logger(name: str) -> EventLogger:
    return EventLogger(logging.getLogger(name), {})


def user_ref(user_id: int) -> str:
    """A stable pseudonym, so logs can follow one user without holding their Telegram ID."""
    return sha256(f"recall:{user_id}".encode()).hexdigest()[:12]


def new_id() -> str:
    return uuid.uuid4().hex[:12]


@contextmanager
def bind(**fields) -> Iterator[None]:
    token = _context.set({**_context.get(), **fields})
    try:
        yield
    finally:
        _context.reset(token)


def count(**numbers: int) -> None:
    """Add to totals in every open measured block, such as tokens within a turn."""
    for stats in _open_blocks.get():
        for name, value in numbers.items():
            stats[name] = stats.get(name, 0) + value


def add(name: str, value: Any) -> None:
    """Append to a list in every open measured block, such as the tools a turn used."""
    for stats in _open_blocks.get():
        stats.setdefault(name, []).append(value)


@contextmanager
def measured(
    log: EventLogger, event: str, expected: tuple[type[Exception], ...] = (), **fields
) -> Iterator[dict]:
    """Log one event when the block ends. Expected errors are warnings, with no stack."""
    stats = dict(fields)
    blocks = _open_blocks.set((*_open_blocks.get(), stats))
    started = time.perf_counter()
    failure = None
    try:
        yield stats
    except Exception as exc:
        failure = exc
        raise
    finally:
        _open_blocks.reset(blocks)
        stats["duration_ms"] = round((time.perf_counter() - started) * 1000)
        if failure is None:
            log.info(event, outcome="ok", **stats)
        elif isinstance(failure, expected):
            log.warning(event, outcome="rejected", error_type=type(failure).__name__, **stats)
        else:
            log.error(event, outcome="error", exc_info=failure, **stats)


def _event(record: logging.LogRecord) -> dict:
    event = {
        "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
        "level": record.levelname.lower(),
        "process": _process,
        "logger": record.name,
        "event": record.getMessage(),
        **_context.get(),
        **getattr(record, "fields", {}),
    }
    if record.exc_info and record.exc_info[1]:
        kind, _, stack = record.exc_info
        event["error_type"] = kind.__name__
        event["stack"] = [
            f"{frame.filename.rsplit('/', 1)[-1]}:{frame.lineno} {frame.name}"
            for frame in traceback.extract_tb(stack)
        ]
    return event


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return BOT_TOKEN.sub("bot[redacted]", json.dumps(_event(record), default=str))


class ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        event = _event(record)
        hidden = ("ts", "process", "logger", "event", "level", "stack")
        details = " ".join(f"{k}={v}" for k, v in event.items() if k not in hidden)
        line = f"{event['ts'][11:19]} {record.levelname:7} {event['event']} {details}".rstrip()
        return BOT_TOKEN.sub("bot[redacted]", line)


def setup(data_dir: Path, process: str) -> None:
    """Log to data/logs/<process>.jsonl and to stderr, which MCP's stdio transport leaves free."""
    global _process
    _process = process
    folder = data_dir / "logs"
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = folder / f"{process}.jsonl"
    file = RotatingFileHandler(path, maxBytes=LOG_BYTES, backupCount=LOG_BACKUPS)
    file.setFormatter(JsonFormatter())
    path.chmod(0o600)
    console = logging.StreamHandler()
    console.setFormatter(ConsoleFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[file, console], force=True)
    for name in QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    get_logger(__name__).info("process_started")
