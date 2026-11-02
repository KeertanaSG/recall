"""The Claude agent loop, plus the citation check on its final answer."""

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from recall.core.retrieval import Retriever, hybrid_search, result_key
from recall.core.tools import ADD_ONLY_TOOLS, TableDesigner, Tools
from recall.llm.base import LLM, LLMError, Stop, ToolResult
from recall.logs import add, bind, get_logger, measured, new_id
from recall.prompts import render
from recall.storage.base import ROW_METADATA, MemoryStore, ToolError

log = get_logger(__name__)

MAX_STEPS = 12
MAX_SOURCES_LISTED = 3
MAX_SOURCE_DETAILS = 2
MAX_SOURCE_LENGTH = 80
PREFETCH_LIMIT = 15
REMINDER_INTERNALS = {"kind", "explicit", "status", "sent_at", "linked_table", "linked_id"}
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")
CITED_ROWS = r"_?[a-z][a-z0-9_]*#\d+(?:\s*,\s*\d+)*"
CITATION = re.compile(rf"\s*\[{CITED_ROWS}(?:\s*,\s*{CITED_ROWS})*\]")
CITED = re.compile(r"(_?[a-z][a-z0-9_]*)#(\d+(?:\s*,\s*\d+)*)")

SYSTEM_PROMPT = render("agent_system")

CAPTURE_NOTE = render("capture_note")


class TurnFailed(Exception):
    """The turn couldn't finish. Any partial writes have been rolled back."""


@dataclass
class Source:
    table: str
    id: int
    summary: str
    on: str = ""


@dataclass
class Reply:
    text: str
    sources: list[Source] = field(default_factory=list)

    def render(self) -> str:
        if not self.sources:
            return self.text
        lines = summarize_sources(self.sources)
        if len(self.sources) <= MAX_SOURCES_LISTED or len(lines) == len(self.sources):
            lines = [source.summary for source in self.sources]  # Grouping wouldn't shorten it.
        return f"{self.text}\n\nFrom your notes\n" + "\n".join(lines)


def short_date(value: str) -> str:
    return datetime.fromisoformat(value[:10]).strftime("%b %-d")


def row_date(row: dict) -> str:
    """When the entry happened if it says, otherwise when it was saved."""
    for name, value in row.items():
        if name not in ROW_METADATA and isinstance(value, str) and DATE.match(value):
            return value[:10]
    return row["created_at"][:10]


def summarize_row(row: dict, hidden: frozenset[str] = frozenset(), with_date: bool = True) -> str:
    """Show a saved row in one short line: the main value, then the details that say the most."""
    fields = [
        (name, value)
        for name, value in row.items()
        if value is not None
        and name not in ROW_METADATA
        and name not in REMINDER_INTERNALS
        and name not in hidden
    ]
    main = [str(fields[0][1])] if fields else []
    numbers, words = [], []
    for name, value in fields[1:]:
        if isinstance(value, (int, float)):
            value = int(value) if float(value).is_integer() else value
            numbers.append(f"{name.replace('_', ' ')} {value}")
        elif not DATE.match(str(value)):
            words.append(str(value))
    parts = main + (numbers + words)[:MAX_SOURCE_DETAILS]
    if with_date:
        dated = any(name != "created_at" and DATE.match(str(value)) for name, value in fields)
        parts.append(f"{'' if dated else 'saved '}{short_date(row_date(row))}")
    line = " · ".join(parts)
    return line if len(line) <= MAX_SOURCE_LENGTH else line[: MAX_SOURCE_LENGTH - 1] + "…"


def summarize_sources(sources: list[Source]) -> list[str]:
    """A long list says nothing at a glance, so group it by subject with its date range."""
    groups: dict[str, list[Source]] = {}
    for source in sources:
        groups.setdefault(source.table, []).append(source)
    lines = []
    for table, group in sorted(groups.items(), key=lambda item: -len(item[1])):
        dates = sorted(source.on for source in group if source.on)
        span = short_date(dates[0]) if dates else ""
        if dates and dates[0] != dates[-1]:
            span += f" to {short_date(dates[-1])}"
        entries = "1 entry" if len(group) == 1 else f"{len(group)} entries"
        name = table.lstrip("_").replace("_", " ")
        lines.append(f"{name}: {entries}" + (f", {span}" if span else ""))
    return lines


def resolve_citations(text: str, store: MemoryStore) -> tuple[str, list[Source]]:
    """Replace citation markers with sources read from the database, not from the model."""
    sources: list[Source] = []
    missing = False
    seen: set[tuple[str, int]] = set()
    for match in (
        cited for citation in CITATION.finditer(text) for cited in CITED.finditer(citation[0])
    ):
        table = match.group(1)
        for row_id in (int(part) for part in match.group(2).split(",")):
            if (table, row_id) in seen:
                continue
            seen.add((table, row_id))
            row = store.get_row(table, row_id)
            if row is None:
                missing = True
            else:
                summary = summarize_row(row, store.boolean_columns(table))
                sources.append(Source(table, row_id, summary, row_date(row)))
    clean = CITATION.sub("", text).strip()
    if missing:
        clean += (
            "\n\nWarning: part of this answer cites an entry I can't find, so don't rely on it."
        )
    return clean, sources


def message_context(store: MemoryStore, now: datetime) -> str:
    tz = ZoneInfo(store.prefs().timezone)
    local = now.astimezone(tz)
    calendar = ", ".join(
        (local + timedelta(days=offset)).strftime("%a %Y-%m-%d") for offset in range(1, 15)
    )
    tables = [
        f"{table['name']}({', '.join(c['name'] + ('*' if c['required'] else '') for c in table['columns'])})"
        f" - {table['row_count']} rows"
        for table in store.tables()
    ]
    return render(
        "context",
        time=local.strftime("%A %Y-%m-%d %H:%M"),
        zone=tz.key,
        calendar=calendar,
        tables="; ".join(tables) or "none yet",
    )


def prefetched_entries(store: MemoryStore, text: str) -> str:
    """Run the free local searches first, so a simple question may need no search tool."""
    entries = {result_key(r): r["row"] for r in hybrid_search(store, text, PREFETCH_LIMIT)}
    if not entries:
        return ""
    return render("prefetch", entries=json.dumps(entries, ensure_ascii=False, default=str))


class Agent:
    def __init__(
        self,
        llm: LLM,
        model: str,
        effort: str,
        retrieval_model: str | None = None,
        design_model: str | None = None,
        prefetch: bool = False,
    ):
        self.llm = llm
        self.model = model
        self.effort = effort
        self.prefetch = prefetch
        self.retriever = Retriever(llm, retrieval_model or model)
        self.designer = TableDesigner(llm, design_model or model)

    def respond(
        self, store: MemoryStore, text: str, now: datetime | None = None, image: bytes | None = None
    ) -> Reply:
        now = now or datetime.now(UTC)
        with (
            bind(turn=new_id()),
            measured(log, "turn", kind="chat", has_image=image is not None) as stats,
        ):
            tools = Tools(store, text, now, self.retriever, self.designer)
            context = message_context(store, now)
            if self.prefetch:
                context += prefetched_entries(store, text)
            content = render("turn", context=context, text=text)
            # Only this turn sees a photo: it's never stored, and the chat history keeps the text.
            messages = [*store.recent_messages(), self.llm.user_message(content, image)]
            final = self._attempt(store, messages, tools)
            clean, sources = resolve_citations(final, store)
            stats.update(saved=len(tools.saved), cited=len(sources))
            if store.turn_id is not None:
                sources = []  # A save confirms itself, so a source list would just repeat it.
            try:
                store.log_exchange(text, clean)
            except Exception:
                # The turn already succeeded, and an error here would invite a duplicate resend.
                log.exception("chat_history_failed")
            return Reply(clean, sources)

    def capture(self, store: MemoryStore, text: str, now: datetime | None = None) -> list[str]:
        """Save a message from another app, and return what was added.

        Whoever calls this may only hold write access, so the turn can't read saved entries or
        the chat, can't change anything already saved, and its reply is never returned. It stays
        out of the chat history too, so outside text is never replayed as something the user said.
        """
        now = now or datetime.now(UTC)
        with bind(turn=new_id()), measured(log, "turn", kind="capture") as stats:
            tools = Tools(store, text, now, None, self.designer, ADD_ONLY_TOOLS)
            content = render("turn", context=message_context(store, now), text=text)
            self._attempt(store, [self.llm.user_message(content)], tools, CAPTURE_NOTE)
            stats["saved"] = len(tools.saved)
            if tools.failures and not tools.saved:
                raise TurnFailed("Every save failed.")  # Not the same as nothing to save.
            return tools.saved

    def _attempt(
        self, store: MemoryStore, messages: list[dict], tools: Tools, note: str = ""
    ) -> str:
        try:
            return self._run(messages, tools, note)
        except Exception as exc:
            # Any failure, expected or not, must leave no half-finished turn behind.
            if not isinstance(exc, LLMError | TurnFailed):
                log.exception("turn_crashed")
            if store.turn_id is not None:
                store.undo(store.turn_id)
            raise TurnFailed(str(exc)) from exc

    def _run(self, messages: list[dict], tools: Tools, note: str = "") -> str:
        for _ in range(MAX_STEPS):
            reply = self.llm.chat(
                self.model, SYSTEM_PROMPT, messages, tools.schemas, self.effort, note
            )
            if reply.stop is Stop.REFUSED:
                raise TurnFailed("The model declined this message.")
            if reply.stop is Stop.CUT_OFF:
                raise TurnFailed("The model's response was cut off.")

            messages.append(self.llm.assistant_message(reply))
            if reply.stop is not Stop.TOOL_USE:
                return reply.text.strip()

            results = []
            for call in reply.tool_calls:
                add("tools", call.name)
                try:
                    with measured(log, "tool_call", expected=(ToolError,), tool=call.name):
                        results.append(ToolResult(call.id, tools.run(call.name, call.input)))
                except ToolError as exc:
                    tools.failures += 1
                    results.append(ToolResult(call.id, str(exc), True))
            messages.append(self.llm.tool_results_message(results))
        raise TurnFailed("The agent took too many steps.")
