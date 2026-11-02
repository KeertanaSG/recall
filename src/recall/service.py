"""The one door to a user's memory.

The bot, API and MCP server are thin layers over RecallService, so they all pass the same
permission checks. No method takes a user ID: the user always comes from a verified identity,
so a caller can't ask for someone else's data. Tokens are also rate limited and audited.
"""

import json
from contextlib import contextmanager, nullcontext
from datetime import UTC, datetime

from recall.access import OWNER, READ, WRITE, Identity, Limit, TokenStore, UsageLimiter
from recall.config import Settings
from recall.core.agent import Agent, Reply, TurnFailed
from recall.core.nudges import Nudger
from recall.core.retrieval import hybrid_search
from recall.core.tools import Tools
from recall.logs import bind, get_logger, measured, user_ref
from recall.storage.base import StoreProvider, ToolError

log = get_logger(__name__)

LIMIT_MESSAGES = {
    Limit.USER: "You've reached today's message limit. Please try again tomorrow.",
    Limit.TOTAL: "Recall has reached its limit for today. Please try again tomorrow.",
}


class AccessDenied(Exception):
    """The token lacks the scope, or the user isn't allowed."""


class RateLimited(Exception):
    """The token or user has reached a limit."""


class NotFound(Exception):
    """The entry doesn't exist for this user."""


class Unavailable(Exception):
    """A model step failed, and nothing was saved."""


EXPECTED = (AccessDenied, NotFound, RateLimited, ToolError, Unavailable)


class RecallService:
    def __init__(
        self,
        settings: Settings,
        tokens: TokenStore,
        stores: StoreProvider,
        agent: Agent,
        limiter: UsageLimiter,
        nudger: Nudger | None = None,
    ):
        self.settings = settings
        self.tokens = tokens
        self.stores = stores
        self.agent = agent
        self.limiter = limiter
        self.nudger = nudger

    def _record(self, identity: Identity, action: str, outcome: str) -> None:
        if identity.token_id is None:
            return
        try:
            self.tokens.record(identity, action, outcome)
        except Exception:
            # This runs as a request ends, so it mustn't replace the request's own result.
            log.exception("audit_failed", action=action, audit_outcome=outcome)

    def _authorize(self, identity: Identity, action: str, scope: str) -> None:
        if scope not in identity.scopes or not self.settings.allows(identity.user_id):
            log.warning("access_denied", action=action, scope=scope)
            self._record(identity, action, "denied")
            raise AccessDenied(f"This token can't {action}.")

    def _spend(self, identity: Identity) -> None:
        limit = self.limiter.spend(identity.user_id)
        if limit is not Limit.NONE:
            raise RateLimited(LIMIT_MESSAGES[limit])

    @contextmanager
    def access(self, identity: Identity, action: str, scope: str):
        """Authorize a call, open that user's store, and audit the outcome."""
        by_token = identity.token_id is not None
        who = {"user": user_ref(identity.user_id), "channel": "token" if by_token else "telegram"}
        with bind(**who, **({"token_id": identity.token_id} if by_token else {})):
            self._authorize(identity, action, scope)
            limit = self.settings.api_requests_per_minute
            if by_token and not self.tokens.within_rate_limit(identity.token_id, limit):
                log.warning("rate_limited", action=action)
                self._record(identity, action, "rate_limited")
                raise RateLimited("Too many requests. Try again in a minute.")
            outcome = "ok"
            # The bot opens a store every tick, so only token calls are logged as requests.
            request = (
                measured(log, "request", EXPECTED, action=action) if by_token else nullcontext()
            )
            try:
                with request, self.stores.open(identity.user_id) as store:
                    yield store
            except BaseException:
                outcome = "error"
                raise
            finally:
                self._record(identity, action, outcome)

    def users(self) -> list[int]:
        """Everyone with saved data who is still allowed, for background work like reminders."""
        return [user_id for user_id in self.stores.user_ids() if self.settings.allows(user_id)]

    def chat(self, identity: Identity, text: str, image: bytes | None = None) -> Reply:
        """Run a full turn, which can read and change everything, so it's for the owner only."""
        with self.access(identity, "chat", OWNER) as store:
            self._spend(identity)
            return self.agent.respond(store, text, image=image)

    def scan_nudges(self, identity: Identity, force: bool = False) -> list[int]:
        """Look for unfinished business. Runs at most daily unless forced."""
        with self.access(identity, "scan for nudges", OWNER) as store:
            return self.nudger.scan(store, force=force)

    def delete_all(self, identity: Identity) -> None:
        self._authorize(identity, "delete everything", OWNER)
        self.stores.delete(identity.user_id)

    def search(
        self, identity: Identity, text: str, table: str | None = None, limit: int = 8
    ) -> list[dict]:
        """Search by meaning and keyword, with no model call. The caller judges relevance."""
        with self.access(identity, "search", READ) as store:
            return hybrid_search(store, text, min(max(limit, 1), 20), table)

    def about(self, identity: Identity, name: str) -> list[dict]:
        """Everything that mentions a person, place or thing, across all tables."""
        with self.access(identity, "about", READ) as store:
            return store.mentions(name)

    def query(self, identity: Identity, sql: str) -> dict:
        with self.access(identity, "query", READ) as store:
            return store.read_query(sql)

    def list_tables(self, identity: Identity) -> list[dict]:
        with self.access(identity, "list_tables", READ) as store:
            return store.tables()

    def get_entry(self, identity: Identity, table: str, row_id: int) -> dict:
        with self.access(identity, "get_entry", READ) as store:
            row = store.get_row(table, row_id)
            if row is None:
                raise NotFound(f"No entry {table}#{row_id}.")
            return {"entry": row, "history": store.row_history(table, row_id)}

    def list_reminders(self, identity: Identity) -> list[dict]:
        with self.access(identity, "list_reminders", READ) as store:
            return store.pending_reminders()

    def remember(self, identity: Identity, text: str) -> dict:
        """Save through the agent, so table design still applies. Returns only what was added."""
        with self.access(identity, "remember", WRITE) as store:
            self._spend(identity)
            try:
                return {"saved": self.agent.capture(store, text)}
            except TurnFailed as exc:
                raise Unavailable("Nothing was saved. Please try again.") from exc

    def set_reminder(
        self, identity: Identity, text: str, remind_at: str, kind: str = "other"
    ) -> dict:
        """Schedule a reminder the user asked for. It's delivered on Telegram."""
        with self.access(identity, "set_reminder", WRITE) as store:
            tools = Tools(store, text, datetime.now(UTC), None, None)
            arguments = {"text": text, "remind_at": remind_at, "kind": kind, "explicit": True}
            return json.loads(tools.run("set_reminder", arguments))
