"""Every rejection and fallback path, as tables of input and expected outcome."""

import asyncio
import json
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace as NS

import pytest
from fastapi.testclient import TestClient

from recall import cli
from recall.access import READ, WRITE, Limit, TokenStore
from recall.channels import api, mcp_server
from recall.config import Settings
from recall.core.agent import TurnFailed, resolve_citations
from recall.core.nudges import find_candidates, is_quiet
from recall.core.retrieval import Retriever, find_entries
from recall.core.tools import Tools
from recall.service import RecallService, Unavailable
from recall.storage import store as store_module
from recall.storage.base import Prefs, ToolError
from recall.storage.store import LocalEmbedder, SqliteStoreProvider, Store, coerce_value

NOW = datetime(2026, 11, 2, 9, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    ("kind", "value", "expected"),
    [
        ("boolean", True, 1),
        ("integer", "7", 7),
        ("integer", 7.0, 7),
        ("real", "1.5", 1.5),
        ("date", "2026-11-02T10:00", "2026-11-02"),
        ("datetime", "2026-11-02T10:00", "2026-11-02T10:00:00"),
        ("text", " hi ", "hi"),
    ],
)
def test_values_convert(kind, value, expected):
    assert coerce_value("c", kind, value) == expected


@pytest.mark.parametrize(
    ("kind", "value"),
    [
        ("boolean", "maybe"),
        ("integer", True),
        ("integer", 1.5),
        ("integer", "abc"),
        ("real", float("nan")),
        ("real", float("inf")),
        ("date", "soon"),
        ("datetime", "soon"),
        ("unknown", "x"),
    ],
)
def test_bad_values_are_refused(kind, value):
    with pytest.raises(ToolError, match="isn't a valid"):
        coerce_value("c", kind, value)


@pytest.fixture
def notes(store):
    store.create_table("notes", "Notes", [{"name": "body", "type": "text"}])
    store.add_row("notes", {"body": "cab"})
    store.turn_id = None
    return store


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda s: s.create_table("t", "T", []), "at least one column"),
        (lambda s: s.create_table("t", "T", [{"name": "a", "type": "text"}] * 2), "unique"),
        (lambda s: s.create_table("t", "T", [{"name": "a", "type": "blob"}]), "Column type"),
        (lambda s: s.add_column("notes", "body", "text"), "already exists"),
        (lambda s: s.add_column("notes", "extra", "blob"), "Column type"),
        (lambda s: s.update_row("notes", 1, {}, "r"), "No changes"),
        (lambda s: s.update_row("notes", 9, {"body": "x"}, "r"), "No row"),
        (lambda s: s.add_reminder("x", NOW, "other", True, "notes", 9), "to link to"),
        (lambda s: s.cancel_reminder(9, "r"), "No active reminder"),
        (lambda s: s.read_query("PRAGMA user_version=1"), "Query failed|Only SELECT"),
        (lambda s: s.read_query(""), "Only SELECT"),
    ],
)
def test_store_rejections(notes, call, message):
    with pytest.raises(ToolError, match=message):
        call(notes)


def test_forget_and_done_and_cancel(notes):
    assert notes.forget("notes", 9) is False
    reminder = notes.add_reminder("x", NOW, "other", True)
    notes.cancel_reminder(reminder, "changed")
    assert notes.get_row("_reminders", reminder)["status"] == "cancelled"
    notes.mark_done(reminder)
    assert notes.get_row("_reminders", reminder)["status"] == "done"
    assert notes.keyword_search(["   "]) == []


def test_sqlite_errors_inside_a_write_roll_back(notes):
    with pytest.raises(ToolError, match="couldn't be saved"), notes._transaction():
        notes.conn.execute("INSERT INTO missing VALUES (1)")
    assert notes.conn.in_transaction is False


def test_chat_history_never_starts_with_a_reply(notes):
    for index in range(7):
        notes.log_exchange(f"q{index}", f"a{index}")
    notes.conn.execute(
        "DELETE FROM _messages WHERE role = 'user' AND id = (SELECT MIN(id) FROM _messages)"
    )
    assert notes.recent_messages()[0]["role"] == "user"


def test_embedder_loads_once_and_embeds_both_kinds(tmp_path, monkeypatch):
    built = []

    class Model:
        def __init__(self, *_args, **_kwargs):
            built.append(1)

        def query_embed(self, texts):
            return [NS(tolist=lambda: [1.0]) for _ in texts]

        passage_embed = query_embed

    monkeypatch.setattr(store_module, "TextEmbedding", Model)
    embedder = LocalEmbedder(tmp_path)
    assert embedder.embed(["a"], "query") == embedder.embed(["a"], "document") == [[1.0]]
    assert len(built) == 1


def test_repeat_citations_show_once(notes):
    assert len(resolve_citations("A [notes#1] and B [notes#1].", notes)[1]) == 1


def test_quiet_hours_shapes():
    moment = datetime(2026, 11, 2, 12, 0)
    assert is_quiet(moment, Prefs("UTC", quiet_start="10:00", quiet_end="14:00"))
    assert not is_quiet(moment, Prefs("UTC", quiet_start="09:00", quiet_end="09:00"))


def test_detection_and_picking_survive_a_search_outage(notes, embedder):
    notes.add_row("notes", {"body": "airport pickup"})
    embedder.failing = True
    assert find_candidates(notes, NOW) == []
    assert [r["row"]["id"] for r in find_entries(notes, "airport", 5)] == [2]


def test_find_entries_returns_real_matches_only(notes):
    notes.add_row("notes", {"body": "airport pickup"})
    assert [r["row"]["id"] for r in find_entries(notes, "airport", 5)] == [2]


def test_grading_is_skipped_when_nothing_was_found(store):
    class Plain:
        def ask_json(self, *_args):
            return {"queries": [], "keywords": []}

    result = Retriever(Plain(), "m").search(store, "anything")
    assert result["nothing_relevant"] and result["filtered_for_relevance"]


def test_tools_change_a_row_and_show_its_history(notes):
    tools = Tools(notes, "s", NOW, None, None)
    assert json.loads(tools.run("add_column", {"table": "notes", "name": "x", "type": "text"})) == {
        "added": "x"
    }
    change = {"table": "notes", "id": 1, "changes": {"body": "bus"}, "reason": "wrong"}
    assert json.loads(tools.run("update_row", change)) == {"updated": "notes#1"}
    assert notes.get_row("notes", 1)["body"] == "bus"
    history = json.loads(tools.run("row_history", {"table": "notes", "id": 1}))
    assert "wrong" in str(history)


def test_search_and_about_tools(notes):
    seen = []
    retriever = NS(search=lambda *args: seen.append(args) or {"results": []})
    tools = Tools(notes, "s", NOW, retriever, None)
    tools.run("search", {"text": "cab", "limit": 500})
    assert seen[0][3] == 20
    assert json.loads(tools.run("about", {"name": "cab"}))[0]["row"]["id"] == 1


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (
            {"text": "x", "remind_at": "2030-01-01T09:00", "kind": "weird", "explicit": True},
            "kind must",
        ),
        ({"text": "x", "remind_at": "soon", "kind": "other", "explicit": True}, "must look like"),
        ({"bad": 1}, "Bad arguments"),
    ],
)
def test_reminder_tool_rejections(notes, arguments, message):
    with pytest.raises(ToolError, match=message):
        Tools(notes, "s", NOW, None, None).run("set_reminder", arguments)


def test_reminder_tool_cancels_and_keeps_timezones(notes):
    tools = Tools(notes, "s", NOW, None, None)
    made = json.loads(
        tools.run(
            "set_reminder",
            {"text": "x", "remind_at": "2030-01-01T09:00+00:00", "kind": "other", "explicit": True},
        )
    )
    assert (
        json.loads(tools.run("cancel_reminder", {"id": made["id"], "reason": "r"}))["cancelled"]
        == made["id"]
    )


def test_token_listing_and_revoke_cli(tmp_path, monkeypatch, capsys):
    config = Settings("t", frozenset({1}), tmp_path)
    monkeypatch.setattr(cli.Settings, "from_env", lambda: config)

    def run(*argv):
        monkeypatch.setattr("sys.argv", ["recall-tokens", *argv])
        cli.tokens()
        return capsys.readouterr().out

    created = run("create", "--user", "1", "--name", "n", "--scopes", "read,write")
    assert created.startswith("rcl_")
    assert "user 1  n  read,write  expires" in run("list")
    assert run("revoke", "1") == "Revoked.\n"
    assert "revoked" in run("list")
    assert run("revoke", "1") == "No active token with that ID.\n"
    with pytest.raises(SystemExit, match="isn't in ALLOWED"):
        run("create", "--user", "2", "--name", "n")
    with pytest.raises(SystemExit, match="Scopes must"):
        run("create", "--user", "1", "--name", "n", "--scopes", "admin")


@pytest.mark.parametrize(
    ("allowed", "argv", "message"),
    [
        (None, [], "Pass --user"),
        (frozenset({1, 2}), [], "Pass --user"),
        (frozenset({1}), ["--user", "2"], "isn't in ALLOWED"),
    ],
)
def test_demo_needs_a_known_user(tmp_path, monkeypatch, allowed, argv, message):
    monkeypatch.setattr(cli.Settings, "from_env", lambda: Settings("t", allowed, tmp_path))
    monkeypatch.setattr("sys.argv", ["recall-demo", *argv])
    with pytest.raises(SystemExit, match=message):
        cli.demo()


def test_demo_fills_empty_memory_and_backs_up_on_fresh(tmp_path, embedder, monkeypatch):
    config = Settings("t", frozenset({1}), tmp_path)
    monkeypatch.setattr(cli.Settings, "from_env", lambda: config)
    monkeypatch.setattr(cli, "build_stores", lambda _s: SqliteStoreProvider(config, embedder))
    monkeypatch.setattr("sys.argv", ["recall-demo"])
    cli.demo()
    monkeypatch.setattr("sys.argv", ["recall-demo", "--fresh"])
    cli.demo()
    assert len(list((tmp_path / "backups").glob("*.sqlite"))) == 1


def test_missing_settings_are_named(monkeypatch):
    monkeypatch.setattr("recall.config.load_dotenv", lambda: None)
    for name in ("TELEGRAM_BOT_TOKEN", "ALLOWED_USER_IDS", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(SystemExit, match="TELEGRAM_BOT_TOKEN, ALLOWED_USER_IDS, ANTHROPIC_API_KEY"):
        Settings.from_env()


@pytest.fixture
def service(tmp_path, embedder):
    config = Settings("t", frozenset({1}), tmp_path, api_requests_per_minute=100)
    stores = SqliteStoreProvider(config, embedder)
    with stores.open(1) as store:
        store.create_table("notes", "Notes", [{"name": "body", "type": "text"}])
        store.add_row("notes", {"body": "cab"})
    agent = NS(capture=lambda store, text: [])
    return RecallService(
        config, TokenStore(tmp_path / "a.sqlite"), stores, agent, NS(spend=lambda _u: Limit.NONE)
    )


def test_failed_remember_is_unavailable(service):
    def fails(store, text):
        raise TurnFailed("x")

    service.agent = NS(capture=fails)
    identity = service.tokens.verify(service.tokens.create(1, "w", {WRITE}, 1))
    with pytest.raises(Unavailable):
        service.remember(identity, "x")


def test_remaining_api_routes(service):
    token = service.tokens.create(1, "w", {READ, WRITE}, 1)
    headers = {"Authorization": f"Bearer {token}"}
    client = TestClient(api.create_app(service))
    assert client.get("/v1/about/cab", headers=headers).json()["results"]
    body = {"text": "x", "remind_at": "2030-01-01T09:00"}
    assert client.post("/v1/reminders", json=body, headers=headers).status_code == 200
    assert client.get("/v1/reminders", headers=headers).json()["reminders"][0]["text"] == "x"


def test_api_main_stays_on_localhost(service, tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(api.Settings, "from_env", lambda: service.settings)
    monkeypatch.setattr(api, "build_service", lambda _s: service)
    monkeypatch.setattr(api, "setup", lambda *_: None)
    monkeypatch.setattr(api.uvicorn, "run", lambda _app, **kwargs: seen.update(kwargs))
    api.main()
    assert seen["host"] == "127.0.0.1"


def test_every_mcp_tool_answers(service, monkeypatch):
    monkeypatch.setenv("RECALL_TOKEN", service.tokens.create(1, "w", {READ, WRITE}, 1))
    server = mcp_server.create_server(service, service.settings)

    async def run():
        calls = {
            "search_memory": {"text": "cab"},
            "about": {"name": "cab"},
            "get_entry": {"table": "notes", "id": 1},
            "list_reminders": {},
            "remember": {"text": "x"},
            "set_reminder": {"text": "x", "remind_at": "2030-01-01T09:00"},
        }
        return [await server.call_tool(name, args) for name, args in calls.items()]

    assert all("saved data" in str(result) for result in asyncio.run(run()))


@pytest.mark.parametrize(("argv", "transport"), [([], "stdio"), (["--http"], "streamable-http")])
def test_mcp_main_picks_transport_and_stays_on_localhost(service, monkeypatch, argv, transport):
    calls = []
    monkeypatch.setattr(mcp_server.Settings, "from_env", lambda: service.settings)
    monkeypatch.setattr(mcp_server, "build_service", lambda _s: service)
    monkeypatch.setattr(mcp_server, "setup", lambda *_: None)
    monkeypatch.setattr(mcp_server.MCPServer, "run", lambda _self, *a, **k: calls.append((a, k)))
    monkeypatch.setattr("sys.argv", ["recall-mcp", *argv])
    mcp_server.main()
    assert calls[0][0] == (transport,)
    assert calls[0][1].get("host", "127.0.0.1") == "127.0.0.1"


def test_a_new_store_creates_its_folder_with_private_permissions(tmp_path, embedder):
    path = tmp_path / "new" / "u.sqlite"
    with Store(path, embedder, "UTC") as store:
        assert store.tables() == []
    assert path.stat().st_mode & 0o777 == 0o600


def test_undo_leaves_an_unknown_history_entry_alone(notes):
    notes.add_row("notes", {"body": "second"})
    notes.conn.execute(
        """INSERT INTO _history (turn_id, table_name, row_id, op, changed_at)
           VALUES (?, 'notes', 1, 'mystery', 'x')""",
        (notes.turn_id,),
    )
    assert notes.undo()
    assert notes.get_row("notes", 1) and notes.get_row("notes", 2) is None


def test_search_skips_a_vector_whose_row_is_gone(notes):
    notes.conn.execute("DELETE FROM notes WHERE id = 1")
    assert notes.search("cab") == []


def test_forget_works_on_a_row_with_no_embedding(notes):
    notes.conn.execute("DELETE FROM _embeddings")
    assert notes.forget("notes", 1)


def test_muting_twice_keeps_one_entry(notes):
    notes.mute_kind("promise")
    notes.mute_kind("promise")
    assert notes.prefs().muted_kinds == ["promise"]


def test_mcp_only_accepts_this_machine_and_its_public_address(service):
    settings = replace(service.settings, mcp_public_url="https://recall.example.com")
    allowed = mcp_server.transport_security(settings)
    assert allowed.allowed_hosts == ["127.0.0.1:8788", "recall.example.com"]
    assert allowed.allowed_origins == ["https://recall.example.com"]
