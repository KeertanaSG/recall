import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from recall.access import OWNER, READ, WRITE, TokenStore, UsageLimiter, telegram_identity
from recall.channels.api import create_app
from recall.channels.mcp_server import BearerVerifier, create_server
from recall.config import Settings
from recall.service import AccessDenied, NotFound, RateLimited, RecallService
from recall.storage.store import SqliteStoreProvider

NOW = datetime(2026, 11, 2, tzinfo=UTC)


@pytest.fixture
def agent():
    return SimpleNamespace(
        respond=lambda store, text, image=None: f"Replied to {text}",
        capture=lambda store, text: ["notes#2"],
    )


@pytest.fixture
def service(tmp_path, embedder, agent):
    settings = Settings("t", frozenset({1, 2}), tmp_path, "Asia/Kolkata", api_requests_per_minute=5)
    stores = SqliteStoreProvider(settings, embedder)
    with stores.open(1) as store:
        store.create_table("notes", "Notes", [{"name": "body", "type": "text"}])
        store.add_row("notes", {"body": "secret of user one"})
    return RecallService(
        settings,
        TokenStore(tmp_path / "auth.sqlite"),
        stores,
        agent,
        UsageLimiter(tmp_path / "u.sqlite", 2, 100),
    )


@pytest.fixture
def reader(service):
    return service.tokens.create(1, "reader", {READ}, 30)


@pytest.fixture
def writer(service):
    return service.tokens.create(1, "writer", {READ, WRITE}, 30)


@pytest.fixture
def client(service):
    return TestClient(create_app(service))


def auth(token):
    return {"Authorization": f"Bearer {token}"}


def test_tokens_are_stored_only_as_hashes(service, reader):
    assert reader.encode() not in service.tokens.path.read_bytes()


def test_expired_revoked_and_unknown_tokens_fail(service, reader):
    assert service.tokens.verify("rcl_nope") is None
    assert service.tokens.verify(reader, NOW + timedelta(days=4000)) is None
    assert service.tokens.revoke(service.tokens.verify(reader).token_id)
    assert service.tokens.verify(reader) is None


def test_scopes_must_be_known(service):
    with pytest.raises(ValueError):
        service.tokens.create(1, "x", {"admin"}, 1)
    with pytest.raises(ValueError):
        service.tokens.create(1, "x", set(), 1)


def test_read_token_cannot_write(service, reader):
    with pytest.raises(AccessDenied):
        service.remember(service.tokens.verify(reader), "x")


def test_tokens_can_never_act_as_the_owner(service, writer):
    identity = service.tokens.verify(writer)
    with pytest.raises(AccessDenied):
        service.chat(identity, "what have I saved?")
    with pytest.raises(AccessDenied):
        service.delete_all(identity)
    with pytest.raises(ValueError):
        service.tokens.create(1, "x", {OWNER}, 1)
    assert service.list_tables(identity)


def test_telegram_passes_the_same_door(service):
    assert service.chat(telegram_identity(1), "hi") == "Replied to hi"
    assert service.users() == [1]
    with pytest.raises(AccessDenied):
        service.chat(telegram_identity(3), "hi")
    service.delete_all(telegram_identity(1))
    assert service.users() == []


def test_chat_shares_the_daily_limit_with_tokens(service, writer):
    service.remember(service.tokens.verify(writer), "a")
    service.chat(telegram_identity(1), "b")
    with pytest.raises(RateLimited, match="today's message limit"):
        service.chat(telegram_identity(1), "c")


def test_removed_user_loses_access_even_with_a_valid_token(service):
    token = service.tokens.create(3, "old", {READ}, 30)
    with pytest.raises(AccessDenied):
        service.list_tables(service.tokens.verify(token))


def test_users_only_see_their_own_data(service):
    other = service.tokens.verify(service.tokens.create(2, "other", {READ}, 30))
    assert service.list_tables(other) == []
    assert service.search(other, "secret") == []
    with pytest.raises(NotFound):
        service.get_entry(other, "notes", 1)


def test_rate_limit_applies_per_token(service, reader, writer):
    identity = service.tokens.verify(reader)
    for _ in range(5):
        service.list_tables(identity)
    with pytest.raises(RateLimited):
        service.list_tables(identity)
    assert service.list_tables(service.tokens.verify(writer)) == service.list_tables(
        service.tokens.verify(writer)
    )


def test_remember_counts_against_daily_limit(service, writer):
    identity = service.tokens.verify(writer)
    service.remember(identity, "a")
    service.remember(identity, "b")
    with pytest.raises(RateLimited):
        service.remember(identity, "c")


def test_about_finds_a_name_only_as_a_whole_word(service, reader):
    identity = service.tokens.verify(reader)
    assert [r["row"]["id"] for r in service.about(identity, "secret")] == [1]
    assert service.about(identity, "secre") == []


def test_audit_records_actions_but_never_memory_text(service, writer):
    identity = service.tokens.verify(writer)
    service.remember(identity, "zebra private note")
    service.search(identity, "zebra private note")
    data = service.tokens.path.read_bytes()
    assert b"remember" in data and b"zebra" not in data


def test_reminder_needs_write_and_appears_in_the_list(service, writer):
    identity = service.tokens.verify(writer)
    service.set_reminder(identity, "Call the bank", "2030-01-01T09:00")
    assert service.list_reminders(identity)[0]["text"] == "Call the bank"


def test_every_data_route_requires_a_token(client):
    calls = [
        ("post", "/v1/search", {"text": "x"}),
        ("post", "/v1/query", {"sql": "SELECT 1"}),
        ("get", "/v1/tables", None),
        ("get", "/v1/about/x", None),
        ("get", "/v1/entries/notes/1", None),
        ("get", "/v1/reminders", None),
        ("post", "/v1/reminders", {"text": "x", "remind_at": "2030-01-01T09:00"}),
        ("post", "/v1/remember", {"text": "x"}),
    ]
    for method, path, body in calls:
        assert client.request(method, path, json=body).status_code == 401, path
        assert (
            client.request(method, path, json=body, headers=auth("rcl_bad")).status_code == 401
        ), path


def test_api_has_no_delete_route_or_public_docs(client):
    routes = [route for route in client.app.routes if getattr(route, "path", "").startswith("/v1")]
    assert not any("DELETE" in route.methods for route in routes)
    assert client.get("/docs").status_code == 404 and client.get("/openapi.json").status_code == 404
    assert client.get("/v1/health").json() == {"status": "ok"}


def test_api_reads_and_keeps_queries_read_only(client, reader):
    headers = auth(reader)
    assert (
        client.post("/v1/search", json={"text": "secret"}, headers=headers).json()["results"][0][
            "row"
        ]["id"]
        == 1
    )
    assert (
        client.post("/v1/query", json={"sql": "DELETE FROM notes"}, headers=headers).status_code
        == 400
    )
    assert (
        client.post(
            "/v1/query", json={"sql": "SELECT * FROM _history"}, headers=headers
        ).status_code
        == 400
    )
    assert client.get("/v1/entries/_messages/1", headers=headers).status_code == 404
    assert (
        client.get("/v1/entries/notes/1", headers=headers).json()["entry"]["body"]
        == "secret of user one"
    )


def test_api_maps_errors_to_status_codes(client, reader, writer):
    assert client.post("/v1/remember", json={"text": "x"}, headers=auth(reader)).status_code == 403
    assert client.post("/v1/remember", json={"text": "x"}, headers=auth(writer)).json() == {
        "saved": ["notes#2"]
    }
    assert (
        client.post("/v1/remember", json={"text": "x" * 5000}, headers=auth(writer)).status_code
        == 422
    )


def test_mcp_tools_have_no_delete_and_label_reads(service, reader, monkeypatch):
    monkeypatch.setenv("RECALL_TOKEN", reader)
    server = create_server(service, service.settings)

    async def run():
        names = {tool.name for tool in await server.list_tools()}
        return names, await server.call_tool("list_tables", {})

    names, result = asyncio.run(run())
    assert names == {
        "about",
        "search_memory",
        "query_memory",
        "list_tables",
        "get_entry",
        "list_reminders",
        "remember",
        "set_reminder",
    }
    assert "saved data" in str(result)


def test_mcp_rejects_a_missing_token_and_write_without_scope(service, reader, monkeypatch):
    server = create_server(service, service.settings)
    monkeypatch.delenv("RECALL_TOKEN", raising=False)
    with pytest.raises(Exception, match="token"):
        asyncio.run(server.call_tool("list_tables", {}))
    monkeypatch.setenv("RECALL_TOKEN", reader)
    with pytest.raises(Exception, match="can't remember"):
        asyncio.run(server.call_tool("remember", {"text": "x"}))
    with pytest.raises(Exception, match="Query failed"):
        asyncio.run(server.call_tool("query_memory", {"sql": "SELECT * FROM missing"}))


def test_bearer_verifier(service, reader):
    verifier = BearerVerifier(service.tokens)
    assert asyncio.run(verifier.verify_token("rcl_bad")) is None
    access = asyncio.run(verifier.verify_token(reader))
    assert access.scopes == [READ]


def test_a_crash_is_a_plain_500_with_no_detail(service, reader):
    def crash(_identity):
        raise RuntimeError("my private note")

    service.list_tables = crash
    client = TestClient(create_app(service), raise_server_exceptions=False)
    response = client.get("/v1/tables", headers=auth(reader))
    assert response.status_code == 500 and response.json() == {"detail": "Something went wrong."}


def test_a_failed_audit_write_keeps_the_result(service, reader, monkeypatch):
    def fails(*_args, **_kwargs):
        raise OSError("disk full")

    identity = service.tokens.verify(reader)
    monkeypatch.setattr(service.tokens, "record", fails)
    assert service.list_tables(identity)[0]["name"] == "notes"
