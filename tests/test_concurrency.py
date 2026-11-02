"""Real processes writing to one user's memory, as the bot, API and MCP server do."""

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

from conftest import FakeEmbedder

from recall.access import UsageLimiter
from recall.storage.store import Store

WRITERS = 3
ROWS = 25


def write(path: str, writer: int) -> list[str]:
    from pathlib import Path

    errors = []
    with Store(Path(path), FakeEmbedder(), "UTC") as store:
        for index in range(ROWS):
            try:
                store.add_row("notes", {"body": f"writer {writer} row {index}"})
                store.log_exchange(f"q{writer}", f"a{index}")
                store.read_query("SELECT COUNT(*) FROM notes")
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
    return errors


def spend(path: str, _writer: int) -> int:
    from pathlib import Path

    limiter = UsageLimiter(Path(path), 1000, 20)
    return sum(limiter.spend(1).value == "none" for _ in range(30))


def in_processes(function, path):
    with ProcessPoolExecutor(WRITERS, mp_context=get_context("spawn")) as pool:
        return list(pool.map(function, [str(path)] * WRITERS, range(WRITERS)))


def test_no_writes_are_lost_across_processes(tmp_path):
    path = tmp_path / "u.sqlite"
    with Store(path, FakeEmbedder(), "UTC") as store:
        store.create_table("notes", "Notes", [{"name": "body", "type": "text"}])
    assert in_processes(write, path) == [[]] * WRITERS
    with Store(path, FakeEmbedder(), "UTC") as store:
        bodies = [row["body"] for row in store.read_query("SELECT body FROM notes")["rows"]]
        assert len(bodies) == len(set(bodies)) == WRITERS * ROWS
        assert store.conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_daily_limit_holds_across_real_processes(tmp_path):
    assert sum(in_processes(spend, tmp_path / "usage.sqlite")) == 20
