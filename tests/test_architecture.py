"""The module boundaries, checked so they can't quietly erode, and proof that swapping works."""

import ast
from datetime import UTC, datetime
from pathlib import Path

import pytest

import recall.factory as factory
from recall.config import Settings
from recall.core.agent import Agent, TurnFailed
from recall.core.nudges import Nudger
from recall.llm.base import Completion, LLMError, Stop, ToolCall

SOURCE = Path(__file__).parents[1] / "src" / "recall"
# Only these may touch a vendor. Everything else goes through the interfaces.
FORBIDDEN = {
    "core": {
        "anthropic",
        "telegram",
        "fastapi",
        "mcp",
        "sqlite3",
        "sqlite_vec",
        "recall.channels",
        "recall.llm.claude",
        "recall.storage.store",
        "recall.factory",
    },
    "service.py": {
        "anthropic",
        "telegram",
        "fastapi",
        "mcp",
        "sqlite3",
        "sqlite_vec",
        "recall.channels",
        "recall.llm.claude",
        "recall.storage.store",
        "recall.factory",
    },
    "llm/base.py": {"anthropic"},
    "storage/base.py": {"sqlite3", "sqlite_vec"},
}


def imports(path: Path) -> set[str]:
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


@pytest.mark.parametrize("area", FORBIDDEN)
def test_boundaries_hold(area):
    target = SOURCE / area
    files = sorted(target.glob("*.py")) if target.is_dir() else [target]
    assert files
    for path in files:
        for name in imports(path):
            blocked = [ban for ban in FORBIDDEN[area] if name == ban or name.startswith(f"{ban}.")]
            assert not blocked, f"{path.name} imports {name}"


class OtherLLM:
    """A provider with its own message shape, to prove nothing in core depends on Claude's."""

    def __init__(self, *replies, answers=None):
        self.replies = list(replies)
        self.answers = answers or {}
        self.seen = []

    def chat(self, model, system, messages, tools, effort, note=""):
        self.seen.append(list(messages))
        return self.replies.pop(0)

    def ask_json(self, model, system, prompt, schema):
        if isinstance(self.answers, Exception):
            raise self.answers
        return self.answers

    def user_message(self, text, image=None):
        return {"role": "user", "content": {"words": text, "picture": image}}

    def assistant_message(self, completion):
        return {"role": "assistant", "content": {"words": completion.text}}

    def tool_results_message(self, results):
        return {"role": "user", "content": {"outputs": [r.content for r in results]}}


def save(table, **values):
    call = ToolCall("1", "add_row", {"table": table, "values": values})
    return Completion("", Stop.TOOL_USE, [call])


def test_the_agent_runs_on_another_provider(store, expenses):
    llm = OtherLLM(save("expenses", item="tea", amount=40), Completion("Tea, 40.", Stop.DONE))
    reply = Agent(llm, "other-model", "low").respond(store, "tea 40", image=b"x")
    assert reply.text == "Tea, 40."
    assert store.get_row("expenses", 1)["item"] == "tea"
    assert llm.seen[1][-2] == {"role": "assistant", "content": {"words": ""}}
    assert "expenses" in llm.seen[1][-1]["content"]["outputs"][0]


@pytest.mark.parametrize("stop", [Stop.REFUSED, Stop.CUT_OFF])
def test_stopping_early_undoes_the_turn_on_any_provider(store, expenses, stop):
    llm = OtherLLM(save("expenses", item="tea", amount=40), Completion("", stop))
    with pytest.raises(TurnFailed):
        Agent(llm, "m", "low").respond(store, "tea 40")
    assert store.get_row("expenses", 1) is None


def test_provider_errors_fail_the_turn_cleanly(store, expenses):
    class Down(OtherLLM):
        def chat(self, *args, **kwargs):
            raise LLMError("down")

    with pytest.raises(TurnFailed, match="down"):
        Agent(Down(), "m", "low").respond(store, "tea 40")


def test_nudges_survive_a_provider_failure(store):
    candidate = {
        "table": "loans",
        "row": {"id": 1, "person": "A", "created_at": "2026-10-01T00:00:00+00:00"},
    }
    nudger = Nudger(OtherLLM(answers=LLMError("down")), "m")
    assert nudger.judge(store, [candidate], datetime(2026, 11, 1, tzinfo=UTC)) is None


def test_a_new_provider_only_needs_registering(tmp_path, monkeypatch):
    llm = OtherLLM()
    monkeypatch.setitem(factory.LLMS, "other", lambda settings: llm)
    settings = Settings("t", frozenset({1}), tmp_path, llm_provider="other")
    service = factory.build_service(settings)
    assert service.agent.llm is llm and service.nudger.llm is llm


@pytest.mark.parametrize(
    ("field", "kind"),
    [("llm_provider", "LLM provider"), ("embedder", "embedder"), ("storage", "storage backend")],
)
def test_unknown_choices_name_what_is_available(tmp_path, field, kind):
    settings = Settings("t", frozenset({1}), tmp_path, **{field: "nope"})
    with pytest.raises(SystemExit, match=f"Unknown {kind} 'nope'. Available"):
        factory.build_service(settings)


def test_settings_need_no_claude_key_for_another_provider(monkeypatch):
    monkeypatch.setattr("recall.config.load_dotenv", lambda: None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
    monkeypatch.setenv("ALLOWED_USER_IDS", "1")
    monkeypatch.setenv("RECALL_LLM_PROVIDER", "other")
    assert Settings.from_env().llm_provider == "other"
