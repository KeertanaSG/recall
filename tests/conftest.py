import hashlib
import json
import math
import re
from types import SimpleNamespace as NS

import pytest

from recall.llm.litellm_provider import LiteLLMProvider
from recall.storage.base import EMBEDDING_DIM, EmbeddingError
from recall.storage.store import Store


def completion(text="", calls=(), finish=None):
    """A LiteLLM response. `calls` holds (name, input) pairs."""
    tool_calls = [
        NS(id=f"call_{name}", function=NS(name=name, arguments=json.dumps(arguments)))
        for name, arguments in calls
    ]
    message = NS(
        content=text,
        tool_calls=tool_calls or None,
        model_dump=lambda exclude_none: {"role": "assistant", "content": text},
    )
    finish = finish or ("tool_calls" if calls else "stop")
    usage = NS(prompt_tokens=1, completion_tokens=1)
    return NS(choices=[NS(message=message, finish_reason=finish)], usage=usage)


def fake_llm(client):
    return LiteLLMProvider(client.complete)


class FakeEmbedder:
    """Bag-of-words hashing, so shared words give similar vectors with no network."""

    def __init__(self):
        self.failing = False

    def embed(self, texts, input_type):
        if self.failing:
            raise EmbeddingError("down")
        return [self._vector(text) for text in texts]

    @staticmethod
    def _vector(text):
        vector = [0.0] * EMBEDDING_DIM
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.sha256(word.encode()).digest()
            vector[int.from_bytes(digest[:4], "big") % EMBEDDING_DIM] += 1.0
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def store(tmp_path, embedder):
    with Store(tmp_path / "users" / "1.sqlite", embedder, "Asia/Kolkata") as store:
        yield store


@pytest.fixture
def expenses(store):
    store.create_table(
        "expenses",
        "Money the user spent",
        [
            {"name": "item", "type": "text"},
            {"name": "amount", "type": "real"},
            {"name": "spent_on", "type": "date"},
        ],
    )
    store.turn_id = None
    return "expenses"
