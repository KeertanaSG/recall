import hashlib
import math
import re

import pytest

from recall.storage.embeddings import EMBEDDING_DIM, EmbeddingError
from recall.storage.store import Store


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
        [{"name": "item", "type": "text"}, {"name": "amount", "type": "real"}, {"name": "spent_on", "type": "date"}],
    )
    store.turn_id = None
    return "expenses"
