import threading
from pathlib import Path
from typing import Literal, Protocol

from fastembed import TextEmbedding

EMBEDDING_DIM = 384
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

InputType = Literal["document", "query"]


class EmbeddingError(Exception):
    pass


class Embedder(Protocol):
    def embed(self, texts: list[str], input_type: InputType) -> list[list[float]]: ...


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
