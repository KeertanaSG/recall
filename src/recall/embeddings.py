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
        self._model = TextEmbedding(EMBEDDING_MODEL, cache_dir=str(cache_dir))

    def embed(self, texts: list[str], input_type: InputType) -> list[list[float]]:
        embed = self._model.query_embed if input_type == "query" else self._model.passage_embed
        try:
            return [vector.tolist() for vector in embed(texts)]
        except Exception as exc:
            raise EmbeddingError(str(exc)) from exc
