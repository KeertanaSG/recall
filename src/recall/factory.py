"""The one place that chooses implementations. To add a provider, register a builder here."""

from collections.abc import Callable

from recall.access import TokenStore, UsageLimiter
from recall.config import Settings
from recall.core.agent import Agent
from recall.core.nudges import Nudger
from recall.llm.base import LLM
from recall.service import RecallService
from recall.storage.base import Embedder, StoreProvider


def _claude(settings: Settings) -> LLM:
    import anthropic

    from recall.llm.claude import ClaudeLLM

    return ClaudeLLM(anthropic.Anthropic())


def _local_embedder(settings: Settings) -> Embedder:
    from recall.storage.store import LocalEmbedder

    return LocalEmbedder(settings.data_dir / "models")


def _sqlite(settings: Settings, embedder: Embedder) -> StoreProvider:
    from recall.storage.store import SqliteStoreProvider

    return SqliteStoreProvider(settings, embedder)


LLMS: dict[str, Callable[[Settings], LLM]] = {"claude": _claude}
EMBEDDERS: dict[str, Callable[[Settings], Embedder]] = {"local": _local_embedder}
STORES: dict[str, Callable[[Settings, Embedder], StoreProvider]] = {"sqlite": _sqlite}


def _choose(kind: str, name: str, options: dict) -> Callable:
    if name not in options:
        raise SystemExit(f"Unknown {kind} {name!r}. Available: {', '.join(sorted(options))}.")
    return options[name]


def build_stores(settings: Settings) -> StoreProvider:
    embedder = _choose("embedder", settings.embedder, EMBEDDERS)(settings)
    return _choose("storage backend", settings.storage, STORES)(settings, embedder)


def build_service(settings: Settings) -> RecallService:
    llm = _choose("LLM provider", settings.llm_provider, LLMS)(settings)
    return RecallService(
        settings,
        TokenStore(settings.data_dir / "auth.sqlite"),
        build_stores(settings),
        Agent(
            llm,
            settings.model,
            settings.effort,
            settings.retrieval_model,
            settings.design_model,
            settings.prefetch,
        ),
        UsageLimiter(
            settings.data_dir / "usage.sqlite",
            settings.max_messages_per_user_per_day,
            settings.max_messages_per_day,
        ),
        Nudger(llm, settings.retrieval_model),
    )
