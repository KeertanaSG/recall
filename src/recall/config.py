import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def parse_allowed_users(value: str) -> frozenset[int] | None:
    """A blank value stays an error upstream, so opening the bot has to be deliberate."""
    if value.strip() == "*":
        return None
    return frozenset(int(part) for part in value.split(",") if part.strip())


@dataclass(frozen=True)
class Settings:
    telegram_token: str
    allowed_user_ids: frozenset[int] | None  # None means anyone can use the bot.
    data_dir: Path = Path("data")
    default_timezone: str = "Asia/Kolkata"
    model: str = "claude-haiku-5-5"
    effort: str = "low"
    retrieval_model: str = "claude-haiku-5-5"
    design_model: str = "claude-opus-5-5"
    max_reminders_per_day: int = 5
    max_messages_per_user_per_day: int = 30
    max_messages_per_day: int = 300
    prefetch: bool = True
    llm_provider: str = "claude"
    embedder: str = "local"
    storage: str = "sqlite"
    api_requests_per_minute: int = 60
    api_port: int = 8787
    mcp_port: int = 8788
    mcp_public_url: str = "http://127.0.0.1:8788"

    def allows(self, user_id: int) -> bool:
        return self.allowed_user_ids is None or user_id in self.allowed_user_ids

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        llm_provider = os.environ.get("RECALL_LLM_PROVIDER", "claude")
        required = ["TELEGRAM_BOT_TOKEN", "ALLOWED_USER_IDS"]
        if llm_provider == "claude":
            required.append("ANTHROPIC_API_KEY")
        missing = [name for name in required if not os.environ.get(name)]
        if missing:
            raise SystemExit(f"Missing required settings: {', '.join(missing)}. See .env.example.")
        try:
            return cls(
                telegram_token=os.environ["TELEGRAM_BOT_TOKEN"],
                allowed_user_ids=parse_allowed_users(os.environ["ALLOWED_USER_IDS"]),
                data_dir=Path(os.environ.get("RECALL_DATA_DIR", "data")),
                default_timezone=os.environ.get("RECALL_DEFAULT_TIMEZONE", "Asia/Kolkata"),
                model=os.environ.get("RECALL_MODEL", "claude-haiku-5-5"),
                design_model=os.environ.get("RECALL_DESIGN_MODEL", "claude-opus-5-5"),
                effort=os.environ.get("RECALL_EFFORT", "low"),
                retrieval_model=os.environ.get("RECALL_RETRIEVAL_MODEL", "claude-haiku-5-5"),
                max_reminders_per_day=int(os.environ.get("RECALL_MAX_REMINDERS_PER_DAY", "5")),
                max_messages_per_user_per_day=int(
                    os.environ.get("RECALL_MAX_MESSAGES_PER_USER_PER_DAY", "30")
                ),
                max_messages_per_day=int(os.environ.get("RECALL_MAX_MESSAGES_PER_DAY", "300")),
                llm_provider=llm_provider,
                embedder=os.environ.get("RECALL_EMBEDDER", "local"),
                storage=os.environ.get("RECALL_STORAGE", "sqlite"),
                prefetch=os.environ.get("RECALL_PREFETCH", "true").lower() == "true",
                api_requests_per_minute=int(os.environ.get("RECALL_API_REQUESTS_PER_MINUTE", "60")),
                api_port=int(os.environ.get("RECALL_API_PORT", "8787")),
                mcp_port=int(os.environ.get("RECALL_MCP_PORT", "8788")),
                mcp_public_url=os.environ.get("RECALL_MCP_PUBLIC_URL", "http://127.0.0.1:8788"),
            )
        except ValueError as exc:
            raise SystemExit(f"A numeric setting is invalid: {exc}. See .env.example.") from exc
