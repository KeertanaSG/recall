import logging
from logging.handlers import RotatingFileHandler

from recall.channels.telegram import build_application
from recall.config import Settings


LOG_BYTES = 5_000_000
LOG_BACKUPS = 3


def main() -> None:
    settings = Settings.from_env()
    logs = settings.data_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True, mode=0o700)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            RotatingFileHandler(logs / "bot.log", maxBytes=LOG_BYTES, backupCount=LOG_BACKUPS),
        ],
    )
    # httpx logs request URLs at INFO, and Telegram URLs contain the bot token.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    build_application(settings).run_polling()


if __name__ == "__main__":
    main()
