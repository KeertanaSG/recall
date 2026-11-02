from recall.channels.telegram import build_application
from recall.config import Settings
from recall.logs import setup


def main() -> None:
    settings = Settings.from_env()
    setup(settings.data_dir, "bot")
    build_application(settings).run_polling()


if __name__ == "__main__":
    main()
