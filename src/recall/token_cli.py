"""Manage API tokens.

    recall-tokens create --user 123 --name claude-code --scopes read,write --days 30
    recall-tokens list
    recall-tokens revoke 4
"""

import argparse

from recall.auth import READ, TokenStore
from recall.config import Settings


def main() -> None:
    parser = argparse.ArgumentParser(prog="recall-tokens")
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--user", type=int, required=True)
    create.add_argument("--name", required=True)
    create.add_argument("--scopes", default=READ, help="read, or read,write")
    create.add_argument("--days", type=int, default=30)
    commands.add_parser("list")
    revoke = commands.add_parser("revoke")
    revoke.add_argument("id", type=int)
    args = parser.parse_args()

    settings = Settings.from_env()
    tokens = TokenStore(settings.data_dir / "auth.sqlite")
    if args.command == "create":
        if not settings.allows(args.user):
            raise SystemExit(f"User {args.user} isn't in ALLOWED_USER_IDS.")
        token = tokens.create(args.user, args.name, set(args.scopes.split(",")), args.days)
        print(f"{token}\nThis is the only time it's shown. Keep it somewhere safe.")
    elif args.command == "list":
        for row in tokens.list_tokens():
            state = "revoked" if row["revoked_at"] else f"expires {row['expires_at'][:10]}"
            print(f"{row['id']}  user {row['user_id']}  {row['name']}  {row['scopes']}  {state}")
    else:
        print("Revoked." if tokens.revoke(args.id) else "No active token with that ID.")


if __name__ == "__main__":
    main()
