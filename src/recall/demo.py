"""Fill one user's memory with realistic, backdated entries, for demos and manual testing.

    uv run recall-demo                  # into an empty memory
    uv run recall-demo --fresh          # back up and replace what's there
    uv run recall-demo --user 123       # needed when several users are allowed

Rows are written directly, so seeding makes no model calls. Stop the bot first. Some items are
open and some are already resolved, so nudges have real decisions to make.
"""

import argparse
import sqlite3
from datetime import datetime, timedelta, timezone

from recall.config import Settings
from recall.factory import build_stores
from recall.storage.base import MemoryStore
from recall.storage.store import user_db_path

TABLES = {
    "movies": ("Films watched", {"title": "text", "rating": "real", "genre": "text", "companions": "text", "watched_on": "date"}),
    "meals": ("Meals out and what was good", {"dish": "text", "place": "text", "rating": "real", "companions": "text", "eaten_on": "date"}),
    "expenses": ("Money spent", {"item": "text", "amount": "real", "category": "text", "spent_on": "date"}),
    "loans": ("Money lent to people", {"person": "text", "amount": "real", "reason": "text", "repaid": "boolean"}),
    "promises": ("Promises made to people", {"person": "text", "promise": "text", "kept": "boolean"}),
    "tasks": ("Things to do", {"task": "text", "due_on": "date", "done": "boolean"}),
    "borrowed": ("Things borrowed from people", {"item": "text", "owner": "text", "returned": "boolean"}),
    "gifts": ("Gift ideas for people", {"person": "text", "idea": "text", "occasion": "text"}),
    "people": ("Facts about people", {"person": "text", "fact": "text"}),
    "patterns": ("What helps in a given mood or situation", {"situation": "text", "what_helps": "text"}),
    "notes": ("Anything else worth keeping", {"body": "text"}),
}


def rows(today) -> list[tuple]:
    """(table, values, days ago, what the user said)"""

    def day(days_ago: int) -> str:
        return (today - timedelta(days=days_ago)).isoformat()

    return [
        ("movies", {"title": "Dune: Part Two", "rating": 9, "genre": "sci-fi", "companions": "Arjun", "watched_on": day(42)}, 42, "Watched Dune 2 with Arjun, 9/10"),
        ("movies", {"title": "Past Lives", "rating": 8, "genre": "romance", "companions": "Priya", "watched_on": day(30)}, 30, "Past Lives with Priya, 8/10, made me cry"),
        ("movies", {"title": "Oppenheimer", "rating": 8.5, "genre": "drama", "watched_on": day(21)}, 21, "Oppenheimer, 8.5"),
        ("movies", {"title": "Spirited Away", "rating": 10, "genre": "animation", "companions": "Maya", "watched_on": day(9)}, 9, "Rewatched Spirited Away with Maya, still a 10"),
        ("meals", {"dish": "Masala dosa", "place": "MTR", "rating": 9, "companions": "Dad", "eaten_on": day(35)}, 35, "Masala dosa at MTR with dad, 9/10"),
        ("meals", {"dish": "Tonkotsu ramen", "place": "Noodle Bar", "rating": 7, "eaten_on": day(19)}, 19, "Ramen at Noodle Bar, decent, 7"),
        ("meals", {"dish": "Filter coffee", "place": "Cafe Malabar", "rating": 8, "companions": "Leela", "eaten_on": day(6)}, 6, "Great filter coffee at Cafe Malabar with Leela"),
        ("expenses", {"item": "Cab to airport", "amount": 450, "category": "travel", "spent_on": day(26)}, 26, "Cab to airport 450"),
        ("expenses", {"item": "Groceries", "amount": 1800, "category": "food", "spent_on": day(16)}, 16, "Groceries 1800"),
        ("expenses", {"item": "Lunch with Priya", "amount": 600, "category": "food", "spent_on": day(11)}, 11, "Lunch with Priya 600"),
        ("expenses", {"item": "Cab to office", "amount": 220, "category": "travel", "spent_on": day(5)}, 5, "Cab to office 220"),
        ("expenses", {"item": "Dinner at MTR", "amount": 950, "category": "food", "spent_on": day(4)}, 4, "Dinner at MTR 950"),
        ("expenses", {"item": "Electricity bill", "amount": 1450, "category": "bills", "spent_on": day(3)}, 3, "Paid electricity bill 1450"),
        ("loans", {"person": "Arjun", "amount": 2000, "reason": "concert tickets", "repaid": False}, 34, "Lent Arjun 2000 for the concert tickets"),
        ("loans", {"person": "Neha", "amount": 800, "reason": "dinner", "repaid": False}, 17, "Paid for Neha's dinner, she owes me 800"),
        ("loans", {"person": "Priya", "amount": 1200, "reason": "train tickets", "repaid": False}, 25, "Booked Priya's train, she owes me 1200"),
        ("loans", {"person": "Sam", "amount": 500, "reason": "cab", "repaid": True}, 40, "Sam borrowed 500 for a cab"),
        ("notes", {"body": "Priya paid back the 1200 for the train tickets"}, 6, "Priya paid me back the 1200"),
        ("promises", {"person": "Maya", "promise": "send the trip photos"}, 14, "Told Maya I'd send the trip photos"),
        ("promises", {"person": "Dad", "promise": "fix his printer"}, 20, "Promised dad I'd fix his printer"),
        ("promises", {"person": "Riya", "promise": "review her resume"}, 18, "Told Riya I'd review her resume"),
        ("notes", {"body": "Reviewed Riya's resume and sent it back"}, 4, "Reviewed Riya's resume and sent it back"),
        ("tasks", {"task": "Renew car insurance", "due_on": day(10), "done": False}, 24, "Need to renew car insurance by the end of the month"),
        ("tasks", {"task": "Book a dentist appointment", "done": True}, 22, "Book dentist"),
        ("borrowed", {"item": "Circe, the novel", "owner": "Leela", "returned": False}, 38, "Borrowed Circe from Leela"),
        ("borrowed", {"item": "Power drill", "owner": "the neighbour", "returned": True}, 33, "Borrowed the neighbour's drill"),
        ("gifts", {"person": "Maya", "idea": "pottery class", "occasion": "birthday"}, 28, "Maya would love a pottery class for her birthday"),
        ("gifts", {"person": "Dad", "idea": "noise-cancelling headphones"}, 12, "Dad keeps mentioning noise-cancelling headphones, good gift"),
        ("people", {"person": "Arjun", "fact": "allergic to peanuts"}, 45, "Arjun is allergic to peanuts"),
        ("people", {"person": "Leela", "fact": "birthday is on March 3"}, 50, "Leela's birthday is March 3"),
        ("people", {"person": "Maya", "fact": "moved to Pune in August"}, 60, "Maya moved to Pune in August"),
        ("patterns", {"situation": "anxious", "what_helps": "knitting"}, 31, "When I'm anxious, knitting helps"),
        ("patterns", {"situation": "afternoon slump", "what_helps": "ginger tea"}, 15, "Ginger tea fixes my afternoons"),
        ("notes", {"body": "Waiting to hear back from the landlord about the deposit"}, 10, "Still waiting on the landlord about my deposit"),
        ("notes", {"body": "The wifi at the Goa place was terrible, avoid it next time"}, 27, "Goa place wifi was terrible, avoid next time"),
        ("notes", {"body": "Would be nice to learn pottery someday"}, 29, "Would be nice to learn pottery someday"),
    ]


def seed(store: MemoryStore, now: datetime) -> int:
    for name, (description, columns) in TABLES.items():
        store.create_table(
            name,
            description,
            [{"name": c, "type": kind} for c, kind in columns.items()],
            confirmed_distinct=True,
        )
    entries = rows(now.date())
    for table, values, days_ago, said in entries:
        row_id = store.add_row(table, values, said)
        saved = (now - timedelta(days=days_ago)).isoformat(timespec="seconds")
        store.conn.execute(
            f"UPDATE {table} SET created_at = ?, updated_at = ? WHERE id = ?", (saved, saved, row_id)
        )
    # Otherwise /undo would treat the whole seed as one turn and erase it.
    store.conn.execute("DELETE FROM _history")
    store.conn.execute("DELETE FROM _turns")
    store.turn_id = None
    return len(entries)


def back_up(settings: Settings, user_id: int, now: datetime) -> str:
    source = user_db_path(settings.data_dir, user_id)
    folder = settings.data_dir / "backups"
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    target = folder / f"before-demo-{now:%Y%m%d-%H%M%S}.sqlite"
    with sqlite3.connect(source) as live, sqlite3.connect(target) as copy:
        live.backup(copy)
    target.chmod(0o600)
    return str(target)


def main() -> None:
    parser = argparse.ArgumentParser(prog="recall-demo")
    parser.add_argument("--user", type=int)
    parser.add_argument("--fresh", action="store_true", help="Back up and replace existing data")
    args = parser.parse_args()
    settings = Settings.from_env()
    allowed = settings.allowed_user_ids or frozenset()
    if args.user is None and len(allowed) != 1:
        raise SystemExit("Pass --user with the Telegram ID to fill.")
    user_id = args.user if args.user is not None else next(iter(allowed))
    if not settings.allows(user_id):
        raise SystemExit(f"User {user_id} isn't in ALLOWED_USER_IDS.")

    now = datetime.now(timezone.utc)
    stores = build_stores(settings)
    if user_id in stores.user_ids():
        with stores.open(user_id) as store:
            has_data = bool(store.tables())
        if has_data and not args.fresh:
            raise SystemExit("This memory isn't empty. Use --fresh to back it up and replace it.")
        print(f"Backed up to {back_up(settings, user_id, now)}")
        stores.delete(user_id)
    with stores.open(user_id) as store:
        count = seed(store, now)
    print(f"Added {count} entries across {len(TABLES)} tables.")


if __name__ == "__main__":
    main()
