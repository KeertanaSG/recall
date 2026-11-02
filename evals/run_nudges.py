"""Score smart nudges on a labelled synthetic history, using real embeddings and the real model.

    uv run python evals/run_nudges.py

Precision matters most here, because every false alarm teaches someone to mute Recall.
"""

import argparse
import json
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import anthropic
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recall.config import Settings  # noqa: E402
from recall.core.nudges import Nudger, find_candidates  # noqa: E402
from recall.core.retrieval import result_key  # noqa: E402
from recall.llm.claude import ClaudeLLM  # noqa: E402
from recall.storage.embeddings import LocalEmbedder  # noqa: E402
from recall.storage.store import Store  # noqa: E402

RESULTS = Path(__file__).parent / "results" / "nudges.jsonl"
NOW = datetime(2026, 11, 2, 6, 30, tzinfo=timezone.utc)

TABLES = {
    "loans": ("Money lent to people", {"person": "text", "amount": "real", "repaid": "boolean"}),
    "tasks": ("Things to do", {"task": "text", "due_on": "date", "done": "boolean"}),
    "promises": ("Promises made to people", {"person": "text", "promise": "text", "kept": "boolean"}),
    "borrowed": ("Things borrowed", {"item": "text", "owner": "text", "returned": "boolean"}),
    "movies": ("Films watched", {"title": "text", "rating": "real", "is_rewatch": "boolean"}),
    "meals": ("Meals out", {"dish": "text", "place": "text", "rating": "real"}),
    "expenses": ("Money spent", {"item": "text", "amount": "real"}),
    "notes": ("General notes", {"body": "text"}),
}

# (table, values, days ago, should Recall nudge?)
ROWS = [
    ("loans", {"person": "Arjun", "amount": 2000, "repaid": False}, 30, True),
    ("loans", {"person": "Sam", "amount": 500, "repaid": True}, 40, False),
    ("loans", {"person": "Priya", "amount": 1200, "repaid": False}, 25, False),
    ("loans", {"person": "Neha", "amount": 800, "repaid": False}, 15, True),
    ("tasks", {"task": "Send Maya the holiday photos", "done": False}, 14, True),
    ("tasks", {"task": "Renew car insurance", "due_on": "2026-10-20", "done": False}, 25, True),
    ("tasks", {"task": "Book a dentist appointment", "done": True}, 20, False),
    ("promises", {"person": "Dad", "promise": "fix his printer", "kept": False}, 20, True),
    ("promises", {"person": "Riya", "promise": "review her resume", "kept": False}, 18, False),
    ("borrowed", {"item": "Circe, the novel", "owner": "Leela", "returned": False}, 40, True),
    ("borrowed", {"item": "power drill", "owner": "the neighbour", "returned": True}, 35, False),
    ("movies", {"title": "Dune 2", "rating": 9, "is_rewatch": False}, 12, False),
    ("movies", {"title": "Past Lives", "rating": 8, "is_rewatch": False}, 22, False),
    ("movies", {"title": "Heat", "rating": 9, "is_rewatch": True}, 33, False),
    ("meals", {"dish": "Masala dosa", "place": "MTR", "rating": 9}, 9, False),
    ("meals", {"dish": "Ramen", "place": "Noodle Bar", "rating": 7}, 17, False),
    ("expenses", {"item": "Cab to airport", "amount": 450}, 11, False),
    ("expenses", {"item": "Groceries", "amount": 1800}, 6, False),
    ("notes", {"body": "Waiting to hear back from the landlord about the deposit"}, 10, True),
    ("notes", {"body": "Would be nice to learn pottery someday"}, 28, False),
    ("notes", {"body": "Lena's birthday is March 3"}, 50, False),
    ("notes", {"body": "Feeling low since the breakup, need to talk to my therapist"}, 8, False),
    ("notes", {"body": "Priya paid back the 1200 she owed me"}, 5, False),
    ("notes", {"body": "Reviewed Riya's resume and sent it to her"}, 4, False),
]


def build(store: Store) -> set[str]:
    for name, (description, columns) in TABLES.items():
        store.create_table(
            name, description, [{"name": c, "type": kind} for c, kind in columns.items()]
        )
    expected = set()
    for table, values, days_ago, should_nudge in ROWS:
        row_id = store.add_row(table, values)
        saved = (NOW - timedelta(days=days_ago)).isoformat(timespec="seconds")
        store.conn.execute(f"UPDATE {table} SET created_at = ? WHERE id = ?", (saved, row_id))
        if should_nudge:
            expected.add(f"{table}#{row_id}")
    return expected


def score(chosen: set[str], expected: set[str]) -> dict:
    hits = len(chosen & expected)
    return {
        "precision": round(hits / len(chosen), 3) if chosen else None,
        "recall": round(hits / len(expected), 3),
        "false_alarms": sorted(chosen - expected),
        "missed": sorted(expected - chosen),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    load_dotenv()
    settings = Settings.from_env()
    embedder = LocalEmbedder(settings.data_dir / "models")
    llm = ClaudeLLM(anthropic.Anthropic())
    nudger = Nudger(llm, settings.retrieval_model, max_nudges=len(ROWS))

    with tempfile.TemporaryDirectory() as folder, Store(
        Path(folder) / "eval.sqlite", embedder, "Asia/Kolkata"
    ) as store:
        expected = build(store)
        candidates = find_candidates(store, NOW)
        keys = {result_key(c) for c in candidates}
        print(f"Detected {len(candidates)} of {len(ROWS)} rows, covering "
              f"{len(keys & expected)}/{len(expected)} real open items, with no model call")
        print("Detector missed:", sorted(expected - keys) or "nothing")
        runs = []
        for run in range(args.runs):
            started = time.monotonic()
            picks = nudger.judge(store, candidates, NOW)
            elapsed = round(time.monotonic() - started, 2)
            result = score({p["key"] for p in picks}, expected) | {"seconds": elapsed}
            runs.append(result)
            print(f"\nRun {run + 1}: {result}")
            for pick in picks:
                print(f"  {pick['importance']} {pick['key']}: {pick['message']}")

    RESULTS.parent.mkdir(exist_ok=True)
    with RESULTS.open("a") as file:
        record = {
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model": settings.retrieval_model,
            "rows": len(ROWS),
            "open_items": len(expected),
            "candidates": len(candidates),
            "detector_recall": round(len(keys & expected) / len(expected), 3),
            "runs": runs,
        }
        file.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
