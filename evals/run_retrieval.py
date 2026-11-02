"""Compare retrieval on a large history: plain vector search against the LangGraph pipeline.

    uv run python evals/run_retrieval.py                # the search graph alone
    uv run python evals/run_retrieval.py --agent        # the full agent loop instead
    uv run python evals/run_retrieval.py --vector-only  # no model calls
    uv run python evals/run_retrieval.py --label "new grader prompt"
    uv run python evals/compare.py                      # compare saved runs

A found question passes if its needle comes back. An absent question passes if nothing does.
Every configuration adds a line to results/runs.jsonl and writes its per-question detail
to results/<run id>.json.
"""

import argparse
import json
import statistics
import sys
import tempfile
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import litellm
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))

from history import ABSENT, NEEDLES, build_history  # noqa: E402

from recall.config import Settings  # noqa: E402
from recall.core.agent import Agent, TurnFailed  # noqa: E402
from recall.core.retrieval import Retriever  # noqa: E402
from recall.llm.litellm_provider import LiteLLMProvider  # noqa: E402
from recall.storage.store import (
    LocalEmbedder,  # noqa: E402
    Store,  # noqa: E402
)

EVALS = Path(__file__).parent
RESULTS = EVALS / "results"
MODEL = "claude-opus-5-5"
WORKERS = 6
VECTOR_TOP_K = 8
TURN_FAILED = "TURN FAILED"


class TokenCounter:
    """Tracks tokens and model time, split into the main agent loop and the graph's steps."""

    def __init__(self):
        self.lock = threading.Lock()
        self._reset()

    def complete(self, **kwargs):
        step = "graph" if "response_format" in kwargs else "main"
        start = time.time()
        response = litellm.completion(**kwargs)
        elapsed = time.time() - start
        with self.lock:
            self.input += response.usage.prompt_tokens
            self.output += response.usage.completion_tokens
            self.calls[step] += 1
            self.seconds[step] += elapsed
        return response

    def _reset(self) -> None:
        self.input = 0
        self.output = 0
        self.calls = {"main": 0, "graph": 0}
        self.seconds = {"main": 0.0, "graph": 0.0}

    def take(self) -> dict | None:
        with self.lock:
            if not any(self.calls.values()):
                return None
            totals = {
                "input": self.input,
                "output": self.output,
                "calls": self.calls,
                "model_seconds": {step: round(value, 1) for step, value in self.seconds.items()},
            }
            self._reset()
        return totals


def cases():
    yield from ((needle.subject, needle.question, needle.key) for needle in NEEDLES)
    yield from (("absent", question, None) for question in ABSENT)


def run_all(path, embedder, ask):
    local = threading.local()

    def run_case(case):
        subject, question, key = case
        if not hasattr(local, "store"):
            local.store = Store(path, embedder, "Asia/Kolkata")
        start = time.time()
        keys, graph_used, answer = ask(local.store, question)
        seconds = time.time() - start
        print(f"  {seconds:.1f}s {question}", flush=True)
        return {
            "subject": subject,
            "question": question,
            "needle": key,
            "returned": sorted(f"{table}#{row_id}" for table, row_id in keys),
            "answer": answer,
            "seconds": round(seconds, 2),
            "graph_used": graph_used,
        }

    with ThreadPoolExecutor(WORKERS) as pool:
        return list(pool.map(run_case, cases()))


def summarize(name, label, outcomes, locations, rows, tokens):
    subjects = defaultdict(list)
    for outcome in outcomes:
        key = outcome["needle"]
        if key:
            table, row_id = locations[key]
            outcome["passed"] = f"{table}#{row_id}" in outcome["returned"]
        else:
            outcome["passed"] = not outcome["returned"]
        if outcome["answer"].startswith(TURN_FAILED):
            outcome["passed"] = False
        subjects[outcome["subject"]].append(outcome)

    answerable = [o for o in outcomes if o["needle"]]
    absent = [o for o in outcomes if not o["needle"]]
    times = [o["seconds"] for o in outcomes]
    slowest = max(outcomes, key=lambda o: o["seconds"])
    used = [o["graph_used"] for o in outcomes if o["graph_used"] is not None]
    created = datetime.now(UTC)
    return {
        "run_id": f"{created:%Y%m%d-%H%M%S}-{name}",
        "created_at": created.isoformat(timespec="seconds"),
        "config": name,
        "label": label,
        "rows": rows,
        "answerable_passed": sum(o["passed"] for o in answerable),
        "answerable_total": len(answerable),
        "absent_passed": sum(o["passed"] for o in absent),
        "absent_total": len(absent),
        "absent_mean_s": round(statistics.mean(o["seconds"] for o in absent), 2),
        "absent_max_s": max(o["seconds"] for o in absent),
        "median_s": round(statistics.median(times), 2),
        "mean_s": round(statistics.mean(times), 2),
        "sum_s": round(sum(times), 1),
        "slowest_s": slowest["seconds"],
        "slowest_question": slowest["question"],
        "graph_used": sum(used) if used else None,
        "tokens": tokens,
        "subjects": {
            subject: {
                "passed": sum(o["passed"] for o in results),
                "total": len(results),
                "mean_s": round(statistics.mean(o["seconds"] for o in results), 2),
                "max_s": max(o["seconds"] for o in results),
            }
            for subject, results in subjects.items()
        },
    }


def report(summary, outcomes):
    print(f"\n=== {summary['config']} ===")
    print(f"{'subject':10} {'passed':>8} {'mean s':>8} {'max s':>8}")
    for subject, stats in summary["subjects"].items():
        passed = f"{stats['passed']}/{stats['total']}"
        print(f"{subject:10} {passed:>8} {stats['mean_s']:>8.1f} {stats['max_s']:>8.1f}")
    print(
        f"answerable {summary['answerable_passed']}/{summary['answerable_total']} | "
        f"absent {summary['absent_passed']}/{summary['absent_total']}"
    )
    print(
        f"time: median {summary['median_s']}s | mean {summary['mean_s']}s | sum {summary['sum_s']}s | "
        f"slowest {summary['slowest_s']}s ({summary['slowest_question']})"
    )
    if summary["graph_used"] is not None:
        print(f"graph used on {summary['graph_used']}/{len(outcomes)} questions")
    if summary["tokens"]:
        tokens = summary["tokens"]
        print(f"tokens: {tokens['input']} in, {tokens['output']} out")
        for step, calls in tokens["calls"].items():
            if calls:
                seconds = tokens["model_seconds"][step]
                print(
                    f"{step}: {calls} calls, {seconds / calls:.1f}s each, "
                    f"{seconds / len(outcomes):.1f}s per question"
                )
    failures = [o for o in outcomes if not o["passed"]]
    if failures:
        print("  failures:")
        for outcome in failures:
            print(
                f"    {outcome['subject']}: {outcome['question']} -> returned {len(outcome['returned'])} rows"
            )
            if outcome["answer"]:
                print(f"      answer: {outcome['answer']}")


def save(summary, outcomes):
    RESULTS.mkdir(exist_ok=True)
    with (RESULTS / "runs.jsonl").open("a") as runs:
        runs.write(json.dumps(summary) + "\n")
    detail = {"summary": summary, "questions": outcomes}
    (RESULTS / f"{summary['run_id']}.json").write_text(json.dumps(detail, indent=2))


def vector_ask(store, question):
    results = store.search(question, VECTOR_TOP_K)
    return {(r["table"], r["row"]["id"]) for r in results}, None, ""


def graph_ask(retriever):
    def ask(store, question):
        result = retriever.search(store, question)
        return {(r["table"], r["row"]["id"]) for r in result["results"]}, True, ""

    return ask


def agent_ask(agent):
    uses = threading.local()
    original = agent.retriever.search

    def counted(*args, **kwargs):
        uses.graph = True
        return original(*args, **kwargs)

    agent.retriever.search = counted

    def ask(store, question):
        uses.graph = False
        store.log_exchange = lambda *_: (
            None
        )  # Questions run in parallel and must not share context.
        try:
            reply = agent.respond(store, question)
        except TurnFailed as exc:
            return set(), uses.graph, f"{TURN_FAILED}: {exc}"
        return {(s.table, s.id) for s in reply.sources}, uses.graph, reply.text

    return ask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", action="store_true")
    parser.add_argument("--vector-only", action="store_true")
    parser.add_argument("--label", default="", help="What changed in this run")
    parser.add_argument("--model", default=Settings.model, help="Model for the main agent loop")
    parser.add_argument(
        "--retrieval-model",
        default=Settings.retrieval_model,
        help="Model for the graph's plan and grade steps",
    )
    parser.add_argument(
        "--prefetch", action="store_true", help="Search locally before the first model call"
    )
    args = parser.parse_args()
    load_dotenv()

    embedder = LocalEmbedder(EVALS / ".cache" / "models")
    tokens = TokenCounter()
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "history.sqlite"
        start = time.time()
        with Store(path, embedder, "Asia/Kolkata") as store:
            locations = build_history(store)
            rows = sum(t["row_count"] for t in store.tables())
        print(
            f"Built {rows} rows in {time.time() - start:.0f}s. "
            f"{len(NEEDLES)} answerable, {len(ABSENT)} absent."
        )

        suffix = "".join(
            [
                "" if args.model == MODEL else f"-main-{args.model.split('-')[1]}",
                "" if args.retrieval_model == MODEL else f"-{args.retrieval_model.split('-')[1]}",
                "-prefetch" if args.prefetch else "",
            ]
        )
        configs = []
        llm = LiteLLMProvider(tokens.complete)
        if args.agent:
            agent = Agent(llm, args.model, "low", prefetch=args.prefetch)
            agent.retriever = Retriever(llm, args.retrieval_model)
            configs.append((f"agent{suffix}", agent_ask(agent)))
        else:
            configs.append((f"vector-top{VECTOR_TOP_K}", vector_ask))
            if not args.vector_only:
                retriever = Retriever(llm, args.retrieval_model)
                configs.append((f"graph{suffix}", graph_ask(retriever)))

        for name, ask in configs:
            outcomes = run_all(path, embedder, ask)
            summary = summarize(name, args.label, outcomes, locations, rows, tokens.take())
            report(summary, outcomes)
            save(summary, outcomes)


if __name__ == "__main__":
    main()
