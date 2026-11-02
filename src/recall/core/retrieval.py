"""The RAG pipeline behind the agent's search tool, built as a LangGraph graph.

plan -> retrieve -> grade

Planning widens the question into search phrases and keywords. Retrieval merges
semantic and keyword matches, so names and exact words aren't lost. Grading drops
rows that don't answer the question, and that step is what lets Recall say it found
nothing. If a model step fails, search still works, just with less filtering.
"""

import json
import re
from typing import TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from recall.llm.base import LLM, LLMError
from recall.logs import get_logger, measured
from recall.prompts import render
from recall.storage.base import MemoryStore, ToolError

log = get_logger(__name__)

MAX_CANDIDATES = 15
FUSION_K = 60  # The standard reciprocal rank fusion constant.
MIN_MATCH_SCORE = 0.63  # Set from the demo data: unrelated words reached 0.59 to 0.62.
WORD = re.compile(r"[A-Za-z0-9]{4,}")

PLAN_SYSTEM = render("search.plan_system")

GRADE_SYSTEM = render("search.grade_system")

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "queries": {"type": "array", "items": {"type": "string"}},
        "keywords": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["queries", "keywords"],
    "additionalProperties": False,
}
GRADE_SCHEMA = {
    "type": "object",
    "properties": {"relevant": {"type": "array", "items": {"type": "string"}}},
    "required": ["relevant"],
    "additionalProperties": False,
}


def hybrid_search(
    store: MemoryStore, text: str, limit: int, table: str | None = None
) -> list[dict]:
    """Merge semantic and keyword matches with no model call."""
    rankings = [store.keyword_search([word], limit, table) for word in WORD.findall(text)]
    try:
        rankings.append(store.search(text, limit, table))
    except ToolError:
        log.warning("semantic_search_unavailable")
    return fuse(rankings, limit)


def find_entries(store: MemoryStore, text: str, limit: int) -> list[dict]:
    """Real matches only, for when the user picks an entry from a list."""
    rankings = [store.keyword_search([word], limit) for word in WORD.findall(text)]
    try:
        rankings.append([r for r in store.search(text, limit) if r["score"] >= MIN_MATCH_SCORE])
    except ToolError:
        log.warning("semantic_search_unavailable")
    return fuse(rankings, limit)


class RetrievalState(TypedDict, total=False):
    question: str
    table: str | None
    queries: list[str]
    keywords: list[str]
    candidates: list[dict]
    relevant: list[dict]
    graded: bool


def result_key(result: dict) -> str:
    return f"{result['table']}#{result['row']['id']}"


def fuse(rankings: list[list[dict]], limit: int = MAX_CANDIDATES) -> list[dict]:
    """Reciprocal rank fusion: rows found by several searches rise to the top."""
    scores: dict[str, float] = {}
    results: dict[str, dict] = {}
    for ranking in rankings:
        for rank, result in enumerate(ranking):
            key = result_key(result)
            scores[key] = scores.get(key, 0.0) + 1 / (FUSION_K + rank)
            results.setdefault(key, result)
    ordered = sorted(scores, key=scores.get, reverse=True)
    return [results[key] for key in ordered[:limit]]


class Retriever:
    def __init__(self, llm: LLM, model: str):
        self.llm = llm
        self.model = model
        graph = StateGraph(RetrievalState)
        graph.add_node("plan", self._plan)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("grade", self._grade)
        graph.add_edge(START, "plan")
        graph.add_edge("plan", "retrieve")
        graph.add_edge("retrieve", "grade")
        graph.add_edge("grade", END)
        self.graph = graph.compile()

    def search(
        self, store: MemoryStore, text: str, table: str | None = None, limit: int = 8
    ) -> dict:
        with measured(log, "search", scoped=table is not None) as stats:
            state = self.graph.invoke(
                {"question": text, "table": table},
                config={"configurable": {"store": store}},
            )
            results = state["relevant"][:limit]
            stats.update(
                candidates=len(state["candidates"]),
                results=len(results),
                graded=state["graded"],
            )
            return {
                "results": results,
                "nothing_relevant": not results,
                "filtered_for_relevance": state["graded"],
            }

    def _plan(self, state: RetrievalState, config: RunnableConfig) -> RetrievalState:
        subjects = [
            f"{table['name']}({', '.join(column['name'] for column in table['columns'])})"
            for table in config["configurable"]["store"].tables()
        ]
        prompt = render("search.plan", subjects="; ".join(subjects), question=state["question"])
        plan = self._ask(PLAN_SYSTEM, prompt, PLAN_SCHEMA) or {}
        return {
            "queries": [state["question"], *plan.get("queries", [])],
            "keywords": plan.get("keywords", []),
        }

    def _retrieve(self, state: RetrievalState, config: RunnableConfig) -> RetrievalState:
        store: MemoryStore = config["configurable"]["store"]
        table = state["table"]
        rankings = []
        for query in state["queries"]:
            try:
                rankings.append(store.search(query, MAX_CANDIDATES, table))
            except ToolError:
                log.warning("semantic_search_unavailable")
                break
        # One ranking per keyword, so a row matching several keywords rises in the fusion.
        for keyword in state["keywords"]:
            rankings.append(store.keyword_search([keyword], MAX_CANDIDATES, table))
        return {"candidates": fuse(rankings)}

    def _grade(self, state: RetrievalState) -> RetrievalState:
        candidates = state["candidates"]
        if not candidates:
            return {"relevant": [], "graded": True}
        entries = json.dumps(
            {result_key(c): c["row"] for c in candidates}, ensure_ascii=False, default=str
        )
        prompt = render("search.grade", question=state["question"], entries=entries)
        verdict = self._ask(GRADE_SYSTEM, prompt, GRADE_SCHEMA)
        if verdict is None:
            return {"relevant": candidates, "graded": False}
        keep = set(verdict["relevant"])
        return {"relevant": [c for c in candidates if result_key(c) in keep], "graded": True}

    def _ask(self, system: str, prompt: str, schema: dict) -> dict | None:
        try:
            return self.llm.ask_json(self.model, system, prompt, schema)
        except LLMError:
            log.warning("search_step_failed", schema=next(iter(schema["properties"])))
            return None
