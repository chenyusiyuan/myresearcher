from __future__ import annotations

from typing import Any

from typing_extensions import NotRequired, TypedDict

from config import Configuration, SearchAPI
from langgraph.graph import StateGraph

from graph.agents import (
    build_planner_graph,
    build_researcher_graph,
    build_reviewer_graph,
    build_writer_graph,
)
from graph.state import GlobalState, TodoItem
from graph.supervisor import supervisor_node


class StudioInputState(TypedDict):
    """Minimal input surface exposed to LangGraph Studio."""

    research_topic: str
    search_api: NotRequired[str]
    research_depth: NotRequired[int]


class StudioOutputState(TypedDict):
    """Compact output surface shown at the end of a Studio run."""

    structured_report: str
    todo_items: list[TodoItem]
    review_result: dict[str, Any]


def _coerce_positive_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str) and value.strip().isdigit():
        numeric = int(value.strip())
        return numeric if numeric > 0 else None
    return None


def _build_initial_state(state: StudioInputState) -> dict[str, Any]:
    topic = str(state.get("research_topic") or "").strip()
    if not topic:
        raise ValueError("research_topic is required for LangGraph Studio runs.")

    overrides: dict[str, Any] = {}

    raw_search_api = state.get("search_api")
    if isinstance(raw_search_api, SearchAPI):
        overrides["search_api"] = raw_search_api
    elif isinstance(raw_search_api, str) and raw_search_api.strip():
        overrides["search_api"] = raw_search_api.strip().lower()

    research_depth = _coerce_positive_int(state.get("research_depth"))
    if research_depth is not None:
        overrides["deep_research_depth"] = research_depth

    config = Configuration.from_env(overrides=overrides)
    runtime_config = config.model_dump(mode="json")

    return {
        "research_topic": topic,
        "todo_items": [],
        "research_loop_count": 0,
        "structured_report": "",
        "visited_urls": set(),
        "evidence_store": [],
        "research_data": [],
        "review_result": {},
        "revision_count": 0,
        "max_revisions": int(runtime_config.get("max_revisions") or 2),
        "agent_role": "",
        "config": runtime_config,
        "messages": [],
        "final_report": None,
        "status": "init",
    }


def prepare_studio_input(state: StudioInputState) -> dict[str, Any]:
    """Normalize LangGraph Studio input into the internal global state."""

    return _build_initial_state(state)


def build_studio_graph():
    graph = StateGraph(
        GlobalState,
        input_schema=StudioInputState,
        output_schema=StudioOutputState,
    )
    graph.add_node("prepare", prepare_studio_input)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("planner_agent", build_planner_graph())
    graph.add_node("researcher_agent", build_researcher_graph())
    graph.add_node("writer_agent", build_writer_graph())
    graph.add_node("reviewer_agent", build_reviewer_graph())

    graph.set_entry_point("prepare")
    graph.add_edge("prepare", "supervisor")
    graph.add_edge("planner_agent", "supervisor")
    graph.add_edge("researcher_agent", "supervisor")
    graph.add_edge("writer_agent", "supervisor")
    graph.add_edge("reviewer_agent", "supervisor")
    return graph.compile(name="deep_researcher")
