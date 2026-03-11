from __future__ import annotations

from typing import Any

from typing_extensions import NotRequired, TypedDict

from config import Configuration, SearchAPI
from graph.builder import route_after_review, route_after_task_batch, route_research_more, route_tasks
from graph.nodes.planner import planner_node
from graph.nodes.research_more import research_more_node
from graph.nodes.reviewer import reviewer_node
from graph.nodes.task import task_node
from graph.nodes.writer import writer_node
from graph.state import ResearchState, TodoItem
from langgraph.graph import END, StateGraph


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
    }


def prepare_studio_input(state: StudioInputState) -> dict[str, Any]:
    """Normalize LangGraph Studio input into the internal research state."""

    return _build_initial_state(state)


def build_studio_graph():
    graph = StateGraph(
        ResearchState,
        input_schema=StudioInputState,
        output_schema=StudioOutputState,
    )
    graph.add_node("prepare", prepare_studio_input)
    graph.add_node("planner", planner_node)
    graph.add_node("task_node", task_node)
    graph.add_node("writer", writer_node)
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("research_more", research_more_node)

    graph.set_entry_point("prepare")
    graph.add_edge("prepare", "planner")
    graph.add_conditional_edges("planner", route_tasks, ["task_node"])
    graph.add_conditional_edges(
        "task_node",
        route_after_task_batch,
        {"writer": "writer"},
    )
    graph.add_edge("writer", "reviewer")
    graph.add_conditional_edges(
        "reviewer",
        route_after_review,
        {
            "end": END,
            "research_more": "research_more",
            "rewrite": "writer",
        },
    )
    graph.add_conditional_edges("research_more", route_research_more, ["task_node"])
    return graph.compile(name="deep_researcher")
