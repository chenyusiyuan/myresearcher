from __future__ import annotations

from langgraph.graph import END, StateGraph
from langgraph.types import Send

from .nodes.planner import planner_node
from .nodes.research_more import research_more_node
from .nodes.reviewer import reviewer_node
from .nodes.task import task_node
from .nodes.writer import writer_node
from .state import ResearchState


def _coerce_positive_int(value: object) -> int | None:
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str) and value.strip().isdigit():
        parsed = int(value.strip())
        return parsed if parsed > 0 else None
    return None


def _normalize_depends_on(task: dict) -> list[int]:
    raw_depends_on = task.get("depends_on")
    if not isinstance(raw_depends_on, list):
        return []

    normalized: list[int] = []
    seen: set[int] = set()
    for item in raw_depends_on:
        parsed = _coerce_positive_int(item)
        if parsed is None or parsed in seen:
            continue
        seen.add(parsed)
        normalized.append(parsed)
    return normalized


def _task_sort_key(task: dict) -> tuple[int, int, str]:
    priority = _coerce_positive_int(task.get("priority")) or 10**9
    task_id = _coerce_positive_int(task.get("id")) or 10**9
    title = str(task.get("title") or "").strip()
    return (priority, task_id, title)


def _build_task_send(state: ResearchState, task: dict) -> Send:
    return Send(
        "task_node",
        {
            "task": task,
            "config": state["config"],
            "research_topic": state["research_topic"],
            "visited_urls": state["visited_urls"],
            "research_loop_count": state.get("research_loop_count", 0),
        },
    )


def _select_runnable_tasks(state: ResearchState) -> list[dict]:
    todo_items = [
        task
        for task in state.get("todo_items", [])
        if isinstance(task, dict) and str(task.get("status") or "").strip() == "pending"
    ]
    if not todo_items:
        return []

    completed_ids = {
        parsed_id
        for task in state.get("todo_items", [])
        if isinstance(task, dict)
        and str(task.get("status") or "").strip() == "completed"
        and (parsed_id := _coerce_positive_int(task.get("id"))) is not None
    }

    runnable = [
        task
        for task in todo_items
        if all(dep in completed_ids for dep in _normalize_depends_on(task))
    ]
    selected = runnable or todo_items
    return sorted(selected, key=_task_sort_key)


def route_tasks(state: ResearchState) -> list[Send]:
    """Fan out each planned task into an independent task node run."""
    return [_build_task_send(state, task) for task in _select_runnable_tasks(state)]


def route_after_task_batch(state: ResearchState) -> list[Send] | str:
    next_tasks = _select_runnable_tasks(state)
    if next_tasks:
        return [_build_task_send(state, task) for task in next_tasks]
    return "writer"


def route_research_more(state: ResearchState) -> list[Send]:
    """Dispatch reviewer-added pending tasks back into task_node fan-out."""
    return [_build_task_send(state, task) for task in _select_runnable_tasks(state)]


def route_after_review(state: ResearchState) -> str:
    review = state.get("review_result", {})
    if review.get("approved"):
        return "end"
    revision_count = state.get("revision_count", 0)
    max_revisions = state.get("max_revisions", 2)
    # missing_topics 表示需要补充搜索，优先处理（至少允许一次补搜，即使已超轮次）
    if review.get("missing_topics"):
        if revision_count <= max_revisions + 1:
            return "research_more"
    if revision_count > max_revisions:
        return "end"
    return "rewrite"


def build_graph():
    graph = StateGraph(ResearchState)
    graph.add_node("planner", planner_node)
    graph.add_node("task_node", task_node)
    graph.add_node("writer", writer_node)
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("research_more", research_more_node)

    graph.set_entry_point("planner")
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
    return graph.compile()
