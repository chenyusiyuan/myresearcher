from __future__ import annotations

from langgraph.graph import END, StateGraph
from langgraph.types import Send

from .nodes.planner import planner_node
from .nodes.research_more import research_more_node
from .nodes.reviewer import reviewer_node
from .nodes.task import task_node
from .nodes.writer import writer_node
from .state import ResearchState


def route_tasks(state: ResearchState) -> list[Send]:
    """Fan out each planned task into an independent task node run."""
    return [
        Send(
            "task_node",
            {
                "task": task,
                "config": state["config"],
                "research_topic": state["research_topic"],
                "visited_urls": state["visited_urls"],
                "research_loop_count": state.get("research_loop_count", 0),
            },
        )
        for task in state.get("todo_items", [])
    ]


def route_research_more(state: ResearchState) -> list[Send]:
    """Dispatch reviewer-added pending tasks back into task_node fan-out."""
    missing_topics = {
        str(topic).strip()
        for topic in state.get("review_result", {}).get("missing_topics", [])
        if str(topic).strip()
    }

    return [
        Send(
            "task_node",
            {
                "task": task,
                "config": state["config"],
                "research_topic": state["research_topic"],
                "visited_urls": state["visited_urls"],
                "research_loop_count": state.get("research_loop_count", 0),
            },
        )
        for task in state.get("todo_items", [])
        if task.get("status") == "pending"
        and (
            not missing_topics
            or str(task.get("title", "")).strip() in missing_topics
            or str(task.get("query", "")).strip() in missing_topics
        )
    ]


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
    graph.add_edge("task_node", "writer")
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
