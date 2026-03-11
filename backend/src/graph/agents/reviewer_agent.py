from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, StateGraph

from ..nodes.reviewer import reviewer_node
from ..state import AgentMessage, EvidenceItem, TodoItem, merge_agent_messages, merge_todo_items
from ..supervisor import build_agent_message


class ReviewerAgentState(TypedDict, total=False):
    research_topic: str
    todo_items: Annotated[list[TodoItem], merge_todo_items]
    evidence_store: Annotated[list[EvidenceItem], operator.add]
    review_result: dict[str, Any]
    revision_count: int
    structured_report: str
    config: dict[str, Any]
    messages: Annotated[list[AgentMessage], merge_agent_messages]


def _reviewer_handoff(state: ReviewerAgentState) -> dict[str, Any]:
    review_result = state.get("review_result", {})
    approved = bool(review_result.get("approved"))
    research_briefs = [
        item
        for item in review_result.get("research_briefs", [])
        if isinstance(item, dict)
    ]
    missing_topics = [
        str(item).strip()
        for item in review_result.get("missing_topics", [])
        if str(item).strip()
    ]
    section_patch_plan = [
        item
        for item in review_result.get("section_patch_plan", [])
        if isinstance(item, dict)
    ]

    if approved:
        message_type = "report_approved"
        payload = {
            "score": review_result.get("score"),
            "feedback": review_result.get("feedback"),
        }
    elif research_briefs or missing_topics:
        message_type = "review_dispatch"
        payload = {
            "research_briefs": research_briefs,
            "missing_topics": missing_topics,
            "section_patch_plan": section_patch_plan,
        }
    elif section_patch_plan:
        message_type = "patch_order"
        payload = {
            "section_patch_plan": section_patch_plan,
            "feedback": review_result.get("feedback"),
        }
    else:
        message_type = "rewrite_order"
        payload = {
            "feedback": review_result.get("feedback"),
            "weak_sections": review_result.get("weak_sections", []),
        }

    return {
        "messages": [
            build_agent_message(
                "reviewer_agent",
                "supervisor",
                message_type,
                payload,
            )
        ]
    }


def build_reviewer_graph():
    graph = StateGraph(ReviewerAgentState)
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("reviewer_handoff", _reviewer_handoff)
    graph.set_entry_point("reviewer")
    graph.add_edge("reviewer", "reviewer_handoff")
    graph.add_edge("reviewer_handoff", END)
    return graph.compile(name="reviewer_agent")
