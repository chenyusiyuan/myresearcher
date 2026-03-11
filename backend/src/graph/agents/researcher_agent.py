from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, StateGraph

from ..nodes.task import task_node
from ..state import AgentMessage, EvidenceItem, TodoItem, merge_agent_messages, merge_todo_items
from ..supervisor import build_agent_message


class ResearcherAgentState(TypedDict, total=False):
    task: dict[str, Any]
    config: dict[str, Any]
    research_topic: str
    visited_urls: Annotated[set[str], operator.or_]
    research_loop_count: int
    research_data: Annotated[list[dict[str, Any]], operator.add]
    evidence_store: Annotated[list[EvidenceItem], operator.add]
    todo_items: Annotated[list[TodoItem], merge_todo_items]
    messages: Annotated[list[AgentMessage], merge_agent_messages]


def _researcher_handoff(state: ResearcherAgentState) -> dict[str, Any]:
    task = dict(state.get("task") or {})
    task_id = task.get("id")
    completed_task = next(
        (
            item
            for item in state.get("todo_items", [])
            if isinstance(item, dict) and item.get("id") == task_id
        ),
        {},
    )
    research_payload = next(
        (
            item
            for item in state.get("research_data", [])
            if isinstance(item, dict) and item.get("task_id") == task_id
        ),
        {},
    )
    evidence_items = [
        item
        for item in state.get("evidence_store", [])
        if isinstance(item, dict) and item.get("task_id") == task_id
    ]
    payload = {
        "task_id": completed_task.get("id") or task_id,
        "title": completed_task.get("title") or task.get("title"),
        "query": completed_task.get("query") or task.get("query"),
        "summary": completed_task.get("summary") or research_payload.get("summary"),
        "source_count": len(research_payload.get("sources") or []),
        "evidence_count": len(evidence_items),
    }
    return {
        "messages": [
            build_agent_message(
                "researcher_agent",
                "supervisor",
                "evidence_delivery",
                payload,
            )
        ]
    }


def build_researcher_graph():
    graph = StateGraph(ResearcherAgentState)
    graph.add_node("task_node", task_node)
    graph.add_node("researcher_handoff", _researcher_handoff)
    graph.set_entry_point("task_node")
    graph.add_edge("task_node", "researcher_handoff")
    graph.add_edge("researcher_handoff", END)
    return graph.compile(name="researcher_agent")
