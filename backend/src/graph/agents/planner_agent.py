from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, StateGraph

from ..nodes.planner import planner_node
from ..state import AgentMessage, TodoItem, merge_agent_messages, merge_todo_items
from ..supervisor import build_agent_message


class PlannerAgentState(TypedDict):
    research_topic: str
    config: dict[str, Any]
    todo_items: Annotated[list[TodoItem], merge_todo_items]
    agent_role: str
    messages: Annotated[list[AgentMessage], merge_agent_messages]


def _planner_handoff(state: PlannerAgentState) -> dict[str, Any]:
    tasks = [
        dict(item)
        for item in state.get("todo_items", [])
        if isinstance(item, dict)
    ]
    return {
        "messages": [
            build_agent_message(
                "planner_agent",
                "supervisor",
                "task_assignment",
                {"tasks": tasks, "task_count": len(tasks)},
            )
        ]
    }


def build_planner_graph():
    graph = StateGraph(PlannerAgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("planner_handoff", _planner_handoff)
    graph.set_entry_point("planner")
    graph.add_edge("planner", "planner_handoff")
    graph.add_edge("planner_handoff", END)
    return graph.compile(name="planner_agent")
