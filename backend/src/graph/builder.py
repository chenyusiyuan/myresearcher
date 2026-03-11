from __future__ import annotations

from langgraph.graph import StateGraph

from .agents import (
    build_planner_graph,
    build_researcher_graph,
    build_reviewer_graph,
    build_writer_graph,
)
from .state import GlobalState
from .supervisor import supervisor_node


def build_graph():
    graph = StateGraph(GlobalState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("planner_agent", build_planner_graph())
    graph.add_node("researcher_agent", build_researcher_graph())
    graph.add_node("writer_agent", build_writer_graph())
    graph.add_node("reviewer_agent", build_reviewer_graph())

    graph.set_entry_point("supervisor")
    graph.add_edge("planner_agent", "supervisor")
    graph.add_edge("researcher_agent", "supervisor")
    graph.add_edge("writer_agent", "supervisor")
    graph.add_edge("reviewer_agent", "supervisor")
    return graph.compile()
