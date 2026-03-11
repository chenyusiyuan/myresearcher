from __future__ import annotations

import operator
from typing import Annotated, Any, Optional, TypedDict

from langgraph.graph import END, StateGraph

from ..nodes.writer import writer_node
from ..state import AgentMessage, EvidenceItem, TodoItem, merge_agent_messages, merge_todo_items
from ..supervisor import build_agent_message


class WriterAgentState(TypedDict, total=False):
    research_topic: str
    todo_items: Annotated[list[TodoItem], merge_todo_items]
    research_data: Annotated[list[dict[str, Any]], operator.add]
    evidence_store: Annotated[list[EvidenceItem], operator.add]
    review_result: dict[str, Any]
    structured_report: str
    final_report: Optional[str]
    config: dict[str, Any]
    messages: Annotated[list[AgentMessage], merge_agent_messages]


def _writer_handoff(state: WriterAgentState) -> dict[str, Any]:
    report = str(state.get("structured_report") or "").strip()
    patch_plan = state.get("review_result", {}).get("section_patch_plan", [])
    return {
        "messages": [
            build_agent_message(
                "writer_agent",
                "supervisor",
                "report_ready",
                {
                    "report_length": len(report),
                    "patched": bool(isinstance(patch_plan, list) and patch_plan),
                },
            )
        ],
        "final_report": report or None,
    }


def build_writer_graph():
    graph = StateGraph(WriterAgentState)
    graph.add_node("writer", writer_node)
    graph.add_node("writer_handoff", _writer_handoff)
    graph.set_entry_point("writer")
    graph.add_edge("writer", "writer_handoff")
    graph.add_edge("writer_handoff", END)
    return graph.compile(name="writer_agent")
