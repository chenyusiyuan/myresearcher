from __future__ import annotations

import operator
from typing import Annotated, Any, NotRequired, Optional, TypedDict


def _coerce_task_id(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def merge_todo_items(
    current: list["TodoItem"],
    updates: list["TodoItem"],
) -> list["TodoItem"]:
    """Merge todo items by id while preserving the original order."""
    if not current:
        return list(updates)
    if not updates:
        return list(current)

    merged = [dict(item) for item in current]
    index_by_id = {
        task_id: index
        for index, item in enumerate(merged)
        if (task_id := _coerce_task_id(item.get("id"))) is not None
    }

    for update in updates:
        task_id = _coerce_task_id(update.get("id"))
        if task_id is None:
            merged.append(dict(update))
            continue

        if task_id in index_by_id:
            original = merged[index_by_id[task_id]]
            merged[index_by_id[task_id]] = {**original, **dict(update)}
        else:
            index_by_id[task_id] = len(merged)
            merged.append(dict(update))

    return merged


_MAX_AGENT_MESSAGES = 64


def merge_agent_messages(
    current: list["AgentMessage"],
    updates: list["AgentMessage"],
) -> list["AgentMessage"]:
    if not current:
        merged = list(updates)
    elif not updates:
        merged = list(current)
    else:
        merged = [*current, *updates]
    if len(merged) <= _MAX_AGENT_MESSAGES:
        return merged
    return merged[-_MAX_AGENT_MESSAGES:]


class EvidenceItem(TypedDict):
    task_id: int
    url: str
    title: str
    snippet: str
    relevance_score: float
    claim_text: NotRequired[Optional[str]]
    support_type: NotRequired[Optional[str]]
    section_hint: NotRequired[Optional[str]]


class TodoItem(TypedDict):
    id: int
    title: str
    intent: str
    query: str
    status: str
    summary: Optional[str]
    sources_summary: Optional[str]
    priority: NotRequired[int]
    depends_on: NotRequired[list[int]]
    search_budget: NotRequired[int]
    search_type: NotRequired[str]


class AgentMessage(TypedDict):
    from_agent: str
    to_agent: str
    type: str
    payload: dict[str, Any]
    timestamp: str


class ResearchState(TypedDict):
    research_topic: str
    todo_items: Annotated[list[TodoItem], merge_todo_items]
    research_loop_count: int
    structured_report: str
    visited_urls: Annotated[set[str], operator.or_]
    evidence_store: Annotated[list[EvidenceItem], operator.add]
    research_data: Annotated[list[dict], operator.add]
    review_result: dict
    revision_count: int
    max_revisions: int
    agent_role: str
    config: dict


class GlobalState(ResearchState):
    messages: Annotated[list[AgentMessage], merge_agent_messages]
    final_report: Optional[str]
    status: str
