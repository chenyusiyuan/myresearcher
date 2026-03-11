from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from langgraph.graph import END
from langgraph.types import Command, Send

from .state import AgentMessage, GlobalState


def build_agent_message(
    from_agent: str,
    to_agent: str,
    message_type: str,
    payload: dict[str, Any] | None = None,
) -> AgentMessage:
    return {
        "from_agent": str(from_agent or "").strip(),
        "to_agent": str(to_agent or "").strip(),
        "type": str(message_type or "").strip(),
        "payload": dict(payload or {}),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _coerce_positive_int(value: object) -> int | None:
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str) and value.strip().isdigit():
        parsed = int(value.strip())
        return parsed if parsed > 0 else None
    return None


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _task_sort_key(task: dict[str, Any]) -> tuple[int, int, str]:
    priority = _coerce_positive_int(task.get("priority")) or 10**9
    task_id = _coerce_positive_int(task.get("id")) or 10**9
    title = str(task.get("title") or "").strip()
    return (priority, task_id, title)


def select_runnable_tasks(state: dict[str, Any]) -> list[dict[str, Any]]:
    pending_tasks = [
        dict(task)
        for task in state.get("todo_items", [])
        if isinstance(task, dict) and str(task.get("status") or "").strip() == "pending"
    ]
    return sorted(pending_tasks, key=_task_sort_key)


def _build_researcher_send(state: dict[str, Any], task: dict[str, Any]) -> Send:
    return Send(
        "researcher_agent",
        {
            "task": dict(task),
            "runtime_config": state.get("config", {}),
            "root_research_topic": state.get("research_topic", ""),
            "visited_urls": state.get("visited_urls", set()),
            "input_research_loop_count": int(state.get("research_loop_count", 0)),
            "messages": [],
        },
    )


def _mark_tasks_in_progress(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    updates: list[dict[str, Any]] = []
    for task in tasks:
        updated = dict(task)
        updated["status"] = "in_progress"
        updates.append(updated)
    return updates


def _latest_supervisor_message(state: dict[str, Any]) -> AgentMessage | None:
    messages = state.get("messages", [])
    if not isinstance(messages, list):
        return None

    for item in reversed(messages):
        if not isinstance(item, dict):
            continue
        if str(item.get("to_agent") or "").strip() not in {"", "supervisor"}:
            continue
        message_type = str(item.get("type") or "").strip()
        if not message_type:
            continue
        return item  # type: ignore[return-value]
    return None


def _dispatch_researchers(
    state: GlobalState,
    tasks: list[dict[str, Any]],
    *,
    supplemental: bool = False,
) -> Command:
    if not tasks:
        return Command(goto="writer_agent", update={"status": "writing"})

    update: dict[str, Any] = {
        "status": "researching",
        "todo_items": _mark_tasks_in_progress(tasks),
    }
    if supplemental:
        update["research_loop_count"] = int(state.get("research_loop_count", 0)) + 1

    return Command(
        goto=[_build_researcher_send(state, task) for task in tasks],
        update=update,
    )


def _finalize_report(state: GlobalState) -> Command:
    report = str(state.get("structured_report") or state.get("final_report") or "").strip()
    return Command(
        goto=END,
        update={
            "status": "done",
            "final_report": report,
            "structured_report": report,
        },
    )


def supervisor_node(state: GlobalState) -> Command:
    status = str(state.get("status") or "init").strip() or "init"
    last_message = _latest_supervisor_message(state)
    review_result = state.get("review_result", {})
    if not isinstance(review_result, dict):
        review_result = {}
    revision_count = _safe_int(state.get("revision_count", 0), 0)
    max_revisions = _safe_int(state.get("max_revisions", 2), 2)

    if status == "init":
        return Command(goto="planner_agent", update={"status": "planning"})

    if last_message is None:
        runnable_tasks = select_runnable_tasks(state)
        if runnable_tasks:
            return _dispatch_researchers(state, runnable_tasks)
        if str(state.get("structured_report") or "").strip():
            return Command(goto="reviewer_agent", update={"status": "reviewing"})
        return Command(goto="planner_agent", update={"status": "planning"})

    message_type = str(last_message.get("type") or "").strip()
    if message_type == "task_assignment":
        runnable_tasks = select_runnable_tasks(state)
        if runnable_tasks:
            return _dispatch_researchers(state, runnable_tasks)
        return Command(goto="writer_agent", update={"status": "writing"})

    if message_type == "evidence_delivery":
        next_tasks = select_runnable_tasks(state)
        if next_tasks:
            return _dispatch_researchers(state, next_tasks)
        return Command(goto="writer_agent", update={"status": "writing"})

    if message_type == "report_ready":
        return Command(goto="reviewer_agent", update={"status": "reviewing"})

    if message_type == "report_approved":
        return _finalize_report(state)

    if message_type == "review_dispatch":
        if revision_count > max_revisions + 1:
            return _finalize_report(state)
        next_tasks = select_runnable_tasks(state)
        if next_tasks:
            return _dispatch_researchers(state, next_tasks, supplemental=True)
        if revision_count > max_revisions:
            return _finalize_report(state)
        return Command(goto="writer_agent", update={"status": "writing"})

    if message_type in {"patch_order", "rewrite_order"}:
        if revision_count > max_revisions:
            return _finalize_report(state)
        return Command(goto="writer_agent", update={"status": "writing"})

    if revision_count > max_revisions:
        return _finalize_report(state)

    if str(state.get("structured_report") or "").strip():
        return Command(goto="reviewer_agent", update={"status": "reviewing"})

    return Command(goto="planner_agent", update={"status": "planning"})
