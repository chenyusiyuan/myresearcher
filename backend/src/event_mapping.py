"""Shared LangGraph event mapping helpers for API and programmatic clients."""

from __future__ import annotations

from typing import Any


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def serialize_todo_items(todo_items: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for item in todo_items or []:
        if not isinstance(item, dict):
            continue
        serialized.append(
            {
                "id": safe_int(item.get("id"), len(serialized) + 1),
                "title": str(item.get("title") or "").strip(),
                "intent": str(item.get("intent") or "").strip(),
                "query": str(item.get("query") or "").strip(),
                "status": str(item.get("status") or "").strip() or "pending",
                "summary": item.get("summary"),
                "sources_summary": item.get("sources_summary"),
            }
        )
    return serialized


def _event_node_name(event: dict[str, Any]) -> str:
    metadata = event.get("metadata")
    if isinstance(metadata, dict):
        node_name = metadata.get("langgraph_node")
        if isinstance(node_name, str) and node_name.strip():
            return node_name.strip()
    name = event.get("name")
    if isinstance(name, str):
        return name.strip()
    return ""


def _extract_task_from_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        task = payload.get("task")
        if isinstance(task, dict):
            return task
    return {}


def _extract_first_dict(value: Any, key: str) -> dict[str, Any]:
    if isinstance(value, dict):
        items = value.get(key)
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    return item
    return {}


def map_langgraph_event(event: dict[str, Any]) -> list[dict[str, Any]]:
    event_type = str(event.get("event") or "").strip()
    node_name = _event_node_name(event)
    data = event.get("data")
    payload = data if isinstance(data, dict) else {}
    mapped_events: list[dict[str, Any]] = []

    if event_type == "on_chain_start" and node_name == "planner":
        return [{"type": "status", "message": "规划研究任务..."}]

    if event_type == "on_chain_end" and node_name == "planner":
        output = payload.get("output")
        if isinstance(output, dict):
            return [{"type": "todo_list", "tasks": serialize_todo_items(output.get("todo_items"))}]
        return []

    if event_type == "on_chain_start" and node_name == "task_node":
        task = _extract_task_from_payload(payload.get("input"))
        task_id = safe_int(task.get("id"))
        return [
            {
                "type": "task_status",
                "task_id": task_id,
                "title": str(task.get("title") or "").strip(),
                "intent": str(task.get("intent") or "").strip(),
                "query": str(task.get("query") or "").strip(),
                "status": "in_progress",
            }
        ] if task_id else []

    if event_type == "on_chain_end" and node_name == "task_node":
        output = payload.get("output")
        if not isinstance(output, dict):
            return []

        task_payloads = serialize_todo_items(output.get("todo_items"))
        task_payload = task_payloads[0] if task_payloads else {}
        research_payload = _extract_first_dict(output, "research_data")
        task_id = safe_int(task_payload.get("id") or research_payload.get("task_id"))
        summary = task_payload.get("summary")

        if task_id and isinstance(summary, str) and summary.strip():
            mapped_events.append(
                {
                    "type": "task_summary_chunk",
                    "task_id": task_id,
                    "content": summary,
                }
            )

        if task_id:
            mapped_events.append(
                {
                    "type": "task_status",
                    "task_id": task_id,
                    "title": task_payload.get("title"),
                    "intent": task_payload.get("intent"),
                    "query": task_payload.get("query"),
                    "status": str(task_payload.get("status") or "completed"),
                    "summary": summary,
                    "sources_summary": task_payload.get("sources_summary"),
                }
            )

        if task_id and research_payload:
            mapped_events.append(
                {
                    "type": "sources",
                    "task_id": task_id,
                    "sources_summary": research_payload.get("sources_summary")
                    or task_payload.get("sources_summary"),
                    "latest_sources": research_payload.get("sources_summary")
                    or task_payload.get("sources_summary"),
                    "raw_context": research_payload.get("context"),
                    "backend": research_payload.get("backend"),
                }
            )

        notices = research_payload.get("notices") if isinstance(research_payload, dict) else None
        if task_id and isinstance(notices, list):
            for notice in notices:
                message = str(notice).strip()
                if message:
                    mapped_events.append(
                        {
                            "type": "status",
                            "task_id": task_id,
                            "message": message,
                        }
                    )
        return mapped_events

    if event_type == "on_chain_start" and node_name == "writer":
        return [{"type": "status", "message": "正在生成研究报告..."}]

    if event_type == "on_chain_end" and node_name == "writer":
        output = payload.get("output")
        if isinstance(output, dict):
            report = str(output.get("structured_report") or "").strip()
            if report:
                return [{"type": "final_report", "report": report}]
        return []

    if event_type == "on_chain_start" and node_name == "reviewer":
        return [{"type": "status", "message": "Reviewer 正在审查报告..."}]

    if event_type == "on_chain_end" and node_name == "reviewer":
        output = payload.get("output")
        if not isinstance(output, dict):
            return []
        review_result = output.get("review_result")
        if not isinstance(review_result, dict):
            return []

        mapped_events.append(
            {
                "type": "review_result",
                "result": review_result,
                "revision_count": safe_int(output.get("revision_count"), 0),
            }
        )

        approved = bool(review_result.get("approved"))
        missing_topics = [
            str(topic).strip()
            for topic in review_result.get("missing_topics", [])
            if str(topic).strip()
        ]
        if approved:
            mapped_events.append({"type": "status", "message": "Reviewer 已通过当前报告。"})
        elif missing_topics:
            mapped_events.append(
                {
                    "type": "status",
                    "message": f"Reviewer 要求补充研究：{'、'.join(missing_topics)}",
                }
            )
        else:
            mapped_events.append({"type": "status", "message": "Reviewer 要求继续重写报告。"})
        return mapped_events

    if event_type == "on_chain_start" and node_name == "research_more":
        return [{"type": "status", "message": "Reviewer 已提出补研主题，正在分派补充任务..."}]

    if event_type == "on_chain_end" and node_name == "LangGraph":
        return [{"type": "done"}]

    if event_type == "on_custom_event" and event.get("name") == "report_chunk":
        data = event.get("data") or {}
        token = str(data.get("token") or "")
        if token:
            return [{"type": "report_chunk", "token": token}]

    return []
