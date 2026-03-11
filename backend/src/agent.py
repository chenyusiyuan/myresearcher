"""Compatibility wrapper around the LangGraph-based research workflow."""

from __future__ import annotations

import asyncio
from queue import Queue
from threading import Thread
from typing import Any, Iterator

from config import Configuration
from event_mapping import map_langgraph_event
from graph.builder import build_graph
from models import SummaryStateOutput, TodoItem


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _build_initial_state(topic: str, config: Configuration) -> dict[str, Any]:
    runtime_config = config.model_dump(mode="json")
    return {
        "research_topic": topic.strip(),
        "todo_items": [],
        "research_loop_count": 0,
        "structured_report": "",
        "visited_urls": set(),
        "evidence_store": [],
        "research_data": [],
        "review_result": {},
        "revision_count": 0,
        "max_revisions": _safe_int(runtime_config.get("max_revisions"), 2),
        "agent_role": "",
        "config": runtime_config,
    }


def _convert_todo_items(todo_items: list[dict[str, Any]] | None) -> list[TodoItem]:
    converted: list[TodoItem] = []
    for item in todo_items or []:
        if not isinstance(item, dict):
            continue
        converted.append(
            TodoItem(
                id=_safe_int(item.get("id"), len(converted) + 1),
                title=str(item.get("title") or "").strip(),
                intent=str(item.get("intent") or "").strip(),
                query=str(item.get("query") or "").strip(),
                status=str(item.get("status") or "pending").strip() or "pending",
                summary=item.get("summary"),
                sources_summary=item.get("sources_summary"),
                priority=_safe_int(item.get("priority"), 0) or None,
                depends_on=[
                    _safe_int(dep)
                    for dep in item.get("depends_on", [])
                    if _safe_int(dep) > 0
                ] if isinstance(item.get("depends_on"), list) else [],
                search_budget=_safe_int(item.get("search_budget"), 0) or None,
                search_type=str(item.get("search_type") or "").strip() or None,
            )
        )
    return converted


class DeepResearchAgent:
    """Programmatic wrapper for running the standalone LangGraph workflow."""

    def __init__(self, config: Configuration | None = None) -> None:
        self.config = config or Configuration.from_env()
        self.graph = build_graph()

    def run(self, topic: str) -> SummaryStateOutput:
        result = asyncio.run(self.graph.ainvoke(_build_initial_state(topic, self.config)))
        report = str(result.get("structured_report") or "").strip()
        return SummaryStateOutput(
            running_summary=report,
            report_markdown=report,
            todo_items=_convert_todo_items(result.get("todo_items")),
        )

    def run_stream(self, topic: str) -> Iterator[dict[str, Any]]:
        queue: Queue[dict[str, Any] | None] = Queue()

        def worker() -> None:
            asyncio.run(self._stream_into_queue(topic, queue))

        thread = Thread(target=worker, daemon=True)
        thread.start()
        try:
            while True:
                event = queue.get()
                if event is None:
                    break
                yield event
        finally:
            thread.join(timeout=0.1)

    async def _stream_into_queue(self, topic: str, queue: Queue[dict[str, Any] | None]) -> None:
        done_sent = False
        try:
            async for event in self.graph.astream_events(
                _build_initial_state(topic, self.config),
                version="v2",
            ):
                for mapped_event in map_langgraph_event(event):
                    if mapped_event.get("type") == "done":
                        done_sent = True
                    queue.put(mapped_event)
        except Exception as exc:  # pragma: no cover - defensive guardrail
            queue.put({"type": "error", "detail": str(exc)})
        finally:
            if not done_sent:
                queue.put({"type": "done"})
            queue.put(None)


def run_deep_research(topic: str, config: Configuration | None = None) -> SummaryStateOutput:
    """Convenience helper to run the deep research workflow."""
    return DeepResearchAgent(config=config).run(topic)
