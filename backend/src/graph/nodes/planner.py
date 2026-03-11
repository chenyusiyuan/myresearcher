from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

from openai import AsyncOpenAI

from config import Configuration
from prompts import get_current_date, todo_planner_instructions, todo_planner_system_prompt
from utils import strip_thinking_tokens, with_llm_retry

from ..state import ResearchState, TodoItem

try:
    import json_repair
except ImportError:  # pragma: no cover - optional dependency
    json_repair = None


_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def _sanitize_system_prompt() -> str:
    prompt = todo_planner_system_prompt
    prompt = re.sub(r"\n<NOTE_COLLAB>.*?</NOTE_COLLAB>\n?", "\n", prompt, flags=re.DOTALL)
    prompt = re.sub(r"\n<TOOLS>.*?</TOOLS>\n?", "\n", prompt, flags=re.DOTALL)
    prompt = prompt.replace(
        "4. 在创建或更新任务时，必须调用 `note` 工具同步任务信息（这是唯一会写入笔记的途径）。\n",
        "",
    )
    return prompt.strip()


def _build_planner_prompt(research_topic: str) -> str:
    prompt = todo_planner_instructions.format(
        current_date=get_current_date(),
        research_topic=research_topic,
    )
    return prompt.replace("必要时使用笔记工具记录你的思考过程。", "").strip()


def _build_agent_role(research_topic: str) -> str:
    return (
        "你是一名资深研究分析师，负责将复杂主题拆解为高价值任务，"
        "并在后续研究、写作与审查阶段持续保持证据导向、结构清晰、覆盖完整。"
        f"当前研究主题：{research_topic}"
    )


def _fallback_task(research_topic: str) -> TodoItem:
    return {
        "id": 1,
        "title": "基础背景梳理",
        "intent": "收集主题的核心背景与最新动态",
        "query": f"{research_topic} 最新进展" if research_topic else "基础背景梳理",
        "status": "pending",
        "summary": None,
        "sources_summary": None,
    }


def _parse_selector(value: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    if not value:
        return None, None

    raw_value = value.strip()
    if not raw_value:
        return None, None

    if ":" not in raw_value:
        return None, raw_value

    provider, model = raw_value.split(":", 1)
    provider = provider.strip() or None
    model = model.strip() or None
    return provider, model


def _resolve_model_config(
    runtime_config: dict[str, Any],
    selector_key: str = "strategic_llm",
) -> tuple[Configuration, str, str]:
    config = Configuration.from_env(overrides=runtime_config)
    selected_provider, selected_model = _parse_selector(runtime_config.get(selector_key))
    provider = selected_provider or config.llm_provider
    model = selected_model or config.resolved_model()
    if not provider or not model:
        raise ValueError("Planner node is missing a valid LLM provider or model configuration.")
    return config, provider, model


def _resolve_client(config: Configuration, provider: str) -> AsyncOpenAI:
    if provider == "ollama":
        base_url = config.sanitized_ollama_url()
        api_key = config.llm_api_key or "ollama"
    elif provider == "lmstudio":
        base_url = config.lmstudio_base_url
        api_key = config.llm_api_key or "lmstudio"
    else:
        base_url = config.llm_base_url
        api_key = config.llm_api_key or os.getenv("OPENAI_API_KEY") or "EMPTY"

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    return AsyncOpenAI(**client_kwargs)


async def _request_planner_response(
    research_topic: str,
    runtime_config: dict[str, Any],
) -> str:
    config, provider, model = _resolve_model_config(runtime_config)
    client = _resolve_client(config, provider)
    messages = [
        {"role": "system", "content": _sanitize_system_prompt()},
        {"role": "user", "content": _build_planner_prompt(research_topic)},
    ]
    response = await with_llm_retry(
        lambda: client.chat.completions.create(
            model=model,
            temperature=0,
            messages=messages,
        )
    )
    content = response.choices[0].message.content or ""
    return strip_thinking_tokens(content).strip()


def _load_json(candidate: str) -> Any:
    if json_repair is not None:
        try:
            return json_repair.loads(candidate)
        except Exception:
            pass
    return json.loads(candidate)


def _extract_json_payload(text: str) -> Any:
    candidates: list[str] = []
    fenced_match = _FENCED_JSON_RE.search(text)
    if fenced_match:
        candidates.append(fenced_match.group(1).strip())

    stripped = text.strip()
    if stripped:
        candidates.append(stripped)

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start : end + 1])

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            return _load_json(candidate)
        except Exception:
            continue
    return None


def _extract_tasks(raw_response: str) -> list[dict[str, Any]]:
    payload = _extract_json_payload(raw_response)
    if isinstance(payload, dict):
        tasks = payload.get("tasks")
        if isinstance(tasks, list):
            return [item for item in tasks if isinstance(item, dict)]
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _normalize_tasks(research_topic: str, tasks_payload: list[dict[str, Any]]) -> list[TodoItem]:
    todo_items: list[TodoItem] = []
    for idx, item in enumerate(tasks_payload, start=1):
        title = str(item.get("title") or f"任务{idx}").strip()
        intent = str(item.get("intent") or "聚焦主题的关键问题").strip()
        query = str(item.get("query") or research_topic).strip() or research_topic
        todo_items.append(
            {
                "id": idx,
                "title": title,
                "intent": intent,
                "query": query,
                "status": "pending",
                "summary": None,
                "sources_summary": None,
            }
        )
    return todo_items


async def planner_node(state: ResearchState) -> dict[str, Any]:
    """Generate the initial todo list for the LangGraph research workflow."""
    research_topic = state.get("research_topic", "").strip()
    runtime_config = dict(state.get("config", {}))

    response_text = await _request_planner_response(research_topic, runtime_config)
    tasks_payload = _extract_tasks(response_text)
    todo_items = _normalize_tasks(research_topic, tasks_payload)
    if not todo_items:
        todo_items = [_fallback_task(research_topic)]

    return {
        "todo_items": todo_items,
        "agent_role": _build_agent_role(research_topic),
    }
