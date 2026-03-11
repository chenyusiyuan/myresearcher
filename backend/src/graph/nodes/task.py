from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any

from config import Configuration
from prompts import task_summarizer_instructions
from research_support.compression import ContextCompressor
from research_support.embeddings import Memory
from research_support.prompts import PromptFamily
from services.search import dispatch_search, prepare_research_context
from services.text_processing import strip_tool_calls
from utils import format_sources, strip_thinking_tokens, with_llm_retry

from .planner import _resolve_client, _resolve_model_config


_SOURCE_LIMIT = 8
_FAST_PATH_CHAR_THRESHOLD = 16000
_DEFAULT_SIMILARITY_THRESHOLD = 0.42
_DEFAULT_EMBEDDING_SELECTOR = "openai:text-embedding-3-small"
logger = logging.getLogger(__name__)


def _sanitize_task_prompt() -> str:
    prompt = task_summarizer_instructions
    prompt = re.sub(r"\n<NOTES>.*?</NOTES>\n?", "\n", prompt, flags=re.DOTALL)
    prompt = re.sub(r"\[TOOL_CALL:[^\]]+\]", "", prompt)
    return prompt.strip()


def _build_summary_prompt(
    research_topic: str,
    task: dict[str, Any],
    context: str,
    sources_summary: str,
) -> str:
    return (
        f"任务主题：{research_topic}\n"
        f"任务名称：{task.get('title', '')}\n"
        f"任务目标：{task.get('intent', '')}\n"
        f"检索查询：{task.get('query', '')}\n"
        f"来源概览：\n{sources_summary or '暂无来源'}\n\n"
        f"任务上下文：\n{context}\n\n"
        "请仅基于上述信息输出面向用户的 Markdown 任务总结。"
    )


def _result_url(result: dict[str, Any]) -> str:
    return str(result.get("url") or result.get("href") or "").strip()


def _result_title(result: dict[str, Any]) -> str:
    return str(result.get("title") or _result_url(result) or "Untitled Source").strip()


def _result_snippet(result: dict[str, Any]) -> str:
    snippet = (
        result.get("snippet")
        or result.get("content")
        or result.get("body")
        or result.get("raw_content")
        or ""
    )
    return str(snippet).strip()


def _result_score(result: dict[str, Any]) -> float:
    value = result.get("relevance_score", result.get("score", 0.0))
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _filter_new_results(
    search_result: dict[str, Any] | None,
    visited_urls: set[str],
) -> tuple[dict[str, Any], set[str], list[dict[str, Any]]]:
    payload = dict(search_result or {})
    results = payload.get("results") or []
    filtered_results: list[dict[str, Any]] = []
    updated_urls = set(visited_urls)

    for raw_result in results:
        if not isinstance(raw_result, dict):
            continue
        url = _result_url(raw_result)
        if url and url in updated_urls:
            continue
        if url:
            updated_urls.add(url)
        filtered_results.append(dict(raw_result))

    payload["results"] = filtered_results
    return payload, updated_urls, filtered_results


def _normalize_pages(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    for result in results:
        raw_content = (
            result.get("raw_content")
            or result.get("content")
            or result.get("body")
            or result.get("snippet")
            or ""
        )
        raw_text = str(raw_content).strip()
        if not raw_text:
            continue
        pages.append(
            {
                "url": _result_url(result),
                "title": _result_title(result),
                "raw_content": raw_text,
                "content": str(result.get("content") or result.get("body") or raw_text),
            }
        )
    return pages


def _format_fast_path_context(pages: list[dict[str, Any]]) -> str:
    return PromptFamily.pretty_print_docs(pages[:_SOURCE_LIMIT], top_n=_SOURCE_LIMIT)


def _resolve_embeddings(runtime_config: dict[str, Any]):
    config = Configuration.from_env(overrides=runtime_config)
    selector = str(
        runtime_config.get("embedding_model") or _DEFAULT_EMBEDDING_SELECTOR
    ).strip()
    if ":" in selector:
        provider, model = selector.split(":", 1)
    else:
        provider, model = "openai", selector

    provider = provider.strip() or "openai"
    model = model.strip() or "text-embedding-3-small"
    embedding_kwargs: dict[str, Any] = {}

    if provider == "openai":
        if os.getenv("OPENAI_API_KEY"):
            embedding_kwargs["openai_api_key"] = os.getenv("OPENAI_API_KEY")
        if os.getenv("OPENAI_BASE_URL"):
            embedding_kwargs["openai_api_base"] = os.getenv("OPENAI_BASE_URL")
    elif provider == "custom":
        if config.llm_api_key:
            embedding_kwargs["openai_api_key"] = config.llm_api_key
        if config.llm_base_url:
            embedding_kwargs["openai_api_base"] = config.llm_base_url
    elif provider == "ollama":
        embedding_kwargs["base_url"] = config.ollama_base_url

    return Memory(provider, model, **embedding_kwargs).get_embeddings()


async def _compress_context(query: str, runtime_config: dict[str, Any], pages: list[dict[str, Any]]) -> str:
    total_chars = sum(len(page.get("raw_content", "")) for page in pages)
    if total_chars < _FAST_PATH_CHAR_THRESHOLD:
        return _format_fast_path_context(pages)

    try:
        similarity_threshold = float(
            runtime_config.get("similarity_threshold", _DEFAULT_SIMILARITY_THRESHOLD)
        )
        compressor = ContextCompressor(
            documents=pages,
            embeddings=_resolve_embeddings(runtime_config),
            max_results=_SOURCE_LIMIT,
            similarity_threshold=similarity_threshold,
        )
        return await compressor.async_get_context(query=query, max_results=_SOURCE_LIMIT)
    except Exception as exc:  # pragma: no cover - fallback for missing embedding backends
        logger.warning("Context compression failed, falling back to fast-path context: %s", exc)
        return _format_fast_path_context(pages)


async def _generate_task_summary(
    research_topic: str,
    task: dict[str, Any],
    context: str,
    sources_summary: str,
    runtime_config: dict[str, Any],
) -> str:
    if not context.strip():
        return "暂无可用信息"

    config, provider, model = _resolve_model_config(runtime_config, selector_key="smart_llm")
    client = _resolve_client(config, provider)
    messages = [
        {"role": "system", "content": _sanitize_task_prompt()},
        {
            "role": "user",
            "content": _build_summary_prompt(
                research_topic=research_topic,
                task=task,
                context=context,
                sources_summary=sources_summary,
            ),
        },
    ]
    response = await with_llm_retry(
        lambda: client.chat.completions.create(
            model=model,
            temperature=0,
            messages=messages,
        )
    )
    content = response.choices[0].message.content or ""
    content = strip_tool_calls(strip_thinking_tokens(content)).strip()
    return content or "暂无可用信息"


def _prepend_answer_text(context: str, answer_text: str | None) -> str:
    if not answer_text:
        return context
    answer_block = f"AI直接答案：\n{answer_text.strip()}\n\n"
    if not context.strip():
        return answer_block.strip()
    return f"{answer_block}{context}".strip()


def _build_evidence_items(task_id: int, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "task_id": task_id,
            "url": _result_url(result),
            "title": _result_title(result),
            "snippet": _result_snippet(result),
            "relevance_score": _result_score(result),
        }
        for result in results
        if _result_url(result)
    ]


async def _generate_followup_queries(
    research_topic: str,
    task: dict[str, Any],
    initial_context: str,
    runtime_config: dict[str, Any],
    num_queries: int = 2,
) -> list[str]:
    """Ask the LLM what follow-up queries would deepen the research for this task."""
    if not initial_context.strip():
        return []

    config, provider, model = _resolve_model_config(runtime_config, selector_key="smart_llm")
    client = _resolve_client(config, provider)
    prompt = (
        f"研究主题：{research_topic}\n"
        f"任务名称：{task.get('title', '')}\n"
        f"任务目标：{task.get('intent', '')}\n"
        f"初始检索查询：{task.get('query', '')}\n\n"
        f"以下是初步检索到的内容摘要：\n{initial_context[:3000]}\n\n"
        f"请分析上述内容，找出还未被充分覆盖的角度或缺失的关键信息，"
        f"生成 {num_queries} 个更深入、更具体的后续检索查询。\n"
        f"直接输出查询列表，每行一条，不要编号或解释。"
    )
    try:
        response = await with_llm_retry(
            lambda: client.chat.completions.create(
                model=model,
                temperature=0.3,
                messages=[
                    {"role": "system", "content": "你是一名专业研究员，擅长设计精准的检索查询来深挖主题。"},
                    {"role": "user", "content": prompt},
                ],
            )
        )
        content = (response.choices[0].message.content or "").strip()
        queries = [line.strip() for line in content.splitlines() if line.strip()]
        return queries[:num_queries]
    except Exception as exc:
        logger.warning("Failed to generate follow-up queries: %s", exc)
        return []


async def task_node(state: dict[str, Any]) -> dict[str, Any]:
    """Execute a single planned task via iterative search, compression, and summarization."""
    task = dict(state.get("task") or {})
    runtime_config = dict(state.get("config", {}))
    research_topic = str(state.get("research_topic", "")).strip()
    visited_urls = set(state.get("visited_urls", set()))
    loop_count = int(state.get("research_loop_count", 0))

    config = Configuration.from_env(overrides=runtime_config)
    depth = max(1, int(runtime_config.get("deep_research_depth") or config.deep_research_depth or 1))

    # --- Round 1: initial search ---
    search_result, notices, answer_text, backend_label = await asyncio.to_thread(
        dispatch_search,
        str(task.get("query") or research_topic),
        config,
        loop_count,
    )
    filtered_search_result, updated_urls, filtered_results = _filter_new_results(
        search_result,
        visited_urls,
    )
    all_filtered_results = list(filtered_results)
    all_notices = list(notices)

    # --- Rounds 2..depth: generate follow-up queries and search again ---
    if depth > 1:
        initial_pages = _normalize_pages(filtered_results)
        initial_context_preview = _format_fast_path_context(initial_pages)
        followup_queries = await _generate_followup_queries(
            research_topic=research_topic,
            task=task,
            initial_context=initial_context_preview,
            runtime_config=runtime_config,
            num_queries=depth - 1,
        )
        for fq in followup_queries:
            try:
                fq_result, fq_notices, _, _ = await asyncio.to_thread(
                    dispatch_search,
                    fq,
                    config,
                    loop_count,
                )
                _, updated_urls, fq_filtered = _filter_new_results(fq_result, updated_urls)
                all_filtered_results.extend(fq_filtered)
                all_notices.extend(fq_notices)
            except Exception as exc:
                logger.warning("Follow-up search failed for query '%s': %s", fq, exc)

    # Build merged search result payload for sources_summary
    merged_search_result = {"results": all_filtered_results}
    sources_summary = format_sources(merged_search_result)
    pages = _normalize_pages(all_filtered_results)

    compressed_context = await _compress_context(
        str(task.get("query") or research_topic),
        runtime_config,
        pages,
    ) if pages else ""
    context = compressed_context
    if not context:
        _, fallback_context = prepare_research_context(
            filtered_search_result,
            None,
            config,
        )
        context = fallback_context
    context = _prepend_answer_text(context, answer_text)
    summary = await _generate_task_summary(
        research_topic=research_topic,
        task=task,
        context=context,
        sources_summary=sources_summary,
        runtime_config=runtime_config,
    )

    updated_task = {
        "id": int(task.get("id", 0)),
        "title": str(task.get("title") or "未命名任务"),
        "intent": str(task.get("intent") or ""),
        "query": str(task.get("query") or research_topic),
        "status": "completed",
        "summary": summary,
        "sources_summary": sources_summary or None,
    }

    evidence_items = _build_evidence_items(updated_task["id"], all_filtered_results)
    source_payload = [
        {"url": item["url"], "title": item["title"]}
        for item in evidence_items
    ]

    return {
        "research_data": [
            {
                "task_id": updated_task["id"],
                "topic": updated_task["title"],
                "context": context,
                "summary": summary,
                "sources": source_payload,
                "sources_summary": sources_summary,
                "backend": backend_label,
                "notices": all_notices,
            }
        ],
        "evidence_store": evidence_items,
        "visited_urls": updated_urls,
        "todo_items": [updated_task],
    }
