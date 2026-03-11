from __future__ import annotations

import asyncio
import json
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

try:
    import json_repair
except ImportError:  # pragma: no cover - optional dependency
    json_repair = None


_SOURCE_LIMIT = 8
_FAST_PATH_CHAR_THRESHOLD = 16000
_DEFAULT_SIMILARITY_THRESHOLD = 0.42
_DEFAULT_EMBEDDING_SELECTOR = "openai:text-embedding-3-small"
_DEFAULT_COVERAGE_THRESHOLD = 0.75
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


def _load_json(candidate: str) -> Any:
    if json_repair is not None:
        try:
            return json_repair.loads(candidate)
        except Exception:
            pass
    return json.loads(candidate)


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


async def _extract_claims(
    task_id: int,
    summary: str,
    evidence_items: list[dict[str, Any]],
    runtime_config: dict[str, Any],
) -> list[dict[str, Any]]:
    if not summary.strip() or not evidence_items:
        return list(evidence_items)

    config, provider, model = _resolve_model_config(runtime_config, selector_key="smart_llm")
    client = _resolve_client(config, provider)
    evidence_block = "\n".join(
        (
            f"- URL: {str(item.get('url') or '').strip()}\n"
            f"  标题: {str(item.get('title') or '').strip()}\n"
            f"  摘要: {str(item.get('snippet') or '').strip()[:500] or '暂无摘要'}"
        )
        for item in evidence_items
        if str(item.get("url") or "").strip()
    )
    prompt = (
        f"任务ID：{task_id}\n\n"
        f"任务摘要：\n{summary[:4000]}\n\n"
        f"可用证据列表：\n{evidence_block or '暂无证据'}\n\n"
        "请从任务摘要中提取关键论断，并为每条论断匹配最合适的证据来源。"
        "返回 JSON 数组，每项格式如下：\n"
        '[{"claim_text": "论断内容", "evidence_url": "对应URL", "support_type": "support", "section_hint": "建议章节"}]\n'
        "support_type 只能是 support、contradict、background 之一。"
        "只返回 JSON，不要解释。"
    )

    try:
        response = await with_llm_retry(
            lambda: client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一名信息提取专家，擅长从研究摘要中识别关键论断并将其与来源证据对应。",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
        )
        content = response.choices[0].message.content or ""
        content = strip_tool_calls(strip_thinking_tokens(content)).strip()

        candidates: list[str] = []
        fenced_match = re.search(r"```(?:json)?\s*(.*?)```", content, re.IGNORECASE | re.DOTALL)
        if fenced_match:
            candidates.append(fenced_match.group(1).strip())
        if content:
            candidates.append(content)
        array_start = content.find("[")
        array_end = content.rfind("]")
        if array_start != -1 and array_end != -1 and array_end > array_start:
            candidates.append(content[array_start : array_end + 1])
        object_start = content.find("{")
        object_end = content.rfind("}")
        if object_start != -1 and object_end != -1 and object_end > object_start:
            candidates.append(content[object_start : object_end + 1])

        parsed: Any = None
        for candidate in candidates:
            try:
                parsed = _load_json(candidate)
                break
            except Exception:
                continue

        if isinstance(parsed, dict):
            parsed = parsed.get("claims") or parsed.get("items") or []
        if not isinstance(parsed, list):
            return list(evidence_items)

        updated_items = [dict(item) for item in evidence_items]
        index_by_url = {
            str(item.get("url") or "").strip(): index
            for index, item in enumerate(updated_items)
            if str(item.get("url") or "").strip()
        }

        for item in parsed:
            if not isinstance(item, dict):
                continue
            evidence_url = str(item.get("evidence_url") or item.get("url") or "").strip()
            claim_text = str(item.get("claim_text") or "").strip()
            if not evidence_url or not claim_text:
                continue
            item_index = index_by_url.get(evidence_url)
            if item_index is None:
                continue
            if str(updated_items[item_index].get("claim_text") or "").strip():
                continue

            support_type = str(item.get("support_type") or "support").strip().lower()
            if support_type not in {"support", "contradict", "background"}:
                support_type = "support"
            section_hint = str(item.get("section_hint") or "").strip()
            updated_items[item_index]["claim_text"] = claim_text
            updated_items[item_index]["support_type"] = support_type
            if section_hint:
                updated_items[item_index]["section_hint"] = section_hint

        return updated_items
    except Exception as exc:
        logger.debug("Claim extraction failed: %s", exc)
        return list(evidence_items)


async def _generate_followup_queries(
    research_topic: str,
    task: dict[str, Any],
    initial_context: str,
    runtime_config: dict[str, Any],
    num_queries: int = 2,
) -> list[str]:
    # NOTE: 保留备用，当前已由 _assess_coverage + _rewrite_query 的迭代闭环替代。
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


async def _assess_coverage(
    task: dict[str, Any],
    context: str,
    runtime_config: dict[str, Any],
    coverage_threshold: float = _DEFAULT_COVERAGE_THRESHOLD,
) -> dict[str, Any]:
    fallback = {
        "coverage_score": 0.0,
        "is_sufficient": False,
        "unresolved_questions": [],
        "reasoning": "解析失败",
    }
    normalized_threshold = max(0.0, min(1.0, coverage_threshold))
    threshold_text = f"{normalized_threshold:.2f}".rstrip("0").rstrip(".")
    config, provider, model = _resolve_model_config(runtime_config, selector_key="smart_llm")
    client = _resolve_client(config, provider)
    prompt = (
        f"研究任务：{task.get('title', '')}\n"
        f"任务目标：{task.get('intent', '')}\n"
        f"原始查询：{task.get('query', '')}\n\n"
        f"当前已收集的上下文：\n{context[:3000]}\n\n"
        "请评估当前信息的覆盖度，并以 JSON 格式返回：\n"
        "{\n"
        '  "coverage_score": 0.0到1.0之间的浮点数,\n'
        '  "is_sufficient": true或false,\n'
        '  "unresolved_questions": ["还未覆盖的具体问题1", "问题2"],\n'
        '  "reasoning": "简短说明"\n'
        "}\n\n"
        f"只返回 JSON，不要其他内容。coverage_score >= {threshold_text} 时 is_sufficient 才能为 true。"
    )

    try:
        response = await with_llm_retry(
            lambda: client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一名严格的研究质量评估员。你的任务是判断当前收集到的信息是否足以完成指定的研究任务。",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
        )
        content = response.choices[0].message.content or ""
        content = strip_tool_calls(strip_thinking_tokens(content)).strip()
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            return fallback

        payload = _load_json(match.group(0))
        try:
            coverage_score = float(payload.get("coverage_score", 0.0) or 0.0)
        except (TypeError, ValueError):
            coverage_score = 0.0
        coverage_score = max(0.0, min(1.0, coverage_score))

        raw_is_sufficient = payload.get("is_sufficient")
        if isinstance(raw_is_sufficient, bool):
            is_sufficient = raw_is_sufficient
        else:
            is_sufficient = str(raw_is_sufficient).strip().lower() == "true"
        is_sufficient = is_sufficient and coverage_score >= normalized_threshold

        unresolved_questions = payload.get("unresolved_questions")
        if not isinstance(unresolved_questions, list):
            unresolved_questions = []
        unresolved_questions = [
            str(item).strip()
            for item in unresolved_questions
            if str(item).strip()
        ]

        reasoning = str(payload.get("reasoning") or "").strip() or fallback["reasoning"]
        return {
            "coverage_score": coverage_score,
            "is_sufficient": is_sufficient,
            "unresolved_questions": unresolved_questions,
            "reasoning": reasoning,
        }
    except Exception as exc:
        logger.warning("Failed to assess coverage: %s", exc)
        return fallback


async def _rewrite_query(
    task: dict[str, Any],
    unresolved_questions: list[str],
    tried_queries: list[str],
    runtime_config: dict[str, Any],
) -> str:
    if not unresolved_questions:
        return ""

    config, provider, model = _resolve_model_config(runtime_config, selector_key="smart_llm")
    client = _resolve_client(config, provider)
    tried_query_lines = "\n".join(f"- {query}" for query in tried_queries) or "- 无"
    unresolved_lines = "\n".join(f"- {question}" for question in unresolved_questions) or "- 无"
    prompt = (
        f"研究任务：{task.get('title', '')}\n"
        f"任务目标：{task.get('intent', '')}\n\n"
        f"已尝试的查询（请勿重复）：\n{tried_query_lines}\n\n"
        f"当前未覆盖的问题：\n{unresolved_lines}\n\n"
        "请生成一个新的、更有针对性的搜索查询，直接输出查询字符串，不要解释，不要引号。"
    )

    try:
        response = await with_llm_retry(
            lambda: client.chat.completions.create(
                model=model,
                temperature=0.2,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一名专业研究员，擅长设计精准的搜索查询以填补研究空白。",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
        )
        content = response.choices[0].message.content or ""
        content = strip_tool_calls(strip_thinking_tokens(content)).strip()
        query = content.splitlines()[0].strip() if content else ""
        query = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", query)
        query = query.strip().strip("\"'").strip()
        query = re.sub(r"\s+", " ", query)
        return query
    except Exception as exc:
        logger.warning("Failed to rewrite query: %s", exc)
        return ""


async def task_node(state: dict[str, Any]) -> dict[str, Any]:
    """Execute a single planned task via iterative search, coverage checks, and summarization."""
    task = dict(state.get("task") or {})
    runtime_config = dict(state.get("config", {}))
    research_topic = str(state.get("research_topic", "")).strip()
    visited_urls = set(state.get("visited_urls", set()))
    loop_count = int(state.get("research_loop_count", 0))

    config = Configuration.from_env(overrides=runtime_config)
    try:
        task_search_budget = int(task.get("search_budget") or 0)
    except (TypeError, ValueError):
        task_search_budget = 0
    try:
        max_iterations = max(
            1,
            int(
                task_search_budget
                or runtime_config.get("researcher_max_iterations")
                or config.researcher_max_iterations
                or 1
            ),
        )
    except (TypeError, ValueError):
        max_iterations = max(1, int(config.researcher_max_iterations or 1))
    try:
        coverage_threshold = float(
            runtime_config.get("researcher_coverage_threshold")
            or config.researcher_coverage_threshold
            or _DEFAULT_COVERAGE_THRESHOLD
        )
    except (TypeError, ValueError):
        coverage_threshold = _DEFAULT_COVERAGE_THRESHOLD
    coverage_threshold = max(0.0, min(1.0, coverage_threshold))

    current_query = str(task.get("query") or research_topic).strip()
    tried_queries = [current_query] if current_query else []
    all_filtered_results: list[dict[str, Any]] = []
    all_notices: list[str] = []
    updated_urls = set(visited_urls)
    context = ""
    backend_label = ""
    first_answer_text: str | None = None

    for iteration in range(max_iterations):
        try:
            search_result, notices, answer_text, current_backend_label = await asyncio.to_thread(
                dispatch_search,
                current_query,
                config,
                loop_count,
            )
        except Exception as exc:
            if iteration == 0:
                raise
            logger.warning("Iterative search failed for query '%s': %s", current_query, exc)
            break

        _, updated_urls, filtered_results = _filter_new_results(search_result, updated_urls)
        all_filtered_results.extend(filtered_results)
        all_notices.extend(list(notices or []))
        if current_backend_label:
            backend_label = current_backend_label
        if answer_text and not first_answer_text:
            first_answer_text = answer_text

        pages = _normalize_pages(all_filtered_results)
        compressed_context = (
            await _compress_context(current_query, runtime_config, pages) if pages else ""
        )
        context = compressed_context
        if not context:
            _, fallback_context = prepare_research_context(
                {"results": all_filtered_results},
                None,
                config,
            )
            context = fallback_context
        context = _prepend_answer_text(context, first_answer_text)

        if iteration == max_iterations - 1:
            break

        assessment = await _assess_coverage(
            task,
            context,
            runtime_config,
            coverage_threshold=coverage_threshold,
        )
        try:
            coverage_score = float(assessment.get("coverage_score", 0.0) or 0.0)
        except (TypeError, ValueError):
            coverage_score = 0.0

        if assessment.get("is_sufficient") or coverage_score >= coverage_threshold:
            break

        unresolved_questions = assessment.get("unresolved_questions") or []
        unresolved_questions = [
            str(item).strip()
            for item in unresolved_questions
            if str(item).strip()
        ]
        if not unresolved_questions:
            break

        new_query = await _rewrite_query(
            task=task,
            unresolved_questions=unresolved_questions,
            tried_queries=tried_queries,
            runtime_config=runtime_config,
        )
        if not new_query or new_query in tried_queries:
            break

        tried_queries.append(new_query)
        current_query = new_query

    merged_search_result = {"results": all_filtered_results}
    sources_summary = format_sources(merged_search_result)
    if not context:
        _, fallback_context = prepare_research_context(merged_search_result, None, config)
        context = fallback_context
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
    evidence_items = await _extract_claims(
        task_id=updated_task["id"],
        summary=summary,
        evidence_items=evidence_items,
        runtime_config=runtime_config,
    )
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
