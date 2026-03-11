"""Standalone search dispatch helpers."""

from __future__ import annotations

import os
import logging
import threading
from html import unescape
from typing import Any, Callable, Optional, Tuple

import requests
from bs4 import BeautifulSoup

from config import Configuration
from utils import (
    deduplicate_and_format_sources,
    format_sources,
    get_config_value,
)

logger = logging.getLogger(__name__)

MAX_RESULTS = 8
MAX_TOKENS_PER_SOURCE = 4000
REQUEST_TIMEOUT = 15
_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    )
}
_DDGS_CLIENT = None
_DDGS_LOCK = threading.Lock()
_TAVILY_CLIENT = None
_TAVILY_LOCK = threading.Lock()


def _normalize_result(
    *,
    title: str | None,
    url: str | None,
    snippet: str | None,
    content: str | None = None,
    raw_content: str | None = None,
) -> dict[str, Any]:
    snippet_text = str(snippet or "").strip()
    content_text = str(content or snippet_text).strip()
    raw_text = str(raw_content or content_text or snippet_text).strip()
    return {
        "title": str(title or url or "Untitled Source").strip(),
        "url": str(url or "").strip(),
        "snippet": snippet_text,
        "content": content_text,
        "raw_content": raw_text,
    }


def _dedupe_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for result in results:
        url = str(result.get("url") or "").strip()
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        deduped.append(result)
        if len(deduped) >= MAX_RESULTS:
            break
    return deduped


def _fetch_page_text(url: str, char_limit: int) -> str:
    response = requests.get(url, headers=_DEFAULT_HEADERS, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "").lower()
    if "text/html" not in content_type:
        text = response.text
    else:
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = " ".join(segment.strip() for segment in soup.stripped_strings)

    text = unescape(text).strip()
    if len(text) > char_limit:
        return f"{text[:char_limit]}... [truncated]"
    return text


def _fill_page_content(results: list[dict[str, Any]], fetch_full_page: bool) -> list[dict[str, Any]]:
    if not fetch_full_page:
        return results

    char_limit = MAX_TOKENS_PER_SOURCE * 4
    enriched: list[dict[str, Any]] = []
    for result in results:
        item = dict(result)
        url = str(item.get("url") or "").strip()
        if not url:
            enriched.append(item)
            continue
        try:
            raw_content = _fetch_page_text(url, char_limit)
        except Exception as exc:  # pragma: no cover - network variability
            logger.info("Failed to fetch full page for %s: %s", url, exc)
            raw_content = str(item.get("raw_content") or item.get("content") or item.get("snippet") or "")
        item["raw_content"] = raw_content
        item["content"] = str(item.get("content") or item.get("snippet") or raw_content)
        enriched.append(item)
    return enriched


def _get_ddgs_client():
    global _DDGS_CLIENT
    if _DDGS_CLIENT is not None:
        return _DDGS_CLIENT

    with _DDGS_LOCK:
        if _DDGS_CLIENT is not None:
            return _DDGS_CLIENT
        from ddgs import DDGS

        _DDGS_CLIENT = DDGS()
    return _DDGS_CLIENT


def _get_tavily_client():
    global _TAVILY_CLIENT
    if _TAVILY_CLIENT is not None:
        return _TAVILY_CLIENT

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is required when SEARCH_API=tavily.")

    with _TAVILY_LOCK:
        if _TAVILY_CLIENT is not None:
            return _TAVILY_CLIENT
        from tavily import TavilyClient

        _TAVILY_CLIENT = TavilyClient(api_key=api_key)
    return _TAVILY_CLIENT


def _search_duckduckgo(query: str, config: Configuration) -> dict[str, Any]:
    client = _get_ddgs_client()
    raw_results = list(client.text(query, max_results=MAX_RESULTS))
    results = [
        _normalize_result(
            title=item.get("title"),
            url=item.get("href"),
            snippet=item.get("body"),
        )
        for item in raw_results
        if isinstance(item, dict)
    ]
    return {
        "results": _fill_page_content(_dedupe_results(results), config.fetch_full_page),
        "backend": "duckduckgo",
        "answer": None,
        "notices": [],
    }


def _search_tavily(query: str, config: Configuration) -> dict[str, Any]:
    client = _get_tavily_client()
    payload = client.search(
        query=query,
        max_results=MAX_RESULTS,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=config.fetch_full_page,
    )
    results = [
        _normalize_result(
            title=item.get("title"),
            url=item.get("url"),
            snippet=item.get("content"),
            content=item.get("content"),
            raw_content=item.get("raw_content"),
        )
        for item in payload.get("results", [])
        if isinstance(item, dict)
    ]
    if config.fetch_full_page:
        results = _fill_page_content(_dedupe_results(results), True)
    else:
        results = _dedupe_results(results)
    return {
        "results": results,
        "backend": "tavily",
        "answer": payload.get("answer"),
        "notices": [],
    }


def _search_searxng(query: str, config: Configuration) -> dict[str, Any]:
    base_url = os.getenv("SEARXNG_URL", "http://localhost:8888").rstrip("/")
    response = requests.get(
        f"{base_url}/search",
        params={"q": query, "format": "json"},
        headers=_DEFAULT_HEADERS,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    results = [
        _normalize_result(
            title=item.get("title"),
            url=item.get("url"),
            snippet=item.get("content") or item.get("snippet"),
        )
        for item in payload.get("results", [])[:MAX_RESULTS]
        if isinstance(item, dict)
    ]
    return {
        "results": _fill_page_content(_dedupe_results(results), config.fetch_full_page),
        "backend": "searxng",
        "answer": None,
        "notices": [],
    }


def _search_perplexity(query: str, config: Configuration) -> dict[str, Any]:
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        raise RuntimeError("PERPLEXITY_API_KEY is required when SEARCH_API=perplexity.")

    response = requests.post(
        "https://api.perplexity.ai/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": os.getenv("PERPLEXITY_MODEL", "sonar"),
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a web research assistant. Answer the user after consulting current web information.",
                },
                {"role": "user", "content": query},
            ],
        },
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    answer = (
        ((payload.get("choices") or [{}])[0].get("message") or {}).get("content")
        or ""
    )
    results = [
        _normalize_result(
            title=str(url),
            url=str(url),
            snippet=str(answer)[:800],
            content=str(answer),
            raw_content=str(answer),
        )
        for url in payload.get("citations", [])[:MAX_RESULTS]
    ]
    if config.fetch_full_page:
        results = _fill_page_content(_dedupe_results(results), True)
    else:
        results = _dedupe_results(results)
    return {
        "results": results,
        "backend": "perplexity",
        "answer": str(answer).strip() or None,
        "notices": [],
    }


def _search_advanced(query: str, config: Configuration) -> dict[str, Any]:
    backends: list[tuple[str, Callable[[str, Configuration], dict[str, Any]]]] = []
    if os.getenv("TAVILY_API_KEY"):
        backends.append(("tavily", _search_tavily))
    backends.append(("duckduckgo", _search_duckduckgo))
    if os.getenv("SEARXNG_URL"):
        backends.append(("searxng", _search_searxng))

    merged_results: list[dict[str, Any]] = []
    notices: list[str] = []
    answer_text: Optional[str] = None
    used_backends: list[str] = []

    for backend_name, search_fn in backends:
        try:
            payload = search_fn(query, config)
        except Exception as exc:  # pragma: no cover - external backends vary
            notices.append(f"{backend_name} 搜索失败：{exc}")
            continue

        used_backends.append(backend_name)
        if not answer_text:
            answer = payload.get("answer")
            answer_text = str(answer).strip() if answer else None
        merged_results.extend(payload.get("results", []))
        merged_results = _dedupe_results(merged_results)
        if len(merged_results) >= MAX_RESULTS:
            break

    if not used_backends:
        raise RuntimeError("No standalone search backend is available for SEARCH_API=advanced.")

    return {
        "results": merged_results[:MAX_RESULTS],
        "backend": f"advanced:{'+'.join(used_backends)}",
        "answer": answer_text,
        "notices": notices,
    }


def dispatch_search(
    query: str,
    config: Configuration,
    loop_count: int,
) -> Tuple[dict[str, Any] | None, list[str], Optional[str], str]:
    """Execute configured search backend and normalise response payload."""

    search_api = str(get_config_value(config.search_api)).strip().lower()

    try:
        if search_api == "duckduckgo":
            raw_response = _search_duckduckgo(query, config)
        elif search_api == "tavily":
            raw_response = _search_tavily(query, config)
        elif search_api == "perplexity":
            raw_response = _search_perplexity(query, config)
        elif search_api == "searxng":
            raw_response = _search_searxng(query, config)
        elif search_api == "advanced":
            raw_response = _search_advanced(query, config)
        else:
            raise ValueError(f"Unsupported search backend: {search_api}")
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Search backend %s failed at loop %s: %s", search_api, loop_count, exc)
        raise

    if isinstance(raw_response, str):
        notices = [raw_response]
        logger.warning("Search backend %s returned text notice: %s", search_api, raw_response)
        payload: dict[str, Any] = {
            "results": [],
            "backend": search_api,
            "answer": None,
            "notices": notices,
        }
    else:
        payload = raw_response
        notices = list(payload.get("notices") or [])

    backend_label = str(payload.get("backend") or search_api)
    answer_text = payload.get("answer")
    results = payload.get("results", [])

    if notices:
        for notice in notices:
            logger.info("Search notice (%s): %s", backend_label, notice)

    logger.info(
        "Search backend=%s resolved_backend=%s answer=%s results=%s",
        search_api,
        backend_label,
        bool(answer_text),
        len(results),
    )

    return payload, notices, answer_text, backend_label


def prepare_research_context(
    search_result: dict[str, Any] | None,
    answer_text: Optional[str],
    config: Configuration,
) -> tuple[str, str]:
    """Build structured context and source summary for downstream agents."""

    sources_summary = format_sources(search_result)
    context = deduplicate_and_format_sources(
        search_result or {"results": []},
        max_tokens_per_source=MAX_TOKENS_PER_SOURCE,
        fetch_full_page=config.fetch_full_page,
    )

    if answer_text:
        context = f"AI直接答案：\n{answer_text}\n\n{context}"

    return sources_summary, context
