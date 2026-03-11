from __future__ import annotations

import re
from typing import Any

from prompts import report_writer_instructions
from services.text_processing import strip_tool_calls
from utils import strip_thinking_tokens, with_llm_retry

from .planner import _resolve_client, _resolve_model_config


_MAX_CONTEXT_PER_TASK = 8000
_MAX_EVIDENCE_SNIPPET_CHARS = 1000
_MAX_PREVIOUS_REPORT_HEADINGS = 20
_MAX_PREVIOUS_REPORT_FALLBACK_CHARS = 2000


def _sanitize_writer_prompt() -> str:
    prompt = report_writer_instructions
    prompt = re.sub(r"\n<NOTES>.*?</NOTES>\n?", "\n", prompt, flags=re.DOTALL)
    prompt = re.sub(r"\[TOOL_CALL:[^\]]+\]", "", prompt)
    return prompt.strip()


def _build_task_context_block(state: dict[str, Any]) -> str:
    task_meta = {
        int(item["id"]): item
        for item in state.get("todo_items", [])
        if isinstance(item, dict) and isinstance(item.get("id"), int)
    }

    blocks: list[str] = []
    for item in state.get("research_data", []):
        if not isinstance(item, dict):
            continue
        task_id = item.get("task_id")
        task = task_meta.get(task_id, {})
        title = str(task.get("title") or item.get("topic") or f"任务 {task_id}").strip()
        intent = str(task.get("intent") or "").strip()
        query = str(task.get("query") or "").strip()
        summary = str(item.get("summary") or task.get("summary") or "暂无可用信息").strip()
        context = str(item.get("context") or "").strip()
        sources_summary = str(
            item.get("sources_summary") or task.get("sources_summary") or "暂无来源"
        ).strip()
        notices = item.get("notices") or []
        notices_block = "\n".join(f"- {str(notice).strip()}" for notice in notices if str(notice).strip())

        block = [
            f"### 任务 {task_id}: {title}",
            f"- 任务目标：{intent or '暂无相关信息'}",
            f"- 检索查询：{query or '暂无相关信息'}",
            f"- 任务总结：\n{summary}",
            f"- 来源概览：\n{sources_summary}",
        ]
        if notices_block:
            block.append(f"- 系统提示：\n{notices_block}")
        if context:
            block.append(f"- 原始上下文：\n{context[:_MAX_CONTEXT_PER_TASK]}")
        blocks.append("\n".join(block))

    return "\n\n".join(blocks).strip()


def _build_evidence_block(evidence_store: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for item in evidence_store:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        if not url:
            continue
        title = str(item.get("title") or url).strip()
        snippet = str(item.get("snippet") or "").strip()[:_MAX_EVIDENCE_SNIPPET_CHARS]
        score = item.get("relevance_score", 0.0)
        lines.append(
            f"- 任务 {item.get('task_id')} | {title} | {url} | score={score}\n"
            f"  摘要：{snippet or '暂无摘要'}"
        )
    return "\n".join(lines).strip()


def _build_review_block(review_result: dict[str, Any], previous_report: str) -> str:
    if not review_result and not previous_report.strip():
        return ""

    feedback = str(review_result.get("feedback") or "").strip()
    weak_sections = review_result.get("weak_sections") or []
    missing_topics = review_result.get("missing_topics") or []

    parts = ["## 审查反馈"]
    if feedback:
        parts.append(f"总体反馈：\n{feedback}")
    if weak_sections:
        parts.append(
            "需加强章节：\n" + "\n".join(f"- {str(section).strip()}" for section in weak_sections if str(section).strip())
        )
    if missing_topics:
        parts.append(
            "补研主题：\n" + "\n".join(f"- {str(topic).strip()}" for topic in missing_topics if str(topic).strip())
        )
    if previous_report.strip():
        headings = re.findall(r"^#{1,3}\s+.+$", previous_report, flags=re.MULTILINE)
        if headings:
            parts.append(
                "上一版报告结构：\n"
                + "\n".join(headings[:_MAX_PREVIOUS_REPORT_HEADINGS])
            )
        else:
            parts.append(
                "上一版报告摘要：\n"
                + previous_report.strip()[:_MAX_PREVIOUS_REPORT_FALLBACK_CHARS]
            )
    return "\n\n".join(parts).strip()


def _build_writer_user_prompt(state: dict[str, Any]) -> str:
    research_topic = str(state.get("research_topic") or "").strip()
    task_block = _build_task_context_block(state)
    evidence_block = _build_evidence_block(state.get("evidence_store", []))
    review_block = _build_review_block(
        state.get("review_result", {}),
        str(state.get("structured_report") or ""),
    )

    prompt_parts = [
        f"研究主题：{research_topic}",
        "## 任务研究结果",
        task_block or "暂无可用任务结果",
        "## 证据库",
        evidence_block or "暂无可用证据",
        (
            "请基于上述任务研究结果与证据库，输出一份完整的 Markdown 研究报告。"
            "要求保留原 prompt 的结构化风格，并在正文中尽量使用可追溯的来源标题或链接。"
        ),
    ]
    if review_block:
        prompt_parts.append(review_block)
        prompt_parts.append("请在保留有价值内容的前提下，针对审查反馈重写或增强报告。")
    return "\n\n".join(part for part in prompt_parts if part).strip()


def _build_references_section(evidence_store: list[dict[str, Any]]) -> str:
    seen: set[str] = set()
    references: list[str] = []
    for item in evidence_store:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        title = str(item.get("title") or url).strip()
        references.append(f"- [{title}]({url})")

    if not references:
        return ""
    return "## 参考来源\n" + "\n".join(references)


def _ensure_references(report: str, evidence_store: list[dict[str, Any]]) -> str:
    references = _build_references_section(evidence_store)
    if not references:
        return report.strip()

    lowered = report.lower()
    if "## 参考来源" in report or "## references" in lowered:
        return report.strip()

    if not report.strip():
        return references
    return f"{report.strip()}\n\n{references}"


async def writer_node(state: dict[str, Any]) -> dict[str, Any]:
    """Generate the full markdown report from aggregated task outputs."""
    config, provider, model = _resolve_model_config(
        dict(state.get("config", {})),
        selector_key="smart_llm",
    )
    client = _resolve_client(config, provider)
    messages = [
        {"role": "system", "content": _sanitize_writer_prompt()},
        {"role": "user", "content": _build_writer_user_prompt(state)},
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
    report = _ensure_references(content or "报告生成失败，请检查输入。", state.get("evidence_store", []))
    return {"structured_report": report}
