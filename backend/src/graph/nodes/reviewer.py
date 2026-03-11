from __future__ import annotations

import json
import re
from typing import Any

from utils import strip_thinking_tokens, with_llm_retry

from .planner import _resolve_client, _resolve_model_config

try:
    import json_repair
except ImportError:  # pragma: no cover - optional dependency
    json_repair = None


_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_MAX_REPORT_CHARS = 12000


def _normalize_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _normalize_score(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _normalize_priority(value: Any) -> str:
    priority = str(value or "").strip().lower()
    if priority in {"high", "medium", "low"}:
        return priority
    return "medium"


def _priority_to_rank(value: Any) -> int:
    priority = _normalize_priority(value)
    if priority == "high":
        return 1
    if priority == "low":
        return 3
    return 2


def _default_search_budget(state: dict[str, Any]) -> int:
    config = state.get("config", {})
    try:
        return max(1, int((config or {}).get("deep_research_depth") or 1))
    except (TypeError, ValueError):
        return 1


def _normalize_research_briefs(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []

    briefs: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        topic = str(item.get("topic") or "").strip()
        intent = str(item.get("intent") or "").strip()
        query = str(item.get("query") or "").strip()
        if not topic and not query:
            continue
        briefs.append(
            {
                "topic": topic or query,
                "intent": intent or f"补充研究：{topic or query}",
                "query": query or topic,
                "priority": _normalize_priority(item.get("priority")),
            }
        )
    return briefs


def _normalize_section_patch_plan(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []

    plan: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        section = str(item.get("section") or "").strip()
        issue = str(item.get("issue") or "").strip()
        instruction = str(item.get("instruction") or "").strip()
        if not section or not instruction:
            continue
        plan.append(
            {
                "section": section,
                "issue": issue or "该章节仍需增强结构、论证或表达。",
                "instruction": instruction,
            }
        )
    return plan


def _merge_missing_topics(
    missing_topics: list[str],
    research_briefs: list[dict[str, str]],
) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()

    def _append(value: str) -> None:
        text = str(value).strip()
        normalized = text.lower()
        if not text or normalized in seen:
            return
        seen.add(normalized)
        merged.append(text)

    for topic in missing_topics:
        _append(topic)
    for brief in research_briefs:
        _append(str(brief.get("topic") or brief.get("query") or ""))
    return merged


def _build_task_snapshot(state: dict[str, Any]) -> str:
    blocks: list[str] = []
    for item in state.get("todo_items", []):
        if not isinstance(item, dict):
            continue
        blocks.append(
            "\n".join(
                [
                    f"- 任务ID：{item.get('id')}",
                    f"  标题：{str(item.get('title') or '').strip()}",
                    f"  目标：{str(item.get('intent') or '').strip()}",
                    f"  查询：{str(item.get('query') or '').strip()}",
                    f"  状态：{str(item.get('status') or '').strip()}",
                    f"  摘要：{str(item.get('summary') or '暂无可用信息').strip()[:800]}",
                ]
            )
        )
    return "\n\n".join(blocks).strip()


def _build_evidence_snapshot(state: dict[str, Any]) -> str:
    evidence_by_task: dict[Any, list[dict[str, Any]]] = {}
    for item in state.get("evidence_store", []):
        if not isinstance(item, dict):
            continue
        evidence_by_task.setdefault(item.get("task_id"), []).append(item)

    lines: list[str] = []
    for task_id, evidence_items in evidence_by_task.items():
        urls = []
        claim_samples = []
        for evidence in evidence_items[:5]:
            url = str(evidence.get("url") or "").strip()
            if url:
                urls.append(url)
            claim_text = str(evidence.get("claim_text") or "").strip()
            if claim_text:
                claim_samples.append(claim_text[:120])
        lines.append(
            f"- 任务 {task_id}：证据 {len(evidence_items)} 条"
            + (f" | 来源：{', '.join(urls)}" if urls else "")
            + (f" | 论断示例：{'；'.join(claim_samples[:2])}" if claim_samples else "")
        )
    return "\n".join(lines).strip()


def _build_report_preview(state: dict[str, Any]) -> str:
    report = str(state.get("structured_report") or "").strip()
    if len(report) <= _MAX_REPORT_CHARS:
        return report
    return report[:_MAX_REPORT_CHARS] + "\n…（报告内容已截断以控制审查提示长度）"


def _build_review_output_example() -> str:
    example = {
        "approved": False,
        "score": 0.0,
        "feedback": "具体改进建议，区分需补搜索 vs 需改写的问题",
        "missing_topics": ["可直接用于搜索的查询词A", "查询词B"],
        "weak_sections": ["需改写的章节名"],
        "research_briefs": [
            {
                "topic": "补研主题",
                "intent": "补研目标",
                "query": "可直接搜索的查询词",
                "priority": "high",
            }
        ],
        "section_patch_plan": [
            {
                "section": "需改写章节",
                "issue": "该章节存在的问题",
                "instruction": "具体改写要求",
            }
        ],
    }
    return json.dumps(example, ensure_ascii=False, indent=2)


def _build_review_prompt(state: dict[str, Any]) -> str:
    output_example = _build_review_output_example()
    return f"""
你是一名严格的研究报告审稿人，请审查下面这份研究报告，并只返回 JSON。

研究主题：
{str(state.get("research_topic") or "").strip()}

任务快照：
{_build_task_snapshot(state) or "暂无任务信息"}

证据快照：
{_build_evidence_snapshot(state) or "暂无证据信息"}

当前报告：
{_build_report_preview(state)}

请从以下四个维度评估：
1. 证据充分性：结论是否有足够、具体、可追溯的证据支撑
2. 结构完整性：报告结构是否清晰、层次是否完整、重点是否突出
3. 主题覆盖度：是否覆盖了研究主题及所有关键子任务，是否存在缺失主题
4. 事实一致性：报告内容与已收集证据是否一致，是否存在明显不一致或过度推断

【关键判断规则——请严格遵守】

missing_topics（触发补充搜索）必须填写，当且仅当：
- 某个结论或章节缺乏具体数据、案例或来源支撑（证据不足）
- 报告某一维度完全没有被检索过（主题缺失）
- 报告中有明显的"暂无相关信息"或泛泛而谈、无据可查的部分
→ 此时请给出 1~3 个具体的、可直接用于搜索引擎的查询字符串（不是章节名，是检索词）

weak_sections（触发纯重写）仅在以下情况填写：
- 现有证据已经充分，但表述混乱、逻辑不清、结构不合理
- 章节已有足够数据但分析浅薄、缺乏洞见（不是缺数据，是缺分析）
→ 只填章节名，不填检索词

research_briefs（结构化补研单）：
- 当 missing_topics 非空时，尽量给出 1~3 个结构化补研单
- 每项必须包含 topic / intent / query / priority
- priority 只能是 high、medium、low

section_patch_plan（定向改写计划）：
- 当 weak_sections 非空时，尽量给出每个章节的改写计划
- 每项必须包含 section / issue / instruction
- instruction 要足够具体，能够直接指导改写

【重要】：
- 如果一个章节既缺证据又写得差，同时填入 missing_topics（补搜索）和 weak_sections（待重写）
- 不要把证据不足的问题放进 weak_sections 了事——那样系统只会用原有数据重写，毫无意义
- feedback 必须区分"哪些问题需要新的检索"和"哪些问题只需改写"

若报告已达到可交付标准，approved 设为 true，missing_topics、weak_sections、research_briefs、section_patch_plan 均为空数组。
只返回 JSON，不要输出任何额外解释。

返回格式：
{output_example}
""".strip()


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

    for candidate in candidates:
        try:
            return _load_json(candidate)
        except Exception:
            continue
    return {}


def _build_missing_topic_tasks(state: dict[str, Any], missing_topics: list[str]) -> list[dict[str, Any]]:
    existing_items = [item for item in state.get("todo_items", []) if isinstance(item, dict)]
    research_briefs = _normalize_research_briefs(
        state.get("review_result", {}).get("research_briefs", [])
    )
    default_search_budget = _default_search_budget(state)
    existing_ids = [
        int(item["id"])
        for item in existing_items
        if str(item.get("id", "")).isdigit()
    ]
    next_id = max(existing_ids, default=0) + 1

    existing_keys = {
        str(item.get("title") or "").strip().lower()
        for item in existing_items
        if str(item.get("title") or "").strip()
    }
    existing_keys.update(
        str(item.get("query") or "").strip().lower()
        for item in existing_items
        if str(item.get("query") or "").strip()
    )

    tasks: list[dict[str, Any]] = []
    if research_briefs:
        for offset, brief in enumerate(research_briefs):
            title = str(brief.get("topic") or brief.get("query") or "").strip()
            intent = str(brief.get("intent") or "").strip()
            query = str(brief.get("query") or title).strip()
            normalized_keys = {
                value.lower()
                for value in (title, query)
                if value
            }
            if not title or not query or normalized_keys.intersection(existing_keys):
                continue
            existing_keys.update(normalized_keys)
            tasks.append(
                {
                    "id": next_id + len(tasks),
                    "title": title,
                    "intent": intent or f"补充研究该缺失主题：{title}",
                    "query": query,
                    "status": "pending",
                    "summary": None,
                    "sources_summary": None,
                    "priority": _priority_to_rank(brief.get("priority")),
                    "depends_on": [],
                    "search_budget": default_search_budget,
                    "search_type": "search",
                }
            )
        if tasks:
            return tasks

    for offset, topic in enumerate(missing_topics):
        topic_text = str(topic).strip()
        normalized = topic_text.lower()
        if not topic_text or normalized in existing_keys:
            continue
        existing_keys.add(normalized)
        tasks.append(
            {
                "id": next_id + len(tasks),
                "title": topic_text,
                "intent": f"补充研究该缺失主题：{topic_text}",
                "query": topic_text,
                "status": "pending",
                "summary": None,
                "sources_summary": None,
                "priority": 2,
                "depends_on": [],
                "search_budget": default_search_budget,
                "search_type": "search",
            }
        )
    return tasks


async def reviewer_node(state: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the generated report and request rewrite or supplemental research."""
    config, provider, model = _resolve_model_config(
        dict(state.get("config", {})),
        selector_key="strategic_llm",
    )
    client = _resolve_client(config, provider)
    messages = [
        {"role": "system", "content": "你是一名严谨、苛刻、重证据的研究报告审稿人。"},
        {"role": "user", "content": _build_review_prompt(state)},
    ]
    response = await with_llm_retry(
        lambda: client.chat.completions.create(
            model=model,
            temperature=0,
            messages=messages,
        )
    )
    content = strip_thinking_tokens(response.choices[0].message.content or "").strip()
    parsed = _extract_json_payload(content)
    research_briefs = _normalize_research_briefs(parsed.get("research_briefs", []))
    section_patch_plan = _normalize_section_patch_plan(parsed.get("section_patch_plan", []))
    missing_topics = _merge_missing_topics(
        _normalize_list(parsed.get("missing_topics", [])),
        research_briefs,
    )

    review_result = {
        "approved": bool(parsed.get("approved", False)),
        "score": _normalize_score(parsed.get("score", 0.0)),
        "feedback": str(parsed.get("feedback") or "").strip(),
        "missing_topics": missing_topics,
        "weak_sections": _normalize_list(parsed.get("weak_sections", [])),
        "research_briefs": research_briefs,
        "section_patch_plan": section_patch_plan,
    }
    if not review_result["feedback"] and not review_result["approved"]:
        review_result["feedback"] = "报告仍需增强证据、结构或主题覆盖，请结合缺失主题与薄弱章节继续完善。"

    update = {
        "review_result": review_result,
        "revision_count": int(state.get("revision_count", 0)) + 1,
    }

    if not review_result["approved"] and review_result["missing_topics"]:
        review_state = {**state, "review_result": review_result}
        todo_items = _build_missing_topic_tasks(review_state, review_result["missing_topics"])
        if todo_items:
            update["todo_items"] = todo_items

    return update
