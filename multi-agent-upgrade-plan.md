# Multi-Agent 升级计划

> 目标架构：Supervisor + ResearcherAgent×N + WriterAgent + ReviewerAgent
> 实施策略：分四个阶段渐进升级，每阶段独立可交付，不破坏前一阶段成果

---

## 整体升级路径

```
当前（多角色单图）
    ↓ Phase 1
Researcher 自主搜证闭环（task_node 内部 mini-agent 化）
    ↓ Phase 2
EvidenceStore Claim 级增强 + Reviewer 主动派单
    ↓ Phase 3
Writer 局部 Patch + Planner Task Graph
    ↓ Phase 4
真正 Multi-Agent：Supervisor + 独立子图架构
```

---

## Phase 1：Researcher Mini-Agent 化

### 目标

将 `task_node` 从"固定单次执行器"升级为"自主搜证 mini-agent"：
- 自己判断当前证据是否足够
- 自己决定是否需要改写 query 继续搜
- 自己决定何时停止

### 不改动的部分

- 总图结构（`builder.py`）完全不变
- 其他节点（planner / writer / reviewer）完全不变
- 对外接口（返回字段结构）完全不变

### 需改动的文件

- `backend/src/graph/nodes/task.py`（主要改动）
- `backend/src/config.py`（新增两个配置字段）

### 新增局部状态结构

```python
# task_node 内部维护的局部状态（不写入全局 ResearchState）
local_state = {
    "tried_queries": [str],        # 已尝试的 query 列表（去重）
    "evidence_chunks": [str],      # 已收集的上下文片段
    "unresolved_questions": [str], # 当前任务还未覆盖的问题点
    "coverage_score": float,       # 0.0~1.0，当前证据覆盖度评估
    "iteration": int,              # 当前迭代轮次
}
```

### 新增函数说明

#### `_assess_coverage(task, context, runtime_config) -> dict`

调用 LLM（smart_llm）评估当前收集到的上下文对任务目标的覆盖度。

输入：
- `task`：当前任务（含 title / intent / query）
- `context`：当前已收集的压缩上下文
- `runtime_config`：配置

输出 JSON：
```json
{
  "coverage_score": 0.65,
  "is_sufficient": false,
  "unresolved_questions": ["缺少具体数据对比", "未找到2024年最新案例"],
  "reasoning": "当前内容覆盖了基本概念，但缺少定量对比数据"
}
```

System prompt：
```
你是一名严格的研究质量评估员。你的任务是判断当前收集到的信息是否足以完成指定的研究任务。
```

User prompt：
```
研究任务：{task["title"]}
任务目标：{task["intent"]}
原始查询：{task["query"]}

当前已收集的上下文：
{context[:3000]}

请评估当前信息的覆盖度，并以 JSON 格式返回：
{{
  "coverage_score": 0.0到1.0之间的浮点数,
  "is_sufficient": true或false,
  "unresolved_questions": ["还未覆盖的具体问题1", "问题2"],
  "reasoning": "简短说明"
}}

只返回 JSON，不要其他内容。coverage_score >= 0.75 时 is_sufficient 才能为 true。
```

#### `_rewrite_query(task, unresolved_questions, tried_queries, runtime_config) -> str`

根据未解决的问题点，生成一个新的、更有针对性的搜索 query，避免与已尝试的 query 重复。

输入：
- `task`：当前任务
- `unresolved_questions`：coverage 评估返回的未覆盖问题列表
- `tried_queries`：已经尝试过的 query 列表（用于避免重复）
- `runtime_config`：配置

输出：字符串，新的搜索 query

System prompt：
```
你是一名专业研究员，擅长设计精准的搜索查询以填补研究空白。
```

User prompt：
```
研究任务：{task["title"]}
任务目标：{task["intent"]}

已尝试的查询（请勿重复）：
{chr(10).join(f"- {q}" for q in tried_queries)}

当前未覆盖的问题：
{chr(10).join(f"- {q}" for q in unresolved_questions)}

请生成一个新的、更有针对性的搜索查询，直接输出查询字符串，不要解释，不要引号。
```

### 改写后的 `task_node` 流程

```python
async def task_node(state):
    task = ...
    config = ...

    # 配置参数
    max_iterations = runtime_config.get("researcher_max_iterations", 3)
    coverage_threshold = runtime_config.get("researcher_coverage_threshold", 0.75)

    # 局部状态
    tried_queries = [str(task.get("query") or research_topic)]
    all_filtered_results = []
    all_notices = []
    current_query = tried_queries[0]

    for iteration in range(max_iterations):
        # 1. 搜索
        search_result, notices, answer_text, backend_label = await asyncio.to_thread(
            dispatch_search, current_query, config, loop_count
        )
        filtered_result, updated_urls, filtered = _filter_new_results(search_result, visited_urls)
        visited_urls = updated_urls
        all_filtered_results.extend(filtered)
        all_notices.extend(notices)

        # 2. 压缩上下文
        pages = _normalize_pages(all_filtered_results)
        context = await _compress_context(current_query, runtime_config, pages) if pages else ""
        context = _prepend_answer_text(context, answer_text)

        # 3. 评估覆盖度（最后一轮不评估，直接结束）
        if iteration < max_iterations - 1:
            assessment = await _assess_coverage(task, context, runtime_config)
            if assessment.get("is_sufficient") or assessment.get("coverage_score", 0) >= coverage_threshold:
                break  # 证据已足够，停止

            unresolved = assessment.get("unresolved_questions", [])
            if not unresolved:
                break  # 没有明确的未解决问题，停止

            # 4. 改写 query，准备下一轮
            new_query = await _rewrite_query(task, unresolved, tried_queries, runtime_config)
            if not new_query or new_query in tried_queries:
                break  # 没有新 query，停止

            tried_queries.append(new_query)
            current_query = new_query

    # 后续生成 summary、evidence_items 等（与现在相同）
    ...
```

### 新增配置字段（`config.py`）

```python
researcher_max_iterations: int = Field(default=3, description="Researcher 最大搜索迭代轮数")
researcher_coverage_threshold: float = Field(default=0.75, description="覆盖度阈值，达到即停止搜索")
```

### Codex 实现 Prompt

```
你是一名 Python 后端工程师。请按以下要求修改 backend/src/graph/nodes/task.py 和 backend/src/config.py。

【背景】
当前 task_node 是单次执行：搜索一次 → 压缩 → 总结 → 返回。
目标是将其升级为带自主搜证闭环的 mini-agent：搜索 → 评估覆盖度 → 改写 query → 再搜索 → 停止。

【config.py 改动】
在 Configuration 类中新增两个字段：
- researcher_max_iterations: int = Field(default=3)  # 最大搜索迭代轮数
- researcher_coverage_threshold: float = Field(default=0.75)  # 覆盖度阈值

【task.py 改动要求】

1. 新增异步函数 _assess_coverage(task, context, runtime_config) -> dict
   - 使用 smart_llm 调用 LLM
   - System: "你是一名严格的研究质量评估员。你的任务是判断当前收集到的信息是否足以完成指定的研究任务。"
   - User prompt 需包含：任务标题/目标/原始查询、当前上下文（截断至3000字符）
   - 要求 LLM 返回 JSON：{"coverage_score": float, "is_sufficient": bool, "unresolved_questions": [str], "reasoning": str}
   - coverage_score >= 0.75 时 is_sufficient 才能为 true（在 prompt 中说明）
   - 使用 with_llm_retry，解析失败时返回 {"coverage_score": 0.0, "is_sufficient": False, "unresolved_questions": [], "reasoning": "解析失败"}

2. 新增异步函数 _rewrite_query(task, unresolved_questions, tried_queries, runtime_config) -> str
   - 使用 smart_llm 调用 LLM
   - System: "你是一名专业研究员，擅长设计精准的搜索查询以填补研究空白。"
   - User prompt 包含：任务信息、已尝试的 query 列表（要求不重复）、未覆盖的问题列表
   - 要求 LLM 直接输出新的查询字符串
   - 返回清洗后的字符串（去除引号/多余空格）
   - 调用失败时返回空字符串 ""

3. 改写 task_node 函数
   - 从 runtime_config 读取 max_iterations 和 coverage_threshold
   - 用一个 for 循环替换现有的"Round 1 + Round 2..depth"逻辑
   - 循环内：搜索 → 过滤 → 累积结果 → 压缩上下文 → 评估覆盖度 → 判断是否继续
   - 最后一轮（iteration == max_iterations - 1）跳过覆盖度评估直接退出
   - 覆盖度足够、无未解决问题、无新 query、新 query 与已尝试重复，任意一种情况都 break
   - 循环后的 summary 生成、evidence_items 构建、返回值结构与现在完全一致
   - 保留现有的 _generate_followup_queries 函数但不再调用（保留以备后用）

4. 保持所有现有函数签名和返回值结构不变，只改 task_node 内部实现。

请先读取这两个文件的完整内容，再进行修改。不要改动其他文件。
```

---

## Phase 2：EvidenceStore Claim-Level 增强 + Reviewer 主动派单

### 目标

1. **EvidenceStore** 从 URL 级升级到 claim 级，每条证据绑定具体论断
2. **Reviewer** 从"输出建议"升级为"输出执行计划"，直接派发具体的补研单和改写单

### 不改动的部分

- 总图结构（`builder.py`）不变
- `task_node` / `writer_node` 接口不变（内部增强）

### 需改动的文件

- `backend/src/graph/nodes/task.py`（claim 提取）
- `backend/src/graph/nodes/writer.py`（按 claim 绑定证据写作）
- `backend/src/graph/nodes/reviewer.py`（扩展输出 + 主动派单）
- `backend/src/graph/state.py`（EvidenceItem 结构扩展）
- `backend/src/event_mapping.py`（新增 reviewer 事件类型）

### EvidenceItem 结构升级

```python
# 现在
{
    "task_id": 1,
    "url": "https://...",
    "title": "...",
    "snippet": "...",
    "relevance_score": 0.85,
}

# 升级后（新增字段为可选，向后兼容）
{
    "task_id": 1,
    "url": "https://...",
    "title": "...",
    "snippet": "...",
    "relevance_score": 0.85,
    # 新增
    "claim_text": "该技术在2024年市场占有率达到35%",   # 该证据支持的具体论断
    "support_type": "support",   # support / contradict / background
    "section_hint": "市场分析",  # 建议放入的报告章节
}
```

### Reviewer 扩展输出结构

```json
{
  "approved": false,
  "score": 0.72,
  "feedback": "...",
  "missing_topics": ["..."],
  "weak_sections": ["..."],
  "research_briefs": [
    {
      "topic": "RAG 幻觉率定量数据",
      "intent": "找到2023-2024年RAG系统幻觉率的具体统计数据",
      "query": "RAG hallucination rate benchmark 2024",
      "priority": "high"
    }
  ],
  "section_patch_plan": [
    {
      "section": "风险与挑战",
      "issue": "表述混乱，缺乏结构",
      "instruction": "将该章节改写为：问题描述 → 量化影响 → 现有解决方案 的三段式结构"
    }
  ]
}
```

### Codex 实现 Prompt

```
你是一名 Python 后端工程师。请按以下要求修改指定文件，实现 EvidenceStore Claim-Level 增强和 Reviewer 主动派单能力。

【文件1：graph/state.py】
EvidenceItem TypedDict 新增三个可选字段（不影响现有代码）：
- claim_text: Optional[str]
- support_type: Optional[str]  # "support" / "contradict" / "background"
- section_hint: Optional[str]

【文件2：graph/nodes/task.py】
在 _generate_task_summary 完成后，新增一个函数调用 _extract_claims：

新增函数 _extract_claims(task_id, summary, evidence_items, runtime_config) -> list[dict]
- 使用 smart_llm 从任务 summary 中提取关键论断
- System: "你是一名信息提取专家，擅长从研究摘要中识别关键论断并将其与来源证据对应。"
- User prompt：提供任务摘要和证据列表（url + snippet），要求返回每条论断对应哪条证据
- 输出 JSON：[{"claim_text": str, "evidence_url": str, "support_type": str, "section_hint": str}]
- 解析结果后，更新对应 evidence_items 中匹配 url 的条目，填充 claim_text / support_type / section_hint
- 调用失败时静默返回原始 evidence_items，不抛异常
- 函数调用放在 _build_evidence_items 之后，_generate_task_summary 之后

【文件3：graph/nodes/reviewer.py】
1. 扩展 _build_review_prompt，在返回格式中新增两个字段：
   - research_briefs: 需要补充搜索的具体研究简报列表，每项含 topic/intent/query/priority
   - section_patch_plan: 需要局部改写的章节计划，每项含 section/issue/instruction

2. 在 reviewer_node 中解析新字段：
   - 从 parsed 中提取 research_briefs 和 section_patch_plan（缺失时为空列表）
   - 将其加入 review_result 字典

3. _build_missing_topic_tasks 函数：优先使用 research_briefs 中的数据（topic → title, intent → intent, query → query）而非纯 missing_topics 文字，若 research_briefs 为空则降级使用原有 missing_topics 逻辑

【文件4：event_mapping.py】
在 on_chain_end @ reviewer 的处理中：
- review_result 中有 research_briefs 时，额外发出 {"type": "research_briefs", "briefs": [...]} 事件
- review_result 中有 section_patch_plan 时，额外发出 {"type": "patch_plan", "plan": [...]} 事件

请先读取所有涉及文件的完整内容，再进行修改。保持所有现有函数签名不变。
```

---

## Phase 3：Writer 局部 Patch + Planner Task Graph

### 目标

1. **Writer** 在 Reviewer 有 `section_patch_plan` 时不整篇重写，只修改指定章节
2. **Planner** 输出结构化 task graph，包含任务优先级、依赖关系、每任务搜索预算

### Writer Patch 逻辑

```python
async def writer_node(state):
    review = state.get("review_result", {})
    patch_plan = review.get("section_patch_plan", [])
    existing_report = state.get("structured_report", "")

    # 有 patch_plan 且有现有报告时：局部 patch 模式
    if patch_plan and existing_report.strip():
        return await _patch_report(state, existing_report, patch_plan)

    # 否则：全量写作（现有逻辑）
    return await _write_full_report(state)
```

#### `_patch_report` 函数逻辑

对每个 patch 项，单独调用 LLM 修改对应章节，然后将修改后的内容替换回原报告。

System prompt for patch：
```
你是一名精确的报告编辑。你的任务是修改研究报告中的指定章节，不要改动其他章节。
```

User prompt for patch：
```
以下是完整报告：
{existing_report}

请修改章节「{patch["section"]}」：
问题：{patch["issue"]}
修改要求：{patch["instruction"]}

只输出修改后的该章节内容（包含章节标题），不要输出其他章节，不要解释。
```

### Planner Task Graph 结构

```json
{
  "tasks": [
    {
      "id": 1,
      "title": "市场规模分析",
      "intent": "收集2023-2024年市场规模数据",
      "query": "AI market size 2024 statistics",
      "priority": 1,
      "depends_on": [],
      "search_budget": 3,
      "search_type": "search"
    },
    {
      "id": 2,
      "title": "技术对比分析",
      "intent": "对比主要技术方案的优劣势",
      "query": "RAG vs fine-tuning comparison 2024",
      "priority": 2,
      "depends_on": [1],
      "search_budget": 4,
      "search_type": "search"
    }
  ]
}
```

### Codex 实现 Prompt

```
你是一名 Python 后端工程师。请按以下要求修改指定文件。

【文件1：graph/nodes/writer.py】

1. 将现有写作逻辑重构为私有函数 _write_full_report(state) -> dict，返回 {"structured_report": str}。函数体与现有 writer_node 完全相同。

2. 新增函数 _patch_report(state, existing_report, patch_plan) -> dict：
   - 对 patch_plan 中的每一项，调用 LLM（smart_llm）修改对应章节
   - System prompt: "你是一名精确的报告编辑。你的任务是修改研究报告中的指定章节，不要改动其他章节。"
   - User prompt 包含：完整报告原文、当前要修改的章节名、问题描述、修改要求
   - 使用正则从原报告中定位对应章节（匹配 `## 章节名` 或 `### 章节名`），将 LLM 输出替换进去
   - 若章节定位失败，跳过该 patch 项（不报错）
   - 所有 patch 完成后，对最终报告调用 _ensure_references
   - 返回 {"structured_report": patched_report}

3. 改写 writer_node 函数：
   - 从 state 中读取 review_result 的 section_patch_plan 和现有 structured_report
   - 若 section_patch_plan 非空且 structured_report 非空：调用 _patch_report
   - 否则：调用 _write_full_report
   - 两条路径的返回值结构相同

【文件2：graph/nodes/planner.py】

1. 扩展 _normalize_tasks，识别并保存额外字段（向后兼容，字段缺失时使用默认值）：
   - priority: int（默认按顺序：idx）
   - depends_on: list[int]（默认 []）
   - search_budget: int（默认从 config.deep_research_depth 读取）
   - search_type: str（默认 "search"）

2. 更新 planner 的 system prompt（_sanitize_system_prompt 返回的内容）：在现有基础上追加一段说明，要求输出时对每个任务包含 priority / depends_on / search_budget 字段。

3. 保持 TodoItem 结构向后兼容（新字段缺失时不影响现有逻辑）。

请先读取所有涉及文件的完整内容，再进行修改。不要改动其他文件。
```

---

## Phase 4：真正 Multi-Agent 架构

### 目标

将现有节点升级为真正的 agent subgraph，引入 Supervisor 编排层，实现：
- 每个 Agent 有独立的局部状态
- Agent 间通过 `Command` handoff 而非共享状态通信
- Supervisor LLM 负责顶层编排决策

### 架构图

```
SupervisorGraph（顶层）
    │
    ├── PlannerAgent（subgraph）
    │     └── ReAct Loop: 分析主题 → 生成 task graph → 评估完整性 → 输出
    │
    ├── ResearcherAgent × N（subgraph，并行 Send）
    │     └── ReAct Loop: plan_search → search → assess → rewrite → search → stop
    │     └── Tools: search_tool, fetch_tool, assess_coverage_tool
    │
    ├── WriterAgent（subgraph）
    │     └── ReAct Loop: check_evidence → write/patch → self_check → output
    │     └── Tools: get_evidence_tool, patch_section_tool
    │
    └── ReviewerAgent（subgraph）
          └── ReAct Loop: analyze_report → generate_briefs → dispatch → output
          └── Tools: dispatch_research_tool, create_patch_plan_tool
```

### 核心数据流变化

```
现在（共享状态）：
  planner → 写 ResearchState.todo_items
  task_node → 读 ResearchState.todo_items，写 ResearchState.research_data

升级后（消息传递）：
  PlannerAgent → Command(goto="researcher", data={"tasks": [...]})
  ResearcherAgent → Command(goto="writer", data={"evidence": [...], "summaries": [...]})
  ReviewerAgent → Command(goto="researcher", data={"brief": {...}})  # 定向补研
               → Command(goto="writer", data={"patch_plan": [...]})  # 定向 patch
```

### 新文件结构

```
backend/src/
    graph/
        builder.py          ← 改为构建 SupervisorGraph
        state.py            ← 拆分为 GlobalState + 各 Agent LocalState
        supervisor.py       ← 新增：Supervisor 路由逻辑
        agents/             ← 新增目录
            planner_agent.py
            researcher_agent.py
            writer_agent.py
            reviewer_agent.py
        nodes/              ← 保留，供 agent 内部使用
            planner.py
            task.py（Phase 1 已升级）
            writer.py（Phase 3 已升级）
            reviewer.py（Phase 2 已升级）
```

### GlobalState 与 AgentMessage 结构

```python
# GlobalState（Supervisor 层）
class GlobalState(TypedDict):
    research_topic: str
    messages: Annotated[list[AgentMessage], operator.add]  # Agent 间通信消息
    final_report: Optional[str]
    status: str  # planning / researching / writing / reviewing / done

# AgentMessage
class AgentMessage(TypedDict):
    from_agent: str
    to_agent: str
    type: str       # task_assignment / evidence_delivery / review_dispatch / patch_order
    payload: dict
    timestamp: str
```

### Supervisor 路由逻辑

```python
async def supervisor_node(state: GlobalState) -> Command:
    """Supervisor 根据当前状态和最新消息决定下一步调用哪个 Agent"""
    last_message = state["messages"][-1] if state["messages"] else None

    if state["status"] == "init":
        return Command(goto="planner_agent")

    if last_message and last_message["type"] == "task_assignment":
        # Planner 输出任务 → 分派给 Researcher
        tasks = last_message["payload"]["tasks"]
        return Command(goto=[Send("researcher_agent", {"task": t}) for t in tasks])

    if last_message and last_message["type"] == "evidence_delivery":
        # 所有 Researcher 完成 → 触发 Writer
        return Command(goto="writer_agent")

    if last_message and last_message["type"] == "review_dispatch":
        # Reviewer 有补研需求 → 定向发给 Researcher
        briefs = last_message["payload"]["research_briefs"]
        return Command(goto=[Send("researcher_agent", {"task": b}) for b in briefs])

    if last_message and last_message["type"] == "patch_order":
        # Reviewer 有 patch 需求 → 发给 Writer
        return Command(goto="writer_agent", data={"patch_plan": last_message["payload"]})

    if last_message and last_message["type"] == "report_approved":
        return Command(goto=END)
```

### Codex 实现 Prompt

```
你是一名资深 Python 架构师，熟悉 LangGraph 的 subgraph、Command、Send 机制。
请按以下要求将现有的多角色单图架构升级为真正的 Multi-Agent 架构。

【前提条件】
Phase 1、2、3 已完成。现有代码中：
- task.py 已有 _assess_coverage / _rewrite_query 函数和循环逻辑
- reviewer.py 已有 research_briefs / section_patch_plan 输出
- writer.py 已有 _patch_report / _write_full_report 函数

【升级目标】
构建 Supervisor + ResearcherAgent + WriterAgent + ReviewerAgent 架构，
每个 Agent 是独立的 StateGraph subgraph，通过 AgentMessage 通信。

【步骤1：新建 graph/agents/ 目录，创建各 Agent subgraph】

researcher_agent.py：
- 创建 ResearcherAgentState(TypedDict)：包含 task/config/research_topic/visited_urls/messages
- build_researcher_graph() 函数：
  - 节点：task_node（直接复用现有 task.py 的 task_node）
  - 入口：task_node
  - task_node 完成后，将结果打包为 AgentMessage(type="evidence_delivery") 追加到 messages
  - 返回 CompiledGraph

writer_agent.py：
- 创建 WriterAgentState(TypedDict)：包含 research_data/evidence_store/review_result/structured_report/config/messages
- build_writer_graph() 函数：
  - 节点：writer_node（直接复用现有 writer.py 的 writer_node）
  - 入口：writer_node
  - 完成后打包 AgentMessage(type="report_ready") 追加到 messages
  - 返回 CompiledGraph

reviewer_agent.py：
- 创建 ReviewerAgentState
- build_reviewer_graph() 函数：
  - 节点：reviewer_node（直接复用现有 reviewer.py 的 reviewer_node）
  - 根据 review_result 决定消息类型：
    - approved → AgentMessage(type="report_approved")
    - research_briefs 非空 → AgentMessage(type="review_dispatch", payload={"research_briefs": [...]})
    - section_patch_plan 非空 → AgentMessage(type="patch_order", payload={"section_patch_plan": [...]})
    - 否则 → AgentMessage(type="rewrite_order")

【步骤2：新建 graph/supervisor.py】
- 定义 GlobalState(TypedDict)
- 定义 supervisor_node(state) 函数，实现上文描述的路由逻辑
- 使用 LangGraph Command(goto=...) 进行 Agent 间跳转

【步骤3：改写 graph/builder.py】
- build_graph() 改为构建 SupervisorGraph
- 将各 Agent subgraph 注册为节点（add_node 使用 subgraph 实例）
- 入口设为 supervisor_node
- 使用 add_conditional_edges 绑定 supervisor 路由

【步骤4：更新 graph/state.py】
- 新增 GlobalState、AgentMessage TypedDict
- 保留现有 ResearchState（供各 Agent subgraph 内部使用）

【步骤5：更新 main.py 和 agent.py】
- _build_initial_state 改为构建 GlobalState 格式
- 其他保持不变

【约束】
- 尽量复用现有节点函数，不要重写业务逻辑
- event_mapping.py 的事件类型保持不变，只需在 supervisor 层面适配
- 保持 /research/stream SSE 接口不变
- 每个步骤完成后代码可独立运行，不要一次性提交所有改动

请先读取所有涉及文件，理解现有代码结构，再按步骤实施。
```

---

## 注意事项

### 各阶段依赖关系

```
Phase 1（独立）→ Phase 2（依赖 Phase 1 的覆盖度评估）→ Phase 3（依赖 Phase 2 的 patch_plan）→ Phase 4（依赖 Phase 1-3 全部完成）
```

### Phase 4 实施前的检查清单

在喂给 Codex 之前，确认以下条件已满足：

- [ ] `task_node` 有 `_assess_coverage` 和 `_rewrite_query` 函数（Phase 1）
- [ ] `reviewer_node` 返回值中包含 `research_briefs` 和 `section_patch_plan`（Phase 2）
- [ ] `writer_node` 有 `_patch_report` 函数（Phase 3）
- [ ] `config.py` 有 `researcher_max_iterations` 和 `researcher_coverage_threshold` 字段（Phase 1）

### 测试建议

每个 Phase 完成后，用以下请求验证：

```bash
curl -X POST http://localhost:8000/research/stream \
  -H "Content-Type: application/json" \
  -d '{"topic": "2024年大语言模型推理优化技术进展"}' \
  --no-buffer
```

观察 SSE 事件流，确认：
- Phase 1：同一任务出现多次不同 query 的搜索日志
- Phase 2：`review_result` 事件包含 `research_briefs` 字段
- Phase 3：重写轮次中出现 `patch_plan` 相关日志而非整篇重写
- Phase 4：事件流中出现 Agent 间消息传递的 status 事件
