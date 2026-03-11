# Deep Researcher — 技术报告

> 注：本文档保留了迁移过程中的设计背景。当前仓库最新实现已经移除了对 `hello_agents` 和外层 `gpt_researcher` 的运行时依赖，搜索、嵌入与压缩能力均已内置到 `helloagents-deepresearch/backend/src`。

> 版本：v0.2.0 | 日期：2026-03-12

---

## 目录

1. [项目背景与目标](#1-项目背景与目标)
2. [系统架构](#2-系统架构)
3. [核心技术设计](#3-核心技术设计)
   - 3.1 [状态机设计与 Reducer](#31-状态机设计与-reducer)
   - 3.2 [LangGraph 图拓扑（Supervisor 多 Agent）](#32-langgraph-图拓扑supervisor-多-agent)
   - 3.3 [并行任务编排（Command fan-out）](#33-并行任务编排command-fan-out)
   - 3.4 [嵌入语义上下文压缩](#34-嵌入语义上下文压缩)
   - 3.5 [Researcher Mini-Agent 迭代覆盖闭环](#35-researcher-mini-agent-迭代覆盖闭环)
   - 3.6 [EvidenceStore Claim-Level 增强](#36-evidencestore-claim-level-增强)
   - 3.7 [Reviewer 主动路由](#37-reviewer-主动路由)
   - 3.8 [Writer 定向 Patch 架构](#38-writer-定向-patch-架构)
   - 3.9 [模型选择器机制](#39-模型选择器机制)
   - 3.10 [SSE 实时流式架构](#310-sse-实时流式架构)
4. [各模块详解](#4-各模块详解)
   - 4.1 [Planner Agent](#41-planner-agent)
   - 4.2 [Researcher Agent（Task Node）](#42-researcher-agenttask-node)
   - 4.3 [Writer Agent](#43-writer-agent)
   - 4.4 [Reviewer Agent](#44-reviewer-agent)
   - 4.5 [Supervisor Node](#45-supervisor-node)
   - 4.6 [Search Service](#46-search-service)
   - 4.7 [前端 SSE 客户端](#47-前端-sse-客户端)
5. [数据流全链路](#5-数据流全链路)
6. [配置系统设计](#6-配置系统设计)
7. [已知问题与技术债](#7-已知问题与技术债)
8. [性能特征分析](#8-性能特征分析)
9. [未来路线图（TODO）](#9-未来路线图todo)
10. [与同类方案对比](#10-与同类方案对比)

---

## 1. 项目背景与目标

本项目在 HelloAgents DeepResearch 基础上进行了深度重构，从 `hello_agents` 框架迁移至 LangGraph，并经历了四阶段 Multi-Agent 架构升级：

- **原项目优势保留**：多 LLM 提供商支持、多搜索后端、嵌入语义压缩、内置 `ContextCompressor` 压缩链路（源自 `gpt_researcher` 方案）
- **Phase 1-2 新增**：Researcher Mini-Agent 迭代搜证闭环、EvidenceStore Claim-Level 增强、Reviewer 主动路由
- **Phase 3 新增**：Writer 定向 Patch（无需整篇重写）、Planner Task Graph（priority / depends_on / search_budget）
- **Phase 4 新增**：真正 Multi-Agent 架构（Supervisor + ResearcherAgent×N + WriterAgent + ReviewerAgent，每个角色是独立编译的 subgraph）

核心设计目标：
1. 在保持**完全本地化可运行**（支持 Ollama / LMStudio）的前提下，实现生产级多智能体编排
2. 通过 LangGraph `Command(goto=[Send(...)])` 实现**真正并行**（非 asyncio.gather 轮询）的研究任务执行
3. 提供结构化的 Reviewer 质量门控，自动触发补充研究或定向章节改写
4. 通过 Supervisor 中央调度器统一管理 Agent 间的消息传递和状态路由

---

## 2. 系统架构

### 2.1 整体分层

```
┌─────────────────────────────────────────────────┐
│                   前端层（Vue 3）                 │
│  表单输入 → SSE 订阅 → 任务面板 → 报告渲染        │
└─────────────────────┬───────────────────────────┘
                      │ POST /research/stream (SSE)
┌─────────────────────▼───────────────────────────┐
│                   API 层（FastAPI）               │
│  请求解析 → 配置构建 → 事件映射 → SSE 序列化      │
└─────────────────────┬───────────────────────────┘
                      │ graph.astream_events(v2)
┌─────────────────────▼───────────────────────────┐
│          编排层（LangGraph Supervisor）           │
│  supervisor ──▶ planner_agent                   │
│            ──▶ researcher_agent ×N（并行）        │
│            ──▶ writer_agent                     │
│            ──▶ reviewer_agent                   │
└──────┬──────────────────────────────────┬────────┘
       │                                  │
┌──────▼──────┐                  ┌────────▼───────┐
│  搜索服务层  │                  │   LLM 调用层   │
│ dispatch_   │                  │ OpenAI 兼容客户 │
│ search +    │                  │ 端（多提供商）  │
│ backend适配 │                  │                │
└─────────────┘                  └────────────────┘
```

### 2.2 进程模型

- 后端：单进程 Uvicorn（asyncio 事件循环）
- 搜索：`asyncio.to_thread` 转移同步 SearchTool 至线程池，不阻塞事件循环
- 嵌入压缩：`ContextCompressor.async_get_context` 原生异步
- LLM：`AsyncOpenAI` 全异步调用，Writer 全文生成支持 `stream=True` 流式 token 推送

---

## 3. 核心技术设计

### 3.1 状态机设计与 Reducer

系统有两层状态：

**`ResearchState`** — 研究核心状态（所有 Agent 共享）：

```python
class ResearchState(TypedDict):
    research_topic: str
    todo_items:     Annotated[list[TodoItem],     merge_todo_items]  # 自定义合并
    visited_urls:   Annotated[set[str],           operator.or_]      # 并集合并
    evidence_store: Annotated[list[EvidenceItem], operator.add]      # 追加合并
    research_data:  Annotated[list[dict],         operator.add]      # 追加合并
    structured_report: str
    review_result:  dict
    revision_count: int
    max_revisions:  int
    config:         dict
    agent_role:     str
    research_loop_count: int
```

**`GlobalState`** — 继承 ResearchState，追加 Multi-Agent 调度状态：

```python
class GlobalState(ResearchState):
    messages: Annotated[list[AgentMessage], merge_agent_messages]  # 带容量上限的消息总线
    final_report: Optional[str]
    status: str  # init / planning / researching / writing / reviewing / done
```

**四种 Reducer 的语义**：

| 字段 | Reducer | 语义 |
|---|---|---|
| `todo_items` | `merge_todo_items` | 按 id 合并：相同 id 更新字段，新 id 追加。Phase 3 新字段（priority 等）通过 `{**original, **update}` 保留 |
| `visited_urls` | `operator.or_` | 集合并集：多个并行 ResearcherAgent 的已访问 URL 自动合并，实现全局去重 |
| `evidence_store` | `operator.add` | 列表追加：所有任务的证据条目（含 claim_text/support_type/section_hint）累积到同一个证据库 |
| `messages` | `merge_agent_messages` | 列表追加 + 容量上限（64条）：防止无限累积；保留最近 64 条 Agent 消息供 Supervisor 路由 |

**`TodoItem` Phase 3 新增字段**（均为 `NotRequired`，向后兼容）：

```python
class TodoItem(TypedDict):
    id: int
    title: str
    intent: str
    query: str
    status: str          # pending / in_progress / completed
    summary: Optional[str]
    sources_summary: Optional[str]
    priority:     NotRequired[int]        # 1 = 最高优先级
    depends_on:   NotRequired[list[int]]  # 依赖的任务 ID 列表
    search_budget: NotRequired[int]       # 允许的最大搜索迭代次数
    search_type:  NotRequired[str]        # "search" | "browser"
```

---

### 3.2 LangGraph 图拓扑（Supervisor 多 Agent）

```
START
  │
  ▼
supervisor (status=init)
  │
  ▼ goto="planner_agent"
planner_agent
  ├─ planner node（生成带 priority/depends_on/search_budget 的 TaskGraph）
  └─ planner_handoff → AgentMessage(task_assignment) → supervisor
  │
  ▼ supervisor 收到 task_assignment → Command(goto=[Send(...) × N])
researcher_agent(T1)  researcher_agent(T2)  ...（并行 subgraph）
  ├─ task_node（迭代搜证闭环 + claim 提取）
  └─ researcher_handoff → AgentMessage(evidence_delivery) → supervisor
  │
  ▼（所有 Send 完成，reducer 合并，supervisor 收到最后一条 evidence_delivery）
  │ 若还有 pending 任务 → 继续 dispatch researcher（依赖图调度）
  │ 否则 → goto="writer_agent"
writer_agent
  ├─ writer_node（_patch_report 或 _write_full_report，含流式输出）
  └─ writer_handoff → AgentMessage(report_ready) → supervisor
  │
  ▼ supervisor 收到 report_ready → goto="reviewer_agent"
reviewer_agent
  ├─ reviewer_node（四维评估 + research_briefs + section_patch_plan）
  └─ reviewer_handoff → AgentMessage(report_approved | review_dispatch | patch_order | rewrite_order) → supervisor
  │
  ├─ report_approved → END（_finalize_report）
  ├─ review_dispatch（有缺失主题）→ dispatch researchers（supplemental）→ writer → reviewer...
  ├─ patch_order（有 section_patch_plan）→ writer（_patch_report）→ reviewer...
  ├─ rewrite_order（仅重写）→ writer（_write_full_report）→ reviewer...
  └─ revision_count > max_revisions → END（强制终止）
```

**AgentMessage 消息类型**：

| 类型 | 发送方 | 含义 |
|---|---|---|
| `task_assignment` | planner_agent | 规划完成，任务列表已就绪 |
| `evidence_delivery` | researcher_agent | 单个任务研究完成，证据已提交 |
| `report_ready` | writer_agent | 报告（全文或 patch）已生成 |
| `report_approved` | reviewer_agent | 报告通过审查 |
| `review_dispatch` | reviewer_agent | 有缺失主题，需补研后重写 |
| `patch_order` | reviewer_agent | 无需补研，仅定向改写指定章节 |
| `rewrite_order` | reviewer_agent | 无需补研，整篇重写 |

---

### 3.3 并行任务编排（Command fan-out）

Supervisor 通过 `Command(goto=[Send(...), Send(...), ...])` 实现真正并行的 Agent 分发：

```python
def _dispatch_researchers(state, tasks) -> Command:
    return Command(
        goto=[Send("researcher_agent", {
            "task": task,
            "config": state["config"],
            "research_topic": state["research_topic"],
            "visited_urls": state["visited_urls"],
            "research_loop_count": state["research_loop_count"],
        }) for task in tasks],
        update={"status": "researching", "todo_items": _mark_tasks_in_progress(tasks)},
    )
```

**依赖图调度**：`select_runnable_tasks` 仅选取所有 `depends_on` 任务均已完成的 `pending` 任务，实现有序的批次式并行：

```python
completed_ids = {task["id"] for task in todo_items if task["status"] == "completed"}
runnable = [task for task in pending if all(dep in completed_ids for dep in task["depends_on"])]
return sorted(runnable, key=lambda t: (t.get("priority", 999), t["id"]))
```

**与 asyncio.gather 的本质区别**：

| 维度 | asyncio.gather | LangGraph Send() |
|---|---|---|
| 状态隔离 | 共享可变状态，需手动加锁 | 每个节点获得独立状态副本 |
| 扩展性 | 硬编码并发列表 | 动态生成，任务数可变 |
| 结果合并 | 手动收集 | Reducer 自动合并 |
| 可观测性 | 无原生支持 | `astream_events` 原生追踪 |
| 依赖调度 | 不支持 | 通过 Supervisor 按批次分发 |

---

### 3.4 嵌入语义上下文压缩

**问题**：每个任务搜索返回多条结果，每条可能包含完整页面内容（数千字符），直接拼接会超出 LLM 上下文窗口。

**解决方案**：内置 `ContextCompressor`（设计来源于 `gpt_researcher`，当前已在仓库内实现）

```python
async def _compress_context(query, runtime_config, pages) -> str:
    total_chars = sum(len(p["raw_content"]) for p in pages)

    # 快速路径：内容较少时跳过嵌入，直接拼接
    if total_chars < 16000:
        return _format_fast_path_context(pages)

    # 慢速路径：用嵌入相似度筛选最相关内容
    compressor = ContextCompressor(
        documents=pages,
        embeddings=_resolve_embeddings(runtime_config),
        max_results=8,
        similarity_threshold=0.42,
    )
    return await compressor.async_get_context(query=query, max_results=8)
```

**工作原理**：
1. 将每页内容切分为 chunk（`RecursiveCharacterTextSplitter`）
2. 用嵌入模型计算每个 chunk 与查询的余弦相似度
3. 保留相似度 ≥ threshold 的 chunk，丢弃无关内容
4. 保证输出 token 量在合理范围内，同时保留最相关信息

**嵌入提供商支持**：

| 格式 | 提供商 | 说明 |
|---|---|---|
| `openai:text-embedding-3-small` | OpenAI | 需要 API Key |
| `ollama:nomic-embed-text` | Ollama | 完全本地 |
| `custom:bge-m3` | 自定义兼容端点 | 需要 LLM_BASE_URL |

---

### 3.5 Researcher Mini-Agent 迭代覆盖闭环

**问题**：单次搜索可能无法充分覆盖任务的所有关键角度。

**解决方案**：task_node 内部实现 `assess_gap → rewrite_query → re-search → stop` 迭代闭环。

```
for iteration in range(max_iterations):
    dispatch_search(current_query)
    _filter_new_results + _compress_context → context

    if iteration == max_iterations - 1: break

    assessment = await _assess_coverage(task, context)
    if assessment["is_sufficient"] or score >= threshold: break
    if not assessment["unresolved_questions"]: break

    new_query = await _rewrite_query(task, unresolved_questions, tried_queries)
    if not new_query or new_query in tried_queries: break

    tried_queries.append(new_query)
    current_query = new_query
```

**`_assess_coverage`** — 让 LLM 判断当前上下文对任务的覆盖度：
- 输出：`coverage_score (0~1)` / `is_sufficient` / `unresolved_questions`
- 双重保险：`is_sufficient=True` 且 `coverage_score >= threshold` 才停止

**`_rewrite_query`** — 基于未覆盖问题生成新查询：
- 传入已试查询列表，要求不重复
- 提取第一行、清理标点、去除序号前缀

**关键配置**：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `researcher_max_iterations` | 3 | 最大搜索迭代次数 |
| `researcher_coverage_threshold` | 0.75 | 覆盖度阈值，达到即停止 |
| `task.search_budget` | 优先于全局配置 | 每任务独立预算 |

---

### 3.6 EvidenceStore Claim-Level 增强

**问题**：原始 evidence_store 只记录 URL + snippet，Writer 无法精确知道每条证据支持哪个论断、属于哪个章节。

**解决方案**：`_extract_claims` 在摘要生成后调用 LLM 从摘要中提取论断，并与证据 URL 匹配：

```python
# 每条 EvidenceItem 新增字段
class EvidenceItem(TypedDict):
    task_id: int
    url: str
    title: str
    snippet: str
    relevance_score: float
    claim_text:   NotRequired[Optional[str]]  # "X公司2024年营收达到Y亿"
    support_type: NotRequired[Optional[str]]  # support | contradict | background
    section_hint: NotRequired[Optional[str]]  # "市场规模分析"
```

**Writer 消费方式**：`_build_evidence_block` 将 `claim_text / support_type / section_hint` 注入 prompt，让 LLM 在写对应章节时能精确引用：

```
- 任务 2 | AI市场报告 | https://... | score=0.91 | relation=support | section=市场规模
  绑定论断：2024年全球AI市场规模达到1840亿美元，同比增长37%
  摘要：...
```

**降级设计**：`_extract_claims` 任何异常均返回原始 `evidence_items`，不影响主流程。

---

### 3.7 Reviewer 主动路由

**问题**：旧 Reviewer 只输出 `missing_topics / weak_sections`，系统无法区分"缺数据"和"写得差"，所有问题都触发相同的路由。

**解决方案**：Reviewer 输出扩展为六字段，配合明确的路由规则：

```json
{
  "approved": false,
  "score": 0.72,
  "feedback": "缺少技术局限性讨论；风险章节分析浅薄",
  "missing_topics": ["RAG 检索精度优化方法", "向量数据库性能对比"],
  "weak_sections": ["风险与挑战"],
  "research_briefs": [
    {"topic": "RAG精度优化", "intent": "收集具体优化方案", "query": "RAG retrieval accuracy optimization 2024", "priority": "high"}
  ],
  "section_patch_plan": [
    {"section": "风险与挑战", "issue": "缺乏具体风险量化", "instruction": "增加3个具体风险指标并引用数据来源"}
  ]
}
```

**三条路由分支**（通过 reviewer_handoff 的 AgentMessage 类型区分）：

| 条件 | AgentMessage 类型 | Supervisor 动作 |
|---|---|---|
| `approved=true` | `report_approved` | `_finalize_report` → END |
| `research_briefs` 或 `missing_topics` 非空 | `review_dispatch` | 补研 → writer |
| 仅 `section_patch_plan` 非空 | `patch_order` | writer（_patch_report） |
| 仅 `weak_sections` / feedback | `rewrite_order` | writer（_write_full_report） |

**防循环机制**：`revision_count` 在每次 reviewer_node 执行后 +1，Supervisor 检查 `revision_count > max_revisions` 强制终止。

---

### 3.8 Writer 定向 Patch 架构

**问题**：每次审查后都整篇重写，成本高且破坏已经写好的章节。

**解决方案**：`writer_node` 根据是否有 `section_patch_plan` 决定路由：

```python
async def writer_node(state) -> dict:
    patch_plan = state.get("review_result", {}).get("section_patch_plan", [])
    existing_report = str(state.get("structured_report") or "")

    if patch_plan and existing_report.strip():
        return await _patch_report(state, existing_report, patch_plan)
    return await _write_full_report(state)
```

**`_patch_report` 流程**：
1. 遍历 `section_patch_plan` 中每个 `{section, issue, instruction}`
2. `_find_section_span` 通过正则定位章节边界（支持 `#` 到 `######` 所有级别）
3. 构建包含**全量最新 research_data 和 evidence_store** 的 prompt（确保补研数据被利用）
4. LLM 仅输出修改后的该章节内容
5. `_replace_section` 将修改内容拼接回完整报告
6. `_ensure_references` 保证参考来源不丢失

**`_write_full_report`** 支持流式输出（`stream=True`）：
- 每个 token 通过 `adispatch_custom_event("report_chunk", {"token": token})` 推送
- 前端通过 SSE 接收 `report_chunk` 事件实现逐字渲染
- 流式失败时自动降级为非流式调用

---

### 3.9 模型选择器机制

系统支持对**规划/审查**（高决策质量）和**摘要/写作**（高吞吐量）使用不同能力的模型：

```python
# 规划器和审查器使用 strategic_llm（更强的推理能力）
config, provider, model = _resolve_model_config(runtime_config, selector_key="strategic_llm")

# 任务摘要、覆盖评估、论断提取、报告写作使用 smart_llm（更快、更便宜）
config, provider, model = _resolve_model_config(runtime_config, selector_key="smart_llm")
```

选择器格式 `provider:model-name` 允许为不同用途指向**完全不同的服务端点**：

```bash
STRATEGIC_LLM=custom:claude-3-5-sonnet  # 审查用强模型
SMART_LLM=custom:gpt-4o-mini            # 写作用快速模型
```

`_resolve_model_config` 的解析逻辑：
1. 从 `runtime_config` 中读取 `selector_key` 对应的值（如 `"custom:gpt-4o-mini"`）
2. 按 `:` 分割为 `(provider, model)`
3. 构建对应的 `AsyncOpenAI` 客户端（Ollama/LMStudio/Custom 各有不同 base_url），携带 `timeout` 参数
4. 若未指定，降级至 `llm_provider` + `local_llm`（兜底）

---

### 3.10 SSE 实时流式架构

**后端事件生成**：

```python
async def event_iterator() -> AsyncIterator[str]:
    async for event in graph.astream_events(initial_state, version="v2"):
        for mapped_event in _map_langgraph_event(event):
            yield f"data: {json.dumps(mapped_event)}\n\n"
    yield f"data: {json.dumps({'type': 'done'})}\n\n"
```

`astream_events(version="v2")` 穿透 subgraph 边界，为每个节点的 `on_chain_start` / `on_chain_end` 发出事件。`_map_langgraph_event` 将 LangGraph 原始事件转换为前端可消费的语义事件：

| LangGraph 事件 | 转换后类型 |
|---|---|
| `on_chain_start @ supervisor` | `status: "Supervisor 正在编排..."` |
| `on_chain_end @ planner_handoff` | `status: "PlannerAgent 已派发 N 个任务"` |
| `on_chain_end @ planner` | `todo_list: {tasks: [...]}` |
| `on_chain_start @ task_node` | `task_status: {status: "in_progress", priority, depends_on, ...}` |
| `on_chain_end @ task_node` | `task_status(completed) + sources + task_summary_chunk` |
| `on_chain_end @ researcher_handoff` | `status: "ResearcherAgent 已提交任务 N 的研究结果"` |
| `on_chain_start @ writer` | `status: "正在生成研究报告..."` |
| `on_custom_event: report_chunk` | `report_chunk: {token: "..."}` （逐字流式） |
| `on_chain_end @ writer` | `final_report: {report: "..."}` |
| `on_chain_end @ reviewer` | `review_result + research_briefs + patch_plan + status` |
| `on_chain_end @ reviewer_handoff` | `status: 对应 AgentMessage 类型的描述` |
| `on_chain_end @ LangGraph` | `done` |

**前端 SSE 解析**（手动实现，不依赖 EventSource API）：

```typescript
const reader = response.body.getReader();
let buffer = "";

while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });

    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
        const rawEvent = buffer.slice(0, boundary).trim();
        buffer = buffer.slice(boundary + 2);
        if (rawEvent.startsWith("data:")) {
            const event = JSON.parse(rawEvent.slice(5).trim());
            onEvent(event);
            if (event.type === "done" || event.type === "error") return;
        }
        boundary = buffer.indexOf("\n\n");
    }
    if (done) break;
}
```

使用手动解析而非 `EventSource` 的原因：`EventSource` 不支持 `POST` 请求和 `AbortSignal`，而研究请求需要携带请求体并支持用户主动取消。

---

## 4. 各模块详解

### 4.1 Planner Agent

**职责**：将用户输入的研究主题拆解为带有任务图结构的 `TodoItem` 列表

**处理流程**：
1. 使用 `strategic_llm`（高推理能力模型）发出规划请求
2. 系统 prompt 注入任务图字段要求（`priority / depends_on / search_budget / search_type`）
3. 多级 JSON 提取（代码块 → 原始文本 → `{}` 边界 → `[]` 边界）
4. `json_repair` 库兜底处理格式不规范的输出
5. `_normalize_tasks` 对每个字段进行类型强制和合法性检查
6. 若解析完全失败，返回单个兜底任务保证流程不中断

**Task Graph 输出示例**：

```json
[
  {"id": 1, "title": "市场规模", "query": "AI市场规模2024", "priority": 1, "depends_on": [], "search_budget": 3},
  {"id": 2, "title": "技术趋势", "query": "AI技术趋势", "priority": 1, "depends_on": [], "search_budget": 2},
  {"id": 3, "title": "竞争格局", "query": "AI企业竞争", "priority": 2, "depends_on": [1, 2], "search_budget": 2}
]
```

**输出**：`{"todo_items": [...], "agent_role": "..."}`，并通过 `planner_handoff` 发送 `task_assignment` 消息给 Supervisor。

---

### 4.2 Researcher Agent（Task Node）

**职责**：执行单个研究任务，实现搜索 → 过滤 → 压缩 → 覆盖评估 → 迭代 → 摘要 → 论断提取的完整链路

**处理流程**：

```
从 task["search_budget"] 或 config.researcher_max_iterations 确定 max_iterations
        │
for iteration in range(max_iterations):
    asyncio.to_thread(dispatch_search, current_query)    # 同步搜索转异步
            │
    _filter_new_results(visited_urls)                    # URL 去重
            │
    _normalize_pages → _compress_context                 # 嵌入压缩（快/慢路径）
            │
    _prepend_answer_text                                 # 前置 AI 直接答案
            │
    _assess_coverage(task, context) → {score, questions} # 覆盖度评估
            │
    if sufficient or last iteration: break
            │
    _rewrite_query(task, unresolved_questions, tried)    # 生成新查询
        │
_generate_task_summary(smart_llm)                        # 生成 Markdown 摘要
        │
_build_evidence_items → _extract_claims(smart_llm)       # 论断级证据增强
```

**返回字段**：
- `research_data`：完整的任务上下文数据（含 context / summary / sources / backend / notices）
- `evidence_store`：结构化证据条目（URL + title + snippet + score + claim_text + support_type + section_hint）
- `visited_urls`：本次新增的 URL 集合（通过 `operator.or_` 合并）
- `todo_items`：更新后的任务状态（`status="completed"`, `summary`, `sources_summary`，Phase 3 字段由 merge_todo_items 从原始任务保留）

完成后通过 `researcher_handoff` 发送 `evidence_delivery` 消息给 Supervisor。

---

### 4.3 Writer Agent

**职责**：将所有任务的研究数据整合为完整的 Markdown 研究报告，支持定向 patch 和全文重写两种模式

**路由逻辑**：

```python
if patch_plan and existing_report:
    return await _patch_report(state, existing_report, patch_plan)
return await _write_full_report(state)
```

**全文重写（`_write_full_report`）Prompt 构建**：

```
研究主题
  + ## 任务研究结果
    任务ID | 标题 | 目标 | 摘要 | 来源 | 原始上下文（截断至8000字符/任务）
  + ## 证据库
    任务ID | 标题 | URL | 分数 | relation | section | 绑定论断 | 摘要（截断至1000字符/条）
  + [审查反馈块（仅重写时追加）]
    总体反馈 + 需加强章节 + 补研简报 + 定向改写计划 + 上一版报告标题结构
```

**定向 Patch（`_patch_report`）流程**：
1. 遍历 `section_patch_plan` 逐章节处理
2. `_find_section_span` 定位章节（正则匹配 `#` 到 `######`，按层级感知边界）
3. Prompt 包含全量最新 `research_data` 和 `evidence_store`（含补研数据）
4. `_replace_section` 将新内容拼接回报告
5. 串行处理，每次 patch 基于上一次的修改结果

**后处理**：
1. 剥离 `<think>` token（兼容 DeepSeek/QwQ 等思考型模型）
2. 剥离 `[TOOL_CALL:...]` 残留标记
3. `_ensure_references`：检查是否已有参考来源章节，若无则从 `evidence_store` 自动生成

完成后通过 `writer_handoff` 发送 `report_ready` 消息给 Supervisor。

---

### 4.4 Reviewer Agent

**职责**：评估报告质量，输出结构化审查结果，通过消息类型精确表达路由意图

**输入信息**：
- 研究主题
- 任务快照（所有 TodoItem 的状态与摘要，摘要截断至 800 字符）
- 证据快照（按任务分组，含 claim_text 示例）
- 当前报告（截断至 12000 字符）

**评估维度（四维打分）**：
1. **证据充分性**：结论能否被具体、可追溯的证据支撑
2. **结构完整性**：层次清晰度、重点突出程度
3. **主题覆盖度**：是否覆盖所有关键子任务
4. **事实一致性**：报告内容与证据是否一致，有无过度推断

**Prompt 关键判断规则**（硬编码到 prompt，引导 LLM 精确分类）：
- `missing_topics`：仅填证据不足或主题缺失 → 触发补研
- `weak_sections`：仅填证据充分但表述差 → 触发重写
- 同时缺数据又写得差 → 两个字段都填
- `research_briefs`：`missing_topics` 非空时的结构化补研单

**状态更新**：
- `revision_count += 1`
- `review_result`：完整审查结果（含 research_briefs / section_patch_plan）
- `todo_items`（仅 missing_topics 非空时）：新增补研任务，通过 `merge_todo_items` 合并；并补齐 `priority / depends_on / search_budget / search_type`

完成后通过 `reviewer_handoff` 发送对应类型的消息给 Supervisor，实现主动路由。

---

### 4.5 Supervisor Node

**职责**：中央调度器，读取 AgentMessage 消息总线决定下一步路由，不执行任何业务逻辑

**路由状态机**（基于 `status` + `last_message.type`）：

```python
def supervisor_node(state) -> Command:
    if status == "init":
        return Command(goto="planner_agent", ...)

    last_message = _latest_supervisor_message(state)  # 倒序找最近一条 to_agent=supervisor 的消息

    if last_message is None:
        # 容灾：直接按状态推断
        runnable_tasks = select_runnable_tasks(state)
        if runnable_tasks: return _dispatch_researchers(state, runnable_tasks)
        if structured_report: return Command(goto="reviewer_agent", ...)
        return Command(goto="planner_agent", ...)

    match last_message.type:
        "task_assignment"  → dispatch researchers（依赖图调度）
        "evidence_delivery" → dispatch remaining tasks or goto writer
        "report_ready"     → goto reviewer
        "report_approved"  → _finalize_report → END
        "review_dispatch"  → dispatch supplemental researchers or goto writer
        "patch_order"      → goto writer（_patch_report）
        "rewrite_order"    → goto writer（_write_full_report）
```

**`select_runnable_tasks`** — 依赖图感知的任务选择：
- 过滤 `status=pending` 的任务
- 检查 `depends_on` 中所有依赖均已 `completed`
- 按 `(priority, id)` 升序排列，优先执行高优先级任务

---

### 4.6 Search Service

**封装层次**：

```
task.py
  └─ asyncio.to_thread(dispatch_search)
        └─ services/search.py::dispatch_search(query, config, loop_count)
              └─ [DuckDuckGo / Tavily / Perplexity / SearXNG / Advanced]
```

**核心参数**：
- `backend`：搜索后端，由 `config.search_api` 决定
- `fetch_full_page`：是否抓取完整页面（影响 `raw_content` 字段）
- `max_results`：最大结果数（固定为 8，对应 `services/search.py::MAX_RESULTS`）
- `max_tokens_per_source`：每条来源最大 token 数（4000，对应 `services/search.py::MAX_TOKENS_PER_SOURCE`）

**返回值规范化**：
- 若 SearchTool 返回字符串（错误信息），构造空结果结构并记录 notice
- 统一提取 `results / answer / notices / backend` 字段
- `answer` 字段：部分后端（如 Perplexity）会直接返回 AI 生成答案，前置注入上下文

**全局单例设计**：SearchTool 初始化成本较高，使用模块级单例避免重复初始化。

---

### 4.7 前端 SSE 客户端

**响应式状态设计**：

```typescript
const todoTasks = ref<TodoTaskView[]>([])              // 任务列表（含 priority/depends_on/search_budget）
const latestReview = ref<ReviewSnapshot | null>(null)  // 最新审查结果（含 research_briefs/patch_plan）
const reportMarkdown = ref("")                          // 最终报告（实时流式更新）
const progressLogs = ref<string[]>([])                  // 进度日志（含 Agent 名称）
```

**事件处理映射**：

| 事件类型 | 处理逻辑 |
|---|---|
| `todo_list` | 初始化任务列表，设置 `activeTaskId` |
| `task_status` | 更新任务状态；展示 priority/depends_on/search_budget |
| `task_summary_chunk` | 增量追加摘要文本 |
| `sources` | 解析来源文本为 `SourceItem[]`，触发高亮动画 |
| `report_chunk` | 逐 token 追加到 `reportMarkdown`，实现流式渲染 |
| `review_result` | 更新 `ReviewSnapshot`，显示审查评分、research_briefs、patch_plan |
| `research_briefs` | 展示结构化补研简报 |
| `patch_plan` | 展示定向改写计划 |
| `final_report` | 渲染最终 Markdown 报告 |
| `status` | 追加进度日志；含 Agent 名称时高亮对应面板 |

---

## 5. 数据流全链路

```
用户输入 "研究 LLM RAG 技术进展"
    │
    ▼
POST /research/stream
    │
    ▼
_build_initial_state → {
    research_topic: "研究 LLM RAG 技术进展",
    todo_items: [], visited_urls: set(), evidence_store: [],
    research_data: [], revision_count: 0, max_revisions: 2,
    messages: [], status: "init"
}
    │
    ▼
supervisor（status=init）→ goto planner_agent
    │
    ▼
planner_node（strategic_llm）
  → 4个 TodoItem（含 priority/depends_on/search_budget）
  planner_handoff → AgentMessage(task_assignment, task_count=4) → supervisor
    │
    ▼ SSE: todo_list（含 priority/search_budget 字段）
    │
supervisor 收到 task_assignment
  select_runnable_tasks → [T1, T2]（T3 depends_on T1,T2 暂不可运行）
  Command(goto=[Send(T1), Send(T2)], update={status: researching})
    │
    ▼（并行执行）
researcher_agent(T1)                     researcher_agent(T2)
  task_node：迭代搜证（最多3次）           task_node：迭代搜证（最多2次）
    assess_coverage → rewrite_query        assessment → break early
    _extract_claims → EvidenceItem+论断    _extract_claims
  researcher_handoff → evidence_delivery  researcher_handoff → evidence_delivery
    │
    ▼（reducer 合并，supervisor 收到最后一条 evidence_delivery）
supervisor：select_runnable_tasks → [T3]（T1,T2 已完成）
  Command(goto=[Send(T3)])
    │
researcher_agent(T3)...
    │
    ▼ 所有任务完成，supervisor → goto writer_agent
    │
writer_agent
  writer_node（_write_full_report，stream=True）
  → SSE: report_chunk × N tokens（逐字流式）
  → structured_report
  writer_handoff → report_ready → supervisor
    │
    ▼ SSE: final_report
    │
supervisor → goto reviewer_agent
    │
reviewer_agent
  reviewer_node（strategic_llm）
  → {approved: false, score: 0.78,
     missing_topics: ["RAG 幻觉问题"],
     research_briefs: [{topic: "RAG幻觉", query: "RAG hallucination mitigation 2024", priority: "high"}],
     section_patch_plan: [{section: "风险与挑战", instruction: "增加3个具体风险指标"}],
     revision_count: 1}
  reviewer_handoff → review_dispatch → supervisor
    │ SSE: review_result + research_briefs + patch_plan
    │
supervisor 收到 review_dispatch
  select_runnable_tasks → [T5(pending, "RAG幻觉问题")]
  Command(goto=[Send(T5)], update={research_loop_count: 1})
    │
researcher_agent(T5): 搜索 "RAG hallucination mitigation 2024"
    │
supervisor → goto writer_agent
    │
writer_agent
  writer_node → _patch_report（有 section_patch_plan + existing_report）
  → 仅修改"风险与挑战"章节，注入 T5 的新研究数据
  writer_handoff → report_ready → supervisor
    │
supervisor → goto reviewer_agent
reviewer_agent（revision_count=2 → approved 或 end）
    │
    ▼ SSE: done
```

---

## 6. 配置系统设计

`Configuration` 基于 Pydantic BaseModel，通过 `from_env()` 从环境变量加载：

```python
@classmethod
def from_env(cls, overrides=None) -> "Configuration":
    raw_values = {}
    # 1. 按字段名大写读取环境变量
    for field_name in cls.model_fields.keys():
        if field_name.upper() in os.environ:
            raw_values[field_name] = os.environ[field_name.upper()]
    # 2. 显式别名（向后兼容）
    env_aliases = {"smart_llm": os.getenv("SMART_LLM"), ...}
    for k, v in env_aliases.items():
        if v is not None:
            raw_values.setdefault(k, v)
    # 3. 运行时覆盖（来自 API 请求体）
    if overrides:
        raw_values.update({k: v for k, v in overrides.items() if v is not None})
    return cls(**raw_values)
```

**优先级**：运行时覆盖 > 显式环境变量别名 > 字段名大写环境变量 > Pydantic 默认值

**Phase 3 新增配置字段**：

| 字段 | 默认值 | 说明 |
|---|---|---|
| `researcher_max_iterations` | 3 | 每任务最大搜索迭代次数（可被 task.search_budget 覆盖） |
| `researcher_coverage_threshold` | 0.75 | 覆盖度阈值，达到即提前停止迭代 |
| `llm_timeout_seconds` | 30 | LLM API 调用超时（防止代理链超时导致假报错） |

---

## 7. 已知问题与技术债

### 7.1 高优先级

当前未发现会阻断主流程的高优先级已知问题。Phase 1-4 的核心路径已经闭合，现存技术债主要集中在前端体验、历史残留代码和超长运行场景下的可观测性取舍。

---

### 7.2 中优先级

**前端 `in_progress` 重置已有内容**

```typescript
if (status === "in_progress") {
    task.summary = "";
    task.sourceItems = [];
    // 补研轮次中，用户正在查看的内容被清空
}
```

在补研轮次中，某任务再次进入 `in_progress`，已显示的内容会消失。**建议**：仅在任务首次从 `pending` 进入 `in_progress` 时清空。

**定向 Patch 仍是串行、非流式**

`writer_node` 在 `section_patch_plan` 路径下会逐章节串行调用 `_patch_report`，且不会像 `_write_full_report` 那样通过 `report_chunk` 逐 token 推送。功能上正确，但多章节 patch 时用户会感知到明显空窗。**建议**：后续将 patch 路径升级为可选流式，或在前端显式展示“正在定向改写章节 X”。

---

### 7.3 低优先级

**`_generate_followup_queries` 死代码残留**（task.py）：函数注释标注"保留备用，当前已由迭代闭环替代"，但函数体仍占 ~40 行且永远不被调用。

**AgentMessage 只保留最近 64 条**

`merge_agent_messages` 会截断较早的消息，避免 `messages` 无限制增长。这对运行稳定性是合理的，但也意味着超长研究任务下，完整的 Agent 审计链不会全部保留在内存状态中。若未来需要长时审计，应改为日志落盘或独立事件存储。

**NoteTool 死代码残留**：前端 `TodoTaskView` 中的 `toolCalls / noteId / notePath` 字段、`tools-block` 模板区块，以及 `config.py` 中的 `enable_notes / notes_workspace / use_tool_calling` 字段均为历史遗留，对功能无影响但增加维护负担。

---

## 8. 性能特征分析

### 8.1 延迟来源

| 阶段 | 典型延迟 | 影响因素 |
|---|---|---|
| Planner LLM | 3-15s | strategic_llm 能力/速度 |
| 搜索（每任务，每迭代）| 1-5s | 网络延迟、后端响应 |
| 覆盖度评估 LLM | 1-5s/迭代 | smart_llm 速度 |
| 嵌入压缩（每任务）| 0.5-3s（慢路径）| 嵌入模型速度、内容量 |
| 任务摘要 LLM | 2-10s | smart_llm 速度 |
| 论断提取 LLM | 2-8s | smart_llm 速度、证据条数 |
| Writer LLM | 5-20s | 报告长度、模型速度 |
| Reviewer LLM | 3-10s | strategic_llm 速度 |

**总体延迟估算**（默认 3 任务，max 2 搜索迭代/任务，1 轮审查 + patch）：

- 并行研究阶段（实际）：Planner(5s) + [搜索×2 + 评估×1 + 压缩 + 摘要 + 论断提取] 并行(max~30s) ≈ **35s**
- 写作 + 审查 + Patch：Writer(10s) + Reviewer(7s) + Patch(5s) ≈ **22s**
- **总计约 57s**（相比无迭代版本增加约 15-20s，换取更高覆盖度）

### 8.2 并发瓶颈

- **LLM 请求**：受 API 速率限制约束，并行任务数过多可能触发 429；覆盖评估和论断提取增加了单任务的 LLM 调用次数
- **嵌入调用**：使用 OpenAI 嵌入时，多任务并行各自独立调用，可能同时打出多个嵌入请求
- **搜索后端**：DuckDuckGo 有频率限制，并发任务过多可能返回空结果

**建议配置**：`DEEP_RESEARCH_CONCURRENCY=3`（默认 4），避免过度并发触发限流。

### 8.3 内存特征

- `evidence_store`：每次迭代最多 `_SOURCE_LIMIT=8` 条证据/任务 × N任务，snippet 截断至 1000 字符，内存占用可控
- `research_data`：每任务保存完整 context（压缩后），多轮补研累积但不清理
- `messages`：通过 `merge_agent_messages` 限制最多 64 条，内存占用固定上限
- `visited_urls`：纯字符串集合，N任务 × 8条结果，极低内存占用

---

## 9. 未来路线图（TODO）

### 9.1 Multi-Agent 升级路线（状态）

**第一层：最高价值（已完成 ✅）**
- [x] **Researcher Mini-Agent 化**：task_node 内部实现 `assess_gap → rewrite_query → re-search → stop` 迭代闭环
- [x] **EvidenceStore Claim-Level 增强**：evidence_store 升级到 claim 级，每条证据绑定 `claim_text / support_type / section_hint`
- [x] **Reviewer 主动路由**：输出扩展为 `research_briefs / section_patch_plan`，reviewer_handoff 通过消息类型直接表达路由意图

**第二层：架构升级（已完成 ✅）**
- [x] **Writer 局部 Patch 能力**：`_patch_report` 仅修改 `section_patch_plan` 指定章节，注入全量最新研究数据
- [x] **Planner 输出 Task Graph**：输出 `priority / depends_on / search_budget / search_type`，Supervisor 按依赖图调度
- [x] **真正 Multi-Agent 架构**：Supervisor + ResearcherAgent×N + WriterAgent + ReviewerAgent，每个角色是独立编译的 subgraph，通过 AgentMessage 消息总线通信

**第三层：能力扩展（待实现）**
- [ ] **BrowserAgent**：引入 Playwright 驱动真实浏览器的专职 Agent，支持页内链接点击、分页导航、站内目录跳转、JS 渲染页面抓取。默认系统仍为 search-first，仅在搜索摘要不足、需要点进目录/分页、需要跨多页收集信息时 handoff 给 BrowserAgent。动作集合：`search / open / extract_links / click / next_page / stop`
- [ ] **NavigationState**：记录浏览轨迹，支持页面图扩展和链接优先级决策

---

## 10. 与同类方案对比

| 特性 | 本项目 | GPT-Researcher | LangGraph Open Deep Research |
|---|---|---|---|
| 并行编排 | LangGraph Command fan-out（依赖图调度）| asyncio.gather | LangGraph Send() |
| 状态合并 | 自定义 Reducer + merge_agent_messages | 手动累积 | 内置 Reducer |
| 审查循环 | Reviewer 四路路由（approved/dispatch/patch/rewrite）| 无 | 部分支持 |
| 迭代搜证 | assess_coverage + rewrite_query 闭环 | 无 | 无 |
| 证据质量 | Claim-Level（论断绑定 + section hint）| URL 级 | URL 级 |
| 报告更新 | 定向 Patch（指定章节）+ 全文重写 | 全文重写 | 全文重写 |
| 嵌入压缩 | ContextCompressor（gpt_researcher）| 内置 | 无 |
| URL 去重 | operator.or_ 全局集合 | 基于列表 | 有 |
| 搜索后端 | 5 种（含混合） | 10+ 种 | 3-5 种 |
| 本地 LLM | Ollama / LMStudio | 有限 | Ollama |
| 流式输出 | SSE + 逐 token report_chunk + Vue 3 前端 | 无前端 | 无前端 |
| 可配置性 | 环境变量全覆盖 | 代码配置 | 环境变量 |
| 部署复杂度 | 低（单进程）| 中 | 中 |
