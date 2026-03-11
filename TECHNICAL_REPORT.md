# Deep Researcher — 技术报告

> 注：本文档保留了迁移过程中的设计背景。当前仓库最新实现已经移除了对 `hello_agents` 和外层 `gpt_researcher` 的运行时依赖，搜索、嵌入与压缩能力均已内置到 `helloagents-deepresearch/backend/src`。

> 版本：v0.0.1 | 日期：2026-03-11

---

## 目录

1. [项目背景与目标](#1-项目背景与目标)
2. [系统架构](#2-系统架构)
3. [核心技术设计](#3-核心技术设计)
   - 3.1 [状态机设计与 Reducer](#31-状态机设计与-reducer)
   - 3.2 [LangGraph 图拓扑](#32-langgraph-图拓扑)
   - 3.3 [并行任务编排（Send() fan-out）](#33-并行任务编排send-fan-out)
   - 3.4 [嵌入语义上下文压缩](#34-嵌入语义上下文压缩)
   - 3.5 [Reviewer 审查-路由循环](#35-reviewer-审查-路由循环)
   - 3.6 [模型选择器机制](#36-模型选择器机制)
   - 3.7 [SSE 实时流式架构](#37-sse-实时流式架构)
4. [各模块详解](#4-各模块详解)
   - 4.1 [Planner Node](#41-planner-node)
   - 4.2 [Task Node](#42-task-node)
   - 4.3 [Writer Node](#43-writer-node)
   - 4.4 [Reviewer Node](#44-reviewer-node)
   - 4.5 [Research More Node](#45-research-more-node)
   - 4.6 [Search Service](#46-search-service)
   - 4.7 [前端 SSE 客户端](#47-前端-sse-客户端)
5. [数据流全链路](#5-数据流全链路)
6. [配置系统设计](#6-配置系统设计)
7. [已知问题与技术债](#7-已知问题与技术债)
8. [性能特征分析](#8-性能特征分析)
9. [与同类方案对比](#9-与同类方案对比)

---

## 1. 项目背景与目标

本项目在 HelloAgents DeepResearch 基础上进行了深度重构，从 `hello_agents` 框架迁移至 LangGraph，并融合了以下能力：

- **原项目优势保留**：多 LLM 提供商支持、多搜索后端、嵌入语义压缩、`gpt_researcher` ContextCompressor 集成
- **新增能力**：LangGraph 原生并行编排、自定义 Reducer 状态合并、Reviewer 审查循环、SSE 实时流推送、Vue 3 前端

核心设计目标：
1. 在保持**完全本地化可运行**（支持 Ollama / LMStudio）的前提下，实现生产级多智能体编排
2. 通过 LangGraph `Send()` API 实现**真正并行**（非 asyncio.gather 轮询）的研究任务执行
3. 提供结构化的 Reviewer 质量门控，自动触发补充研究或报告重写

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
│              编排层（LangGraph StateGraph）       │
│  planner → [task×N] → writer → reviewer → 路由  │
└──────┬──────────────────────────────────┬────────┘
       │                                  │
┌──────▼──────┐                  ┌────────▼───────┐
│  搜索服务层  │                  │   LLM 调用层   │
│ HelloAgents │                  │ OpenAI 兼容客户 │
│ SearchTool  │                  │ 端（多提供商）  │
└─────────────┘                  └────────────────┘
```

### 2.2 进程模型

- 后端：单进程 Uvicorn（asyncio 事件循环）
- 搜索：`asyncio.to_thread` 转移同步 SearchTool 至线程池，不阻塞事件循环
- 嵌入压缩：`ContextCompressor.async_get_context` 原生异步
- LLM：`AsyncOpenAI` 全异步调用

---

## 3. 核心技术设计

### 3.1 状态机设计与 Reducer

`ResearchState` 是整个图的共享状态，定义在 `graph/state.py`：

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

**三种 Reducer 的语义**：

| 字段 | Reducer | 语义 |
|---|---|---|
| `todo_items` | `merge_todo_items` | 按 id 合并：相同 id 更新字段，新 id 追加。实现任务状态的原地更新 |
| `visited_urls` | `operator.or_` | 集合并集：多个并行 task_node 的已访问 URL 自动合并，实现全局去重 |
| `evidence_store` | `operator.add` | 列表追加：所有任务的证据条目累积到同一个证据库 |
| `research_data` | `operator.add` | 列表追加：所有任务的研究数据供 writer 统一消费 |

**`merge_todo_items` 实现细节**：

```python
def merge_todo_items(current, updates):
    merged = [dict(item) for item in current]
    index_by_id = {int(item["id"]): idx for idx, item in enumerate(merged)
                   if str(item.get("id", "")).isdigit()}
    for update in updates:
        task_id = update.get("id")
        if task_id in index_by_id:
            merged[index_by_id[task_id]] = {**merged[index_by_id[task_id]], **update}
        else:
            merged.append(dict(update))
    return merged
```

合并策略为 **后者优先（update wins）**：task_node 完成后返回 `status="completed"` 等字段，会覆盖 planner 创建时的 `status="pending"`，同时保留 planner 设置的其他字段。

---

### 3.2 LangGraph 图拓扑

```
START
  │
  ▼
planner ──[conditional: route_tasks]──▶ task_node (×N 并行 Send)
                                              │
                                     ◀── fan-in (reducer 合并) ──
                                              │
                                             writer
                                              │
                                           reviewer
                                              │
                              ┌───────────────┼────────────────┐
                    [approved │ revision≥max] │ [missing_topics]│ [weak_sections]
                              ▼               ▼                 ▼
                             END        research_more        writer (rewrite)
                                              │
                                   [route_research_more]
                                              │
                                        task_node (补研并行)
                                              │
                                            writer
                                              │
                                           reviewer ...
```

关键路由函数：

```python
def route_tasks(state) -> list[Send]:
    # 将每个 pending todo_item 封装为独立 Send，真正并行
    return [Send("task_node", {task, config, topic, visited_urls})
            for task in state["todo_items"]]

def route_after_review(state) -> str:
    if review["approved"] or revision_count >= max_revisions:
        return "end"
    if review["missing_topics"]:
        return "research_more"   # 有缺失主题 → 补研
    return "rewrite"             # 仅结构/表达问题 → 直接重写

def route_research_more(state) -> list[Send]:
    # 只发出 pending 且 title/query 在 missing_topics 中的任务
    return [Send("task_node", {...})
            for task in state["todo_items"]
            if task["status"] == "pending"
            and task["title"] in missing_topics]
```

---

### 3.3 并行任务编排（Send() fan-out）

LangGraph `Send()` 的核心优势是**每个 Send 携带独立的局部状态注入**，而非让所有并行节点共享相同的全局状态切片：

```python
Send("task_node", {
    "task": task,              # 当前任务的 TodoItem
    "config": state["config"],
    "research_topic": state["research_topic"],
    "visited_urls": state["visited_urls"],  # 当前已访问集合快照
})
```

**与 asyncio.gather 的本质区别**：

| 维度 | asyncio.gather | LangGraph Send() |
|---|---|---|
| 状态隔离 | 共享可变状态，需手动加锁 | 每个节点获得独立状态副本 |
| 扩展性 | 硬编码并发列表 | 动态生成，任务数可变 |
| 结果合并 | 手动收集 | Reducer 自动合并 |
| 可观测性 | 无原生支持 | `astream_events` 原生追踪 |

fan-in 时，LangGraph 等待所有 `task_node` 实例完成后，将各自返回的 `todo_items / evidence_store / research_data / visited_urls` 通过对应 Reducer 合并到全局状态，再触发 `writer` 节点。

---

### 3.4 嵌入语义上下文压缩

**问题**：每个任务搜索返回 5 条结果，每条可能包含完整页面内容（数千字符），直接拼接会超出 LLM 上下文窗口。

**解决方案**：`ContextCompressor`（来自 `gpt_researcher` 库）

```python
async def _compress_context(query, runtime_config, pages) -> str:
    total_chars = sum(len(p["raw_content"]) for p in pages)

    # 快速路径：内容较少时跳过嵌入，直接拼接
    if total_chars < 8000:
        return _format_fast_path_context(pages)

    # 慢速路径：用嵌入相似度筛选最相关内容
    compressor = ContextCompressor(
        documents=pages,
        embeddings=_resolve_embeddings(runtime_config),
        max_results=5,
        similarity_threshold=0.42,
    )
    return await compressor.async_get_context(query=query, max_results=5)
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

### 3.5 Reviewer 审查-路由循环

Reviewer 是系统的**质量门控**节点，从四个维度评估报告：

1. **证据充分性**：结论是否有可追溯的具体证据
2. **结构完整性**：层次是否清晰，重点是否突出
3. **主题覆盖度**：是否覆盖所有关键子任务
4. **事实一致性**：内容与证据是否一致，有无过度推断

**输出结构**：
```json
{
  "approved": false,
  "score": 0.72,
  "feedback": "缺少对技术局限性的讨论，证据引用不够具体",
  "missing_topics": ["RAG 检索精度优化方法", "向量数据库性能对比"],
  "weak_sections": ["风险与挑战章节"]
}
```

**三条路由分支**：

| 条件 | 路由 | 后续 |
|---|---|---|
| `approved=true` 或 `revision_count >= max_revisions` | `end` | 终止，返回当前报告 |
| `missing_topics` 非空 | `research_more` | 生成补研任务，重新执行 task_node |
| 仅 `weak_sections` 或 feedback | `rewrite` | 将审查反馈注入 writer prompt，直接重写 |

**防循环机制**：`revision_count` 在每次 reviewer 执行后 +1，`route_after_review` 检查 `revision_count >= max_revisions` 强制终止，避免无限重写。

**补研任务去重**：`_build_missing_topic_tasks` 对 `title` 和 `query` 双重去重，防止相同主题被重复研究：

```python
existing_keys = {item["title"].lower() for item in existing_items}
existing_keys |= {item["query"].lower() for item in existing_items}
tasks = [t for t in new_tasks if t["title"].lower() not in existing_keys]
```

---

### 3.6 模型选择器机制

系统支持对**规划/审查**（高决策质量）和**摘要/写作**（高吞吐量）使用不同能力的模型：

```python
# 规划器和审查器使用 strategic_llm（更强的推理能力）
config, provider, model = _resolve_model_config(runtime_config, selector_key="strategic_llm")

# 任务摘要和报告写作使用 smart_llm（更快、更便宜）
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
3. 构建对应的 `AsyncOpenAI` 客户端（Ollama/LMStudio/Custom 各有不同 base_url）
4. 若未指定，降级至 `llm_provider` + `local_llm`（兜底）

---

### 3.7 SSE 实时流式架构

**后端事件生成**：

```python
async def event_iterator() -> AsyncIterator[str]:
    async for event in graph.astream_events(initial_state, version="v2"):
        for mapped_event in _map_langgraph_event(event):
            yield f"data: {json.dumps(mapped_event)}\\n\\n"
    yield f"data: {json.dumps({'type': 'done'})}\\n\\n"
```

`astream_events(version="v2")` 为每个节点的 `on_chain_start` / `on_chain_end` 发出事件，`_map_langgraph_event` 将 LangGraph 原始事件转换为前端可消费的语义事件：

| LangGraph 事件 | 转换后类型 |
|---|---|
| `on_chain_start @ planner` | `status: "规划研究任务..."` |
| `on_chain_end @ planner` | `todo_list: {tasks: [...]}` |
| `on_chain_start @ task_node` | `task_status: {status: "in_progress"}` |
| `on_chain_end @ task_node` | `task_status + sources + task_summary_chunk` |
| `on_chain_start @ writer` | `status: "正在生成研究报告..."` |
| `on_chain_end @ writer` | `final_report: {report: "..."}` |
| `on_chain_end @ reviewer` | `review_result: {approved, score, feedback, ...}` |
| `on_chain_end @ LangGraph` | `done` |

**前端 SSE 解析**（手动实现，不依赖 EventSource API）：

```typescript
const reader = response.body.getReader();
let buffer = "";

while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });

    let boundary = buffer.indexOf("\\n\\n");
    while (boundary !== -1) {
        const rawEvent = buffer.slice(0, boundary).trim();
        buffer = buffer.slice(boundary + 2);
        if (rawEvent.startsWith("data:")) {
            const event = JSON.parse(rawEvent.slice(5).trim());
            onEvent(event);
            if (event.type === "done" || event.type === "error") return;
        }
        boundary = buffer.indexOf("\\n\\n");
    }
    if (done) break;
}
```

使用手动解析而非 `EventSource` 的原因：`EventSource` 不支持 `POST` 请求和 `AbortSignal`，而研究请求需要携带请求体并支持用户主动取消。

---

## 4. 各模块详解

### 4.1 Planner Node

**职责**：将用户输入的研究主题拆解为 3-5 个互补的 `TodoItem` 任务

**处理流程**：
1. 使用 `strategic_llm`（高推理能力模型）发出规划请求
2. 要求模型输出 JSON 格式的任务列表
3. 多级 JSON 提取（代码块 → 原始文本 → `{}` 边界 → `[]` 边界）
4. `json_repair` 库兜底处理格式不规范的输出
5. 若解析完全失败，返回单个兜底任务保证流程不中断

**提示词结构**：
- System prompt：规划专家角色，已剥离 NoteTool 相关指令
- User prompt：注入当前日期 + 研究主题，要求输出 `{"tasks": [...]}` JSON

**输出**：`{"todo_items": [...], "agent_role": "..."}`

---

### 4.2 Task Node

**职责**：执行单个研究任务，完成搜索 → 过滤 → 压缩 → 摘要的完整链路

**处理流程**：

```
asyncio.to_thread(dispatch_search)    # 同步搜索转异步
        │
_filter_new_results(visited_urls)     # URL 去重，过滤已访问链接
        │
prepare_research_context              # 格式化来源摘要（sources_summary）
        │
_normalize_pages                      # 标准化页面数据结构
        │
_compress_context                     # 快速路径 or 嵌入压缩
        │
_prepend_answer_text                  # 若搜索后端有直接答案，前置注入
        │
_generate_task_summary                # smart_llm 生成 Markdown 摘要
        │
_build_evidence_items                 # 构建结构化证据条目
```

**返回字段**：
- `research_data`：完整的任务上下文数据（供 writer 使用）
- `evidence_store`：结构化证据条目（URL + title + snippet + score）
- `visited_urls`：本次新增的 URL 集合（通过 `operator.or_` 合并）
- `todo_items`：更新后的任务状态（`status="completed"`, `summary`, `sources_summary`）

---

### 4.3 Writer Node

**职责**：将所有任务的研究数据整合为完整的 Markdown 研究报告

**Prompt 构建逻辑**：

```
研究主题
  +
## 任务研究结果
  任务ID | 标题 | 目标 | 摘要 | 来源 | 原始上下文（截断至3000字符/任务）
  +
## 证据库
  任务ID | 标题 | URL | 相关度分数 | 摘要（截断至500字符/条）
  +
[审查反馈块（仅重写时追加）]
  审查总体反馈 + 需加强章节 + 补研主题 + 上一版报告标题结构（最多20个）
```

**后处理**：
1. 剥离 `<think>` token（兼容 DeepSeek/QwQ 等思考型模型）
2. 剥离 `[TOOL_CALL:...]` 残留标记
3. `_ensure_references`：检查是否已有参考来源章节，若无则从 `evidence_store` 自动生成

---

### 4.4 Reviewer Node

**职责**：评估报告质量，输出结构化审查结果，决定是否需要补研或重写

**输入信息**：
- 研究主题
- 任务快照（所有 TodoItem 的状态与摘要，摘要截断至 800 字符）
- 证据快照（按任务分组的 URL 列表）
- 当前报告（截断至 12000 字符）

**评估维度**（四维打分）：
1. 证据充分性：结论能否被证据支撑
2. 结构完整性：层次清晰度、重点突出程度
3. 主题覆盖度：是否覆盖所有关键子任务
4. 事实一致性：报告内容与证据是否一致

**输出解析**：与 planner 相同的多级 JSON 提取，确保格式不规范时也能正常解析。

**状态更新**：
- `revision_count += 1`
- `review_result`：完整审查结果
- `todo_items`（仅 missing_topics 非空时）：新增补研任务（通过 `merge_todo_items` Reducer 合并入现有列表）

---

### 4.5 Research More Node

**职责**：作为补研路由的中间节点，纯透传，不修改状态

```python
async def research_more_node(state) -> dict:
    del state
    return {}
```

设计意图：LangGraph 的 `add_conditional_edges` 需要绑定在某个真实节点上。`research_more_node` 作为状态检查点，让 `route_research_more` 函数能读取 Reviewer 更新后的完整状态（包含新增的 pending 补研任务），再通过 `Send()` fan-out 分派。

---

### 4.6 Search Service

**封装层次**：

```
task.py
  └─ asyncio.to_thread(dispatch_search)
        └─ HelloAgents SearchTool.run({backend, query, ...})
              └─ [DuckDuckGo / Tavily / Perplexity / SearXNG / Hybrid]
```

**核心参数**：
- `backend`：搜索后端，由 `config.search_api` 决定
- `fetch_full_page`：是否抓取完整页面（影响 `raw_content` 字段）
- `max_results`：最大结果数（固定为 5）
- `max_tokens_per_source`：每条来源最大 token 数（2000）

**返回值规范化**：
- 若 SearchTool 返回字符串（错误信息），构造空结果结构并记录 notice
- 统一提取 `results / answer / notices / backend` 字段
- `answer` 字段：部分后端（如 Perplexity）会直接返回 AI 生成答案，前置注入上下文

**全局单例设计**：SearchTool 初始化成本较高，使用模块级单例避免重复初始化：

```python
_GLOBAL_SEARCH_TOOL = None

def _get_search_tool():
    global _GLOBAL_SEARCH_TOOL
    if _GLOBAL_SEARCH_TOOL is None:
        _GLOBAL_SEARCH_TOOL = SearchTool(backend="hybrid")
    return _GLOBAL_SEARCH_TOOL
```

---

### 4.7 前端 SSE 客户端

**响应式状态设计**：

```typescript
const todoTasks = ref<TodoTaskView[]>([])      // 任务列表
const latestReview = ref<ReviewSnapshot | null>(null)  // 最新审查结果
const reportMarkdown = ref("")                  // 最终报告
const progressLogs = ref<string[]>([])          // 进度日志
```

**事件处理映射**：

| 事件类型 | 处理逻辑 |
|---|---|
| `todo_list` | 初始化任务列表，设置 `activeTaskId` |
| `task_status` | 更新任务状态；`in_progress` 时清空旧内容并切换 active |
| `task_summary_chunk` | 增量追加摘要文本 |
| `sources` | 解析来源文本为 `SourceItem[]`，触发高亮动画 |
| `review_result` | 更新 `ReviewSnapshot`，显示审查评分和反馈 |
| `final_report` | 渲染最终 Markdown 报告 |
| `status` | 追加进度日志；含 "Reviewer" 时触发审查面板高亮 |

**来源文本解析**（`parseSources`）：将后端返回的自由格式来源文本解析为结构化 `SourceItem`，支持多种格式：
- `* Title : URL`（format_sources 输出）
- `Source: Title / URL: xxx / Most relevant content: yyy`（deduplicate_and_format_sources 输出）
- 纯 URL 行

**`pulse()` 动画**：通过 Vue 的 `requestAnimationFrame` + `setTimeout` 实现"闪烁高亮"效果，在任务状态、来源、摘要、审查结果更新时给用户视觉反馈。

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
    research_data: [], revision_count: 0, max_revisions: 2
}
    │
    ▼
planner_node
  strategic_llm → JSON → 4个 TodoItem
  返回: {todo_items: [T1, T2, T3, T4]}
    │
    ▼ SSE: todo_list
    │
route_tasks → [Send(T1), Send(T2), Send(T3), Send(T4)]
    │
    ▼（并行执行）
task_node(T1)                task_node(T2)              ...
  搜索 "RAG 检索精度"          搜索 "向量数据库对比"
  过滤 visited_urls            过滤 visited_urls
  嵌入压缩 → context           嵌入压缩 → context
  smart_llm → summary          smart_llm → summary
  返回 {research_data[1],      返回 {research_data[2],
         evidence[1..5],              evidence[6..10],
         visited_urls{A,B,C},         visited_urls{D,E},
         todo_items[T1_done]}         todo_items[T2_done]}
    │
    ▼（reducer 合并）
全局状态 {
    research_data: [R1, R2, R3, R4],
    evidence_store: [E1..E20],
    visited_urls: {A,B,C,D,E,...},
    todo_items: [T1_done, T2_done, T3_done, T4_done]
}
    │ SSE: task_status(completed) × 4
    │
    ▼
writer_node
  构建 prompt = 研究主题 + 4个任务上下文 + 证据库
  smart_llm → Markdown 报告
  _ensure_references → 自动追加来源章节
  返回 {structured_report: "..."}
    │ SSE: final_report
    │
    ▼
reviewer_node
  strategic_llm → JSON 审查结果
  {approved: false, score: 0.78,
   missing_topics: ["RAG 幻觉问题"],
   weak_sections: ["风险与挑战"]}
  返回 {review_result: {...}, revision_count: 1,
         todo_items: [T5_pending("RAG 幻觉问题")]}
    │ SSE: review_result
    │
    ▼ route_after_review → "research_more"
    │
research_more_node（透传）
    │
route_research_more → [Send(T5)]
    │
task_node(T5): 搜索 "RAG 幻觉问题"...
    │
writer_node（重写，携带审查反馈）
    │
reviewer_node（revision_count=2 → approved or end）
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

**`config` 字段在状态中的传递**：初始状态的 `config` 字段是完整配置的序列化字典，每个 Send() 调用将其原样传入 task_node，确保所有并行子任务使用相同的配置快照，避免运行中配置漂移。

---

## 7. 已知问题与技术债

### 7.1 高优先级

**config.py 默认值自相矛盾**

```python
llm_provider = "ollama"      # 默认 Ollama
smart_llm = "gpt-4o-mini"   # 但模型名是 OpenAI 模型
strategic_llm = "o4-mini"   # 同上
```

若用户未设置环境变量直接运行，会向本地 Ollama 请求不存在的 `gpt-4o-mini` 模型，必然报错。**建议**：将默认值统一改为 Ollama 可用的模型名（如 `llama3.2`），或改为 `None` 强制用户显式配置。

---

### 7.2 中优先级

**`revision_count >= max_revisions` 导致实际迭代轮数减少一轮**

Reviewer 第1次执行后 `revision_count=1`，第2次执行后 `revision_count=2`，`2 >= 2` 触发强制终止。若 `max_revisions=2`，实际只能完成**1轮**有效的补研/重写循环。**建议**：改为 `> max_revisions` 或将比较移至 reviewer 节点内部。

**前端 `in_progress` 重置已有内容**

```typescript
if (status === "in_progress") {
    task.summary = "";
    task.sourceItems = [];
    // 重写轮次中，用户正在查看的内容被清空
}
```

在 Reviewer 触发重写后，若某任务再次进入 `in_progress`（实际上重写不会重跑 task_node，此问题只在补研轮次发生），已显示的内容会消失。**建议**：仅在任务首次从 `pending` 进入 `in_progress` 时清空。

---

### 7.3 低优先级

**`research_loop_count` 永远为 0**：状态中有该字段，但没有节点在执行后更新它。`dispatch_search` 收到的 `loop_count` 始终是 0，若搜索后端有基于循环数的行为差异，该功能实际不生效。

**`_resolve_embeddings` 的环境变量副作用**：

```python
elif provider == "ollama":
    os.environ.setdefault("OLLAMA_BASE_URL", config.ollama_base_url)
```

在多线程环境（`asyncio.to_thread`）中修改全局环境变量是不安全的模式。应直接通过构造函数参数传递 URL。

**`deduplicate_and_format_sources` 调用被浪费**：`prepare_research_context` 在搜索后调用此函数生成 `fallback_context`，但当 pages 非空时 `_compress_context` 总会返回非空结果，`fallback_context` 永远不被使用。

**NoteTool 死代码残留**：前端 `TodoTaskView` 中的 `toolCalls / noteId / notePath` 字段、`tools-block` 模板区块、`tool_call` 事件处理器，以及 `config.py` 中的 `enable_notes / notes_workspace / use_tool_calling` 字段均为历史遗留，对功能无影响但增加维护负担。

**`merge_todo_items` id 类型检查不一致**：构建 `index_by_id` 时接受字符串 id（通过 `isdigit()` 转换），但 update 循环检查 `isinstance(task_id, int)` 导致字符串 id 的更新会走 append 路径产生重复条目。实际影响极小（所有创建路径均使用 int），但代码不严谨。

---

## 8. 性能特征分析

### 8.1 延迟来源

| 阶段 | 典型延迟 | 影响因素 |
|---|---|---|
| Planner LLM | 3-15s | strategic_llm 能力/速度 |
| 搜索（每任务）| 1-5s | 网络延迟、后端响应 |
| 嵌入压缩（每任务）| 0.5-3s（慢路径）| 嵌入模型速度、内容量 |
| 任务摘要 LLM | 2-10s | smart_llm 速度 |
| Writer LLM | 5-20s | 报告长度、模型速度 |
| Reviewer LLM | 3-10s | strategic_llm 速度 |

**总体延迟估算**（默认 3 任务，1 轮审查）：

- 顺序执行：Planner(5s) + 搜索×3(4s each=12s) + 压缩×3(2s each=6s) + 摘要×3(5s each=15s) + Writer(10s) + Reviewer(5s) ≈ **53s**
- 并行执行（实际）：Planner(5s) + [搜索+压缩+摘要] 并行(max~20s) + Writer(10s) + Reviewer(5s) ≈ **40s**

### 8.2 并发瓶颈

- **LLM 请求**：受 API 速率限制约束，并行任务数过多可能触发 429
- **嵌入调用**：使用 OpenAI 嵌入时，多任务并行各自独立调用，可能同时打出多个嵌入请求
- **搜索后端**：DuckDuckGo 有频率限制，并发任务过多可能返回空结果

**建议配置**：`DEEP_RESEARCH_CONCURRENCY=3`（默认 4），避免过度并发触发限流。

### 8.3 内存特征

- `evidence_store`：5条证据/任务 × N任务，snippet 截断至 500 字符，内存占用可控
- `research_data`：每任务保存完整 context（压缩后），多轮补研累积但不清理
- `visited_urls`：纯字符串集合，N任务 × 5条结果，极低内存占用

---

## 9. 与同类方案对比

| 特性 | 本项目 | GPT-Researcher | LangGraph Open Deep Research |
|---|---|---|---|
| 并行编排 | LangGraph Send() fan-out | asyncio.gather | LangGraph Send() |
| 状态合并 | 自定义 Reducer | 手动累积 | 内置 Reducer |
| 审查循环 | Reviewer 三路路由 | 无 | 部分支持 |
| 嵌入压缩 | ContextCompressor（gpt_researcher） | 内置 | 无 |
| URL 去重 | operator.or_ 全局集合 | 基于列表 | 有 |
| 搜索后端 | 5 种（含混合） | 10+ 种 | 3-5 种 |
| 本地 LLM | Ollama / LMStudio | 有限 | Ollama |
| 流式输出 | SSE + Vue 3 前端 | 无前端 | 无前端 |
| 可配置性 | 环境变量全覆盖 | 代码配置 | 环境变量 |
| 部署复杂度 | 低（单进程） | 中 | 中 |
