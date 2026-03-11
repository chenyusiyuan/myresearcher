# Role Behavior Spec

本文基于当前仓库中的实际实现整理，目标是把每个角色 agent 的行为流转、分支条件、状态更新和 handoff 讲清楚。

当前系统的本质是：

- 这是一个多角色、多 subgraph 的 LangGraph 工作流
- 顶层由 `Supervisor` 规则编排，不是完全放权的自治路由
- 每个角色 agent 有自己的局部状态和 handoff 消息
- 真正的分支主要发生在 `Supervisor`、`ResearcherAgent`、`ReviewerAgent`、`WriterAgent`

## 1. 总体拓扑

实际图结构：

```text
supervisor
  -> planner_agent
  -> researcher_agent x N
  -> writer_agent
  -> reviewer_agent
```

顶层编排逻辑：

```text
init
  -> planner_agent

planner_agent
  -> task_assignment
  -> supervisor

supervisor
  -> 并行 researcher_agent
  -> evidence_delivery
  -> writer_agent
  -> report_ready
  -> reviewer_agent
  -> report_approved / review_dispatch / patch_order / rewrite_order
  -> supervisor
```

其中：

- `planner_agent` 负责拆题，输出 `todo_items`
- `researcher_agent` 负责单任务研究，输出证据与摘要
- `writer_agent` 负责全文写作或定向 patch
- `reviewer_agent` 负责审查，决定通过、补研、局部改写或整篇重写
- `supervisor` 只负责路由和终止判断，不直接做研究内容生成

## 2. 共享状态与消息

### 2.1 GlobalState 关键字段

- `research_topic`: 当前研究主题
- `todo_items`: 任务列表，按 id merge
- `research_loop_count`: 补研轮次计数
- `structured_report`: 当前报告正文
- `visited_urls`: 全局已访问 URL 集合
- `evidence_store`: 证据列表
- `research_data`: 每个任务的上下文、摘要、来源
- `review_result`: reviewer 输出的结构化审查结果
- `revision_count`: 审查轮次
- `max_revisions`: 最大允许修订轮次
- `messages`: agent 间消息总线，最多保留 64 条
- `final_report`: 最终报告
- `status`: `init / planning / researching / writing / reviewing / done`

### 2.2 Reducer 行为

- `todo_items` 按 `id` 合并。相同 id 会覆盖更新字段，不同 id 会追加。
- `messages` 直接追加，超过 64 条时裁掉较早消息。
- `visited_urls` 用集合并集去重。
- `evidence_store` 和 `research_data` 直接累加。

### 2.3 消息类型

- `task_assignment`: Planner 已生成任务图
- `evidence_delivery`: Researcher 已完成某个任务
- `report_ready`: Writer 已生成报告或 patch 后报告
- `report_approved`: Reviewer 通过
- `review_dispatch`: Reviewer 要求补研，可能同时附带 patch 计划
- `patch_order`: Reviewer 不要求补研，只要求定向改写
- `rewrite_order`: Reviewer 不要求补研，也没有 patch plan，要求整篇重写

## 3. Supervisor

### 3.1 角色定位

`Supervisor` 是顶层路由器，不负责生成内容，负责：

- 看当前 `status`
- 读取最近一条发给 `supervisor` 的消息
- 选择下一个 agent
- 判断是否并行分发 researcher
- 判断是否终止

### 3.2 任务挑选逻辑

`Supervisor` 在派发研究任务前，会先调用 `select_runnable_tasks`：

- 只看 `status == "pending"` 的任务
- 优先选择依赖已全部完成的任务
- 如果存在 pending 任务，但没有任何任务满足依赖，也会退化为“直接返回全部 pending 任务”

这意味着：

- 当前依赖调度是“软约束”，不是严格阻塞型 DAG 调度
- 当依赖关系配置得不合理时，系统不会卡死，而是继续推进

### 3.3 并行派发行为

当有可运行任务时，`Supervisor` 会：

- 把这些任务状态改为 `in_progress`
- 设置全局 `status = "researching"`
- 对每个任务构造一个 `Send("researcher_agent", {...})`
- 一次性 fan-out 并行发给多个 `researcher_agent`

如果这是 reviewer 触发的补研回路，还会额外：

- `research_loop_count += 1`

### 3.4 Supervisor 全部分支

#### 分支 A: 初始进入

条件：

- `status == "init"`

动作：

- 跳到 `planner_agent`
- 更新 `status = "planning"`

#### 分支 B: 没有收到任何 supervisor 消息

条件：

- `last_message is None`

动作顺序：

1. 如果有可运行任务，派发 `researcher_agent`
2. 否则如果 `structured_report` 非空，跳到 `reviewer_agent`
3. 否则回到 `planner_agent`

这个分支主要是兜底分支。

#### 分支 C: 收到 `task_assignment`

条件：

- 最近消息类型为 `task_assignment`

动作：

1. 如果有可运行任务，并行派发 `researcher_agent`
2. 如果没有任务可跑，直接跳到 `writer_agent`

#### 分支 D: 收到 `evidence_delivery`

条件：

- 最近消息类型为 `evidence_delivery`

动作：

1. 如果还有可运行任务，继续并行派发 `researcher_agent`
2. 如果所有任务都研究完，跳到 `writer_agent`

#### 分支 E: 收到 `report_ready`

条件：

- 最近消息类型为 `report_ready`

动作：

- 跳到 `reviewer_agent`
- 更新 `status = "reviewing"`

#### 分支 F: 收到 `report_approved`

条件：

- 最近消息类型为 `report_approved`

动作：

- 结束图执行
- 设置 `status = "done"`
- 把 `final_report` 和 `structured_report` 统一为最终报告内容

#### 分支 G: 收到 `review_dispatch`

条件：

- 最近消息类型为 `review_dispatch`

动作顺序：

1. 如果 `revision_count > max_revisions + 1`，直接终止
2. 如果有可运行补研任务，并行派发 `researcher_agent`
3. 如果没有补研任务且 `revision_count > max_revisions`，直接终止
4. 否则跳到 `writer_agent`

这个分支的含义是：

- reviewer 认为需要补研
- 但 reviewer_node 是否真的新建出了 `todo_items`，要看缺失主题是否成功转成任务
- 如果没有新任务，系统不会卡住，而是直接交给 writer 用现有信息继续改稿

#### 分支 H: 收到 `patch_order` 或 `rewrite_order`

条件：

- 最近消息类型为 `patch_order` 或 `rewrite_order`

动作：

1. 如果 `revision_count > max_revisions`，直接终止
2. 否则跳到 `writer_agent`

#### 分支 I: 其他兜底分支

动作顺序：

1. 如果 `revision_count > max_revisions`，终止
2. 否则如果已有 `structured_report`，跳到 `reviewer_agent`
3. 否则跳到 `planner_agent`

### 3.5 Supervisor 的受控点

当前 `Supervisor` 不是 LLM 决策器，而是显式规则路由器：

- 路由完全由 `message_type` 和几个计数器决定
- agent 之间不能自由协商目标
- 新分支必须通过新增消息类型和规则代码来实现

## 4. PlannerAgent

### 4.1 角色定位

`PlannerAgent` 负责把研究主题拆成结构化任务图。

它的 subgraph 很简单：

```text
planner
  -> planner_handoff
  -> END
```

### 4.2 输入

- `research_topic`
- `config`

### 4.3 内部处理流程

#### 步骤 1: 组装 system prompt

Planner 会对系统提示词做清洗：

- 去掉旧框架的 `<NOTE_COLLAB>` 内容
- 去掉 `<TOOLS>` 段
- 去掉“必须调用 note 工具”的旧要求
- 追加当前实现真正需要的字段要求：
  - `priority`
  - `depends_on`
  - `search_budget`
  - `search_type`

#### 步骤 2: 组装 user prompt

使用当前日期和 `research_topic` 生成拆题提示。

#### 步骤 3: 调用 LLM

- 使用 `strategic_llm`
- `temperature = 0`
- 目标是返回任务 JSON

#### 步骤 4: 解析 JSON

解析顺序：

- 先尝试 fenced JSON
- 再尝试完整文本
- 再尝试 `{...}` 片段
- 再尝试 `[...]` 片段

如果解析成功：

- 提取 `tasks`
- 只保留字典项

#### 步骤 5: 标准化任务

对每个任务补齐和修正：

- `id`: 非法则自动重编
- `title`: 缺失时回退为 `任务N`
- `intent`: 缺失时回退为默认目标
- `query`: 缺失时回退为 `research_topic`
- `priority`: 缺失时按顺序编号
- `depends_on`: 只保留正整数，且去重
- `search_budget`: 缺失时回退为 `deep_research_depth`
- `search_type`: 缺失时默认 `"search"`
- `status`: 固定初始化为 `pending`

#### 步骤 6: 空结果 fallback

如果最终没有可用任务：

- 自动生成一个兜底任务 `基础背景梳理`

### 4.4 输出

Planner 节点输出：

- `todo_items`
- `agent_role`

其中 `agent_role` 是一段面向系统的角色说明文本。

### 4.5 handoff

`planner_handoff` 会向 `supervisor` 发送：

- `type = "task_assignment"`
- payload:
  - `tasks`
  - `task_count`

### 4.6 Planner 的分支总结

- LLM 正常返回可解析任务: 使用这些任务
- LLM 返回非法或空内容: 走 fallback 单任务

Planner 当前没有更复杂的自治回路，它的核心职责是一次性拆题。

## 5. ResearcherAgent

### 5.1 角色定位

`ResearcherAgent` 负责完成单个任务的搜证、压缩、覆盖度评估、query 改写、摘要生成和证据抽取。

它的 subgraph：

```text
task_node
  -> researcher_handoff
  -> END
```

虽然 graph 结构只有一个 node，但 `task_node` 内部本身是一个小闭环，因此它是当前最像 mini-agent 的角色。

### 5.2 输入

- `task`
- `config`
- `research_topic`
- `visited_urls`
- `research_loop_count`

### 5.3 初始化逻辑

Researcher 在开始时会确定：

- `current_query`: 优先取 `task.query`，否则回退到 `research_topic`
- `tried_queries`: 初始包含 `current_query`
- `max_iterations`:
  - 优先取 `task.search_budget`
  - 否则取 `runtime_config.researcher_max_iterations`
  - 再否则取配置默认值
  - 至少为 1
- `coverage_threshold`:
  - 优先取 `runtime_config.researcher_coverage_threshold`
  - 否则取配置默认值
  - 约束在 0 到 1

### 5.4 单轮研究循环

每轮循环都执行以下步骤：

```text
search
  -> URL 去重
  -> 聚合历史结果
  -> 构造上下文
  -> 覆盖度评估
  -> 如不足则重写 query
  -> 下一轮
```

#### 步骤 1: 调用搜索

每轮调用：

- `dispatch_search(current_query, config, loop_count)`

异常分支：

- 如果是第 1 轮就失败，直接抛错，整个任务失败
- 如果是后续轮次失败，只记录 warning，然后提前结束循环，保留前面已得到的结果

#### 步骤 2: URL 去重

Researcher 会把当前搜索结果与全局 `visited_urls` 去重：

- 已访问过的 URL 不再重复加入
- 新 URL 会并入 `updated_urls`

#### 步骤 3: 聚合结果

Researcher 不会只看当前轮结果，而是累计：

- `all_filtered_results`
- `all_notices`
- `backend_label`
- `first_answer_text`

其中 `first_answer_text` 只记录第一次拿到的直接答案文本，后续不会覆盖。

#### 步骤 4: 上下文构造

Researcher 会把累计结果转成页面集合，再构造 context。

分支如下：

- 如果总文本长度小于阈值，直接 fast path 格式化
- 如果文本过长，尝试用 embeddings + `ContextCompressor` 做压缩
- 如果压缩失败，回退到 fast path
- 如果仍无 context，再回退到 `prepare_research_context`

最后还会：

- 把第一次搜索返回的 `answer_text` 前置到 context 顶部

#### 步骤 5: 是否到达最后一轮

如果当前已经是最后一轮：

- 不再做覆盖度评估
- 直接退出循环，进入总结阶段

#### 步骤 6: 覆盖度评估

如果还没到最后一轮，则调用 `_assess_coverage`：

- 使用 `smart_llm`
- 输入任务目标、原始 query、当前上下文
- 返回：
  - `coverage_score`
  - `is_sufficient`
  - `unresolved_questions`
  - `reasoning`

分支如下：

- 如果 `is_sufficient == true`，停止循环
- 如果 `coverage_score >= threshold`，停止循环
- 如果没有 `unresolved_questions`，停止循环
- 否则进入 query 改写

#### 步骤 7: Query 改写

当存在未覆盖问题时，调用 `_rewrite_query`：

- 输入当前任务、未覆盖问题、已尝试查询列表
- 使用 `smart_llm`
- 输出一个更具体的新 query

分支如下：

- 如果新 query 为空，停止循环
- 如果新 query 已尝试过，停止循环
- 否则：
  - 把新 query 加入 `tried_queries`
  - `current_query = new_query`
  - 进入下一轮搜索

### 5.5 研究完成后的收尾

循环结束后，Researcher 会继续做以下动作：

#### 动作 1: 生成来源摘要

- 基于累计结果生成 `sources_summary`

#### 动作 2: 补最终 context

如果循环结束后 `context` 仍为空：

- 用 `prepare_research_context` 再兜底一次

#### 动作 3: 生成任务摘要

调用 `_generate_task_summary`：

- 使用 `smart_llm`
- 基于任务上下文和来源概览输出 Markdown 摘要

#### 动作 4: 标记任务完成

更新该任务：

- `status = "completed"`
- 写入 `summary`
- 写入 `sources_summary`

注意：

- 任务里记录回去的 `query` 仍是原始 `task.query`
- 中间迭代改写出来的 query 只服务于本轮 research，不会回写到 todo item

#### 动作 5: 构造证据

先基于搜索结果生成基础证据项：

- `task_id`
- `url`
- `title`
- `snippet`
- `relevance_score`

然后调用 `_extract_claims`：

- 从任务摘要中抽取关键论断
- 尝试把论断绑定到具体 URL
- 给证据补上：
  - `claim_text`
  - `support_type`
  - `section_hint`

异常分支：

- 如果 claim 抽取失败，保留原始基础证据项，不阻塞主流程

### 5.6 输出

Researcher 节点输出：

- `research_data`
- `evidence_store`
- `visited_urls`
- `todo_items`

### 5.7 handoff

`researcher_handoff` 会按任务 id 汇总该任务结果，然后向 `supervisor` 发送：

- `type = "evidence_delivery"`
- payload:
  - `task_id`
  - `title`
  - `query`
  - `summary`
  - `source_count`
  - `evidence_count`

### 5.8 Researcher 的关键分支总结

- 首轮搜索失败: 任务直接失败
- 后续搜索失败: 提前结束循环，使用已有结果
- 覆盖度足够: 提前停止
- 没有未覆盖问题: 提前停止
- query 改写失败或重复: 提前停止
- 压缩失败: 回退 fast path
- claim 抽取失败: 保留原始证据

## 6. WriterAgent

### 6.1 角色定位

`WriterAgent` 负责两种工作：

- 从任务研究结果生成完整报告
- 根据 reviewer 的 patch plan 定向修改已有章节

它的 subgraph：

```text
writer
  -> writer_handoff
  -> END
```

### 6.2 输入

- `research_topic`
- `todo_items`
- `research_data`
- `evidence_store`
- `review_result`
- `structured_report`
- `config`

### 6.3 顶层分支

进入 `writer_node` 后先判断：

- 如果 `review_result.section_patch_plan` 存在，且 `structured_report` 非空
  - 走 patch 路径
- 否则
  - 走全文写作路径

### 6.4 全文写作路径

#### 步骤 1: 构造写作上下文

Writer 会把以下内容拼进 prompt：

- 研究主题
- 每个任务的：
  - 标题
  - 目标
  - 查询
  - 任务总结
  - 来源概览
  - notices
  - 原始上下文
- 全局证据库
- 如存在审查反馈，还会加入：
  - reviewer 总体反馈
  - missing_topics
  - research_briefs
  - weak_sections
  - section_patch_plan
  - 上一版报告结构或摘要

#### 步骤 2: 流式写作

调用 `smart_llm` 生成全文：

- 先尝试 `stream=True`
- 每拿到一个 token，就发 `report_chunk` 事件

#### 步骤 3: 写作 fallback

如果流式失败：

- 若此前尚未拿到任何 chunk，则退化为普通非流式生成
- 若已经拿到部分 chunk，则直接使用已收集内容

#### 步骤 4: 清洗并补参考来源

生成结果会：

- 去掉 thinking tokens
- 去掉 tool call 痕迹
- 自动检查是否已有参考来源章节
- 如果没有，则根据 `evidence_store` 追加 `## 参考来源`

### 6.5 Patch 路径

Patch 路径会串行处理 `section_patch_plan` 中的每一项。

对每个 patch item：

1. 读取 `section / issue / instruction`
2. 如果缺字段，跳过
3. 在当前报告中查找该章节范围
4. 如果找不到对应章节，跳过
5. 把以下内容发给模型：
   - 最新任务研究结果
   - 最新证据库
   - 完整旧报告
   - 当前章节的问题
   - 当前章节的修改要求
6. 要求模型只输出修改后的该章节内容
7. 规范化章节文本
8. 替换回原报告

全部 patch 完成后：

- 再次确保参考来源章节存在

### 6.6 输出

Writer 节点输出：

- `structured_report`

`writer_handoff` 还会额外输出：

- `final_report = structured_report`

### 6.7 handoff

Writer 向 `supervisor` 发送：

- `type = "report_ready"`
- payload:
  - `report_length`
  - `patched`

其中 `patched` 的判断方式是：

- 当前 `review_result.section_patch_plan` 是否为非空列表

### 6.8 Writer 的关键分支总结

- 有 patch plan 且已有报告: 走定向 patch
- 否则: 走全文写作
- 流式失败且没有 chunk: 回退非流式生成
- 找不到目标章节: 跳过该 patch 项

## 7. ReviewerAgent

### 7.1 角色定位

`ReviewerAgent` 是质控角色，负责判断：

- 报告是否可交付
- 是否需要补研
- 是否只需局部改写
- 是否要整篇重写

它的 subgraph：

```text
reviewer
  -> reviewer_handoff
  -> END
```

### 7.2 输入

- `research_topic`
- `todo_items`
- `evidence_store`
- `structured_report`
- `review_result`
- `revision_count`
- `config`

### 7.3 审查步骤

#### 步骤 1: 构造审查快照

Reviewer 会把下面几类信息塞进 prompt：

- 当前研究主题
- 任务快照
- 证据快照
- 报告正文预览

#### 步骤 2: 调用 LLM 审查

- 使用 `strategic_llm`
- `temperature = 0`
- 要求只返回 JSON

Prompt 会强约束模型区分：

- `missing_topics`: 需要补搜索
- `weak_sections`: 有证据但写得差，只需重写
- `research_briefs`: 结构化补研单
- `section_patch_plan`: 定向改写计划
- `approved`: 是否通过

#### 步骤 3: 标准化输出

系统会对模型结果做清洗和规范化：

- `score` 转 float
- `research_briefs` 规范成 `topic / intent / query / priority`
- `section_patch_plan` 规范成 `section / issue / instruction`
- `missing_topics` 与 `research_briefs.topic/query` 合并去重
- `weak_sections` 规范成字符串列表

#### 步骤 4: 默认反馈兜底

如果：

- `approved == false`
- 且 `feedback` 为空

则系统自动写入默认反馈，避免后续没有解释性信息。

#### 步骤 5: 修订轮次增加

每次 reviewer 执行后都会：

- `revision_count += 1`

#### 步骤 6: 缺失主题转任务

如果：

- `approved == false`
- 且 `missing_topics` 非空

则 reviewer 会尝试把缺失主题转成新的 `todo_items`。

转换规则：

- 优先用 `research_briefs`
- 如果 `research_briefs` 不可用，再用原始 `missing_topics`
- 跳过与已有 `title/query` 重复的项
- 为新任务分配递增 id
- 状态初始化为 `pending`
- `priority` 从 `high/medium/low` 转成数值
- `search_budget` 取默认研究深度

如果成功生成任务：

- 更新 `todo_items`

### 7.4 reviewer_handoff 的消息分支

Reviewer 不直接跳转，而是先根据 `review_result` 映射消息类型。

#### 分支 A: 报告通过

条件：

- `approved == true`

发送：

- `type = "report_approved"`
- payload:
  - `score`
  - `feedback`

#### 分支 B: 需要补研

条件：

- `research_briefs` 非空，或 `missing_topics` 非空

发送：

- `type = "review_dispatch"`
- payload:
  - `research_briefs`
  - `missing_topics`
  - `section_patch_plan`

注意：

- 即使同时存在 `section_patch_plan`，只要还需要补研，消息类型仍然优先是 `review_dispatch`
- 也就是说，“先补证据，再写”是这里的优先级

#### 分支 C: 不补研，只做定向改写

条件：

- 没有 `research_briefs`
- 没有 `missing_topics`
- 但 `section_patch_plan` 非空

发送：

- `type = "patch_order"`
- payload:
  - `section_patch_plan`
  - `feedback`

#### 分支 D: 直接整篇重写

条件：

- 不通过
- 没有补研单
- 没有缺失主题
- 没有 patch plan

发送：

- `type = "rewrite_order"`
- payload:
  - `feedback`
  - `weak_sections`

### 7.5 Reviewer 的关键分支总结

- `approved = true`: 直接通过
- 有缺失主题: 转补研任务，发送 `review_dispatch`
- 无缺失主题但有 patch plan: 发送 `patch_order`
- 以上都没有但仍不通过: 发送 `rewrite_order`

## 8. 端到端流转图

### 8.1 主成功路径

```text
supervisor(init)
  -> planner_agent
  -> task_assignment
  -> supervisor
  -> researcher_agent x N
  -> evidence_delivery
  -> supervisor
  -> writer_agent
  -> report_ready
  -> reviewer_agent
  -> report_approved
  -> supervisor
  -> END
```

### 8.2 补研路径

```text
writer_agent
  -> reviewer_agent
  -> review_dispatch
  -> supervisor
  -> researcher_agent x N
  -> evidence_delivery
  -> supervisor
  -> writer_agent
  -> reviewer_agent
```

### 8.3 局部 patch 路径

```text
writer_agent
  -> reviewer_agent
  -> patch_order
  -> supervisor
  -> writer_agent(patch)
  -> reviewer_agent
```

### 8.4 整篇重写路径

```text
writer_agent
  -> reviewer_agent
  -> rewrite_order
  -> supervisor
  -> writer_agent(rewrite full report)
  -> reviewer_agent
```

### 8.5 超过修订上限路径

```text
reviewer_agent
  -> review_dispatch / patch_order / rewrite_order
  -> supervisor
  -> 如果 revision_count 超过阈值
  -> finalize_report
  -> END
```

## 9. 当前实现里最重要的“受控”与“自治”边界

### 9.1 已经具备的 agent 性

- 各角色是独立 subgraph
- 有显式消息总线
- Researcher 内部有自主搜证小闭环
- Reviewer 能输出结构化执行计划
- Writer 能局部 patch
- Supervisor 能并行 fan-out 多 researcher

### 9.2 仍然受控的部分

- 顶层路由不是 LLM 决策，而是规则 if/else
- agent 间消息类型是预定义的，不能自由扩展
- Researcher 的自主边界被 `search_budget`、`max_iterations`、`coverage_threshold` 限制
- Planner 仍是一次性拆题，不会主动回看执行结果重规划
- Writer 不会自主发起补证据请求，只会根据现有 `review_result` 执行

## 10. 一句话总结每个角色

- `Supervisor`: 规则路由器，决定下一个角色和是否终止
- `PlannerAgent`: 一次性拆题器，负责生成任务图
- `ResearcherAgent`: 单任务自主搜证器，内部有 search-assess-rewrite 小闭环
- `WriterAgent`: 报告生成器，支持全文生成和局部 patch
- `ReviewerAgent`: 质控调度器，决定通过、补研、patch 或重写
