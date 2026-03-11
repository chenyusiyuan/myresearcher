你现在仓库公开说明里，还是典型的 Planner → Task 并行 → Writer → Reviewer 研究图，Reviewer 负责补研或重写，整体更像一个中心化编排的研究工作流。

所以升级方向，不是笼统地“都变强”，而是看哪一类角色最值得拿到自主权。

我建议你这样理解：

1. 先区分两种升级

第一种：节点增强
还是一个总图，只是每个节点内部从“一步函数”变成“小闭环”。

比如现在的 researcher_node 可能是：
搜一次 → 抓一次 → 压缩一次 → 返回

升级后变成：
观察结果 → 判断够不够 → 改写 query → 再搜 → 继续/停止

这叫半自治，改动小，收益大。

第二种：真正 agent 化
每个角色变成独立 subgraph / react agent，有自己的状态、工具和 handoff 机制。
这才更接近你前面说的真正 multi-agent。

2. 哪些角色最值得先升级
A. 最该先升级：Researcher

因为你现在的核心瓶颈，本质上就在这里。你的文档已经写得很清楚：当前是 search-first，不是 browser-first；研究过程还是“搜索 URL → 过滤已访问 → 并发抓取 → 压缩片段”，而不是基于页面链接图展开。

所以第一个升级方向，不是先把 Planner 搞得很聪明，而是把 Researcher 从固定执行器升级成自主研究员。

它应该新增这些权能：

自己决定搜几轮

自己判断证据是否足够

自己改写 query

自己决定要不要换站点、换关键词、换角度

自己决定当前子题何时停止

必要时主动请求 browser worker 接管某个网页任务

它升级后的角色不再是：
“给我 subtopic，我去搜一下”

而是：
“我对这个 subtopic 负责，我会自己决定搜到什么程度为止”

这一步是收益最高的。

B. 第二该升级：Reviewer

你现在的 Reviewer 已经不只是打分器了，它能触发补研或重写，这其实已经很接近一个“审稿 agent”了。

下一步最值得做的是，把它从：

指出问题

升级成：

生成修复动作

也就是 Reviewer 不只是输出：

missing_topics

weak_sections

evidence不足

而是直接输出：

补哪个主题

去哪个站点补

哪一节需要补两条证据

哪一段要改成对比写法

哪个 claim 缺来源绑定

再进一步，它甚至可以直接 handoff：

给 Researcher 下补研单

给 Writer 下改写单

这样 Reviewer 就从“评委”变成“质控经理”。

C. 第三再升级：Writer

Writer 不要太早做成完全自治 agent，因为写作本身不是你最痛的地方。

但它可以拿到两类关键权能：

第一类：证据感知
不是拿到一坨 research_data 就写，而是：

检查每个 section 是否有足够 evidence

不足时主动请求补证据

写的时候按 evidence slot 绑定引用

第二类：Patch 能力
不要每次 reviewer 不满意就整篇重写。
更合理的是：

只重写某一节

只补写某一段

只把“结论”改成“共识/冲突/缺口”

这样 Writer 就从“整篇生成器”变成“可局部修补的写作 agent”。

D. Planner 最后再升级

很多人会直觉上先做 Planner，但其实 Planner 不是最优先。

因为你这个项目的问题不在“不会拆题”，而在“拆完之后研究不够自主、证据不够细、浏览不够深”。

Planner 可以升级，但建议控制住，不要让它过度自治。
比较合理的升级是：

从“生成 subtopics”

变成“生成 task graph”

也就是除了子题，还输出：

任务优先级

依赖关系

每个子题预算

哪些需要 browser

哪些只要 search

哪些必须跨站对比

这样 Planner 的升级是结构化调度增强，不是无限放权。

3. 你最值得做的四条升级路线
路线一：Researcher 内部先变成 mini-agent

这是最推荐、也最适合你当前代码的。

你可以先不动总图，只把 task_node / researcher_node 改成小循环：

plan_search -> search -> read -> assess_gap -> rewrite_query -> search ... -> stop

这里每轮都维护一个局部 state：

current_query

tried_queries

visited_urls

evidence_chunks

unresolved_questions

confidence / coverage

这一步做完，你的项目就从“固定 research node”变成“带自主搜证闭环的 research agent”。

这是最应该先做的第一步。

路线二：引入 EvidenceStore

你文档里已经把这个写成后续重点了：当前还是 visited_urls 级引用，不是 claim-level citation。

所以第二条升级方向是做：

URL 集合 → Evidence Store

每条证据保存：

url

title

chunk_text

section

timestamp

source_type

credibility

support / contradict / background

然后：

Writer 按 evidence 写

Reviewer 检查 evidence coverage

Researcher 看到证据冲突时继续补搜

这一条看起来不像“multi-agent”，但实际上它会极大增强 agent 协作，因为没有证据对象，多 agent 之间根本没法高质量协同。

路线三：把 Reviewer 做成主动路由者

也就是让 Reviewer 不只返回一个 JSON，而是返回：

need_more_research

research_briefs

rewrite_plan

section_patch_plan

stop_reason

然后图里不再只是：
reviewer -> writer / researcher

而是：
reviewer -> command(handoff to researcher_agent / writer_agent)

这时你就开始进入真正的多 agent 味道了。

路线四：新增 BrowserAgent，而不是整套系统全改 browser-first

你文档里已经明确列了缺的能力：页内链接提取、点击决策、next page、浏览轨迹、页面图扩展。

但我还是建议你：

不要一上来整体 browser-first。

最划算的做法是加一个 Browser Specialist Agent：

默认还是 search-first

遇到这些情况才 handoff 给 BrowserAgent：

搜索摘要不够

页面需要点进目录/分页

需要站内跳转

需要跨多页收集信息

某个站搜索引擎抓不到但网页里有

BrowserAgent 的动作集合就做成你文档里写的：
search/open/extract_links/click/next_page/stop

这样你不是“全盘重构”，而是“给多 agent 系统加了一个专职网页执行器”。

这是最现实的 browser 升级方式。

4. 如果按优先级排，我建议你这么升
第一层：最值

Researcher mini-agent 化

EvidenceStore

Reviewer 主动修复路由

这三步做完，你项目的含金量会明显提高，而且仍然可控。

第二层：进阶

Writer 局部 patch 化

Planner 输出 task graph 和预算

第三层：亮点增强

BrowserAgent

NavigationState

页内链接 / 分页 / 轨迹记忆

第四层：研究化

Mock Web / 合成页面图

benchmark 与可重复实验

5. 一句话说清“升级后到底变了什么”

升级前：

角色 = 共享状态上的 prompt 节点

升级后：

角色 = 有局部目标、局部状态、局部工具、局部停止条件，并能主动交接工作的 agent

所以不是简单“每个角色更强了”，而是：

每个角色从“执行步骤”变成“承担职责”。

6. 对你当前项目，最推荐的落地版本

我最建议你做的是这个版本：

Supervisor + ResearcherAgent×N + WriterAgent + ReviewerAgent + BrowserAgent(可选)
其中：

Supervisor 负责大框架和预算

ResearcherAgent 负责自主搜证

WriterAgent 负责基于 evidence 成文与 patch

ReviewerAgent 负责审查与派单

BrowserAgent 只处理必须点网页的任务

这比“所有角色都完全自治”更稳，也比“继续只是多节点流水线”更强。

你这个项目下一步最该做的，不是先追求“名义上真正 multi-agent”，而是先把 Researcher 的自主研究闭环 + Reviewer 的主动派单能力做出来。
这两点一出来，项目气质就会明显变。