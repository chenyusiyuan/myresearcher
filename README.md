# Deep Researcher

这是一个可单独复制运行的 LangGraph 深度研究项目。当前版本已经移除了对外层仓库 `gpt_researcher` 和 `hello_agents` 的运行时依赖，直接在 `helloagents-deepresearch` 目录内即可安装、启动和部署。

## 功能

- Planner → Task 并行 → Writer → Reviewer 的完整研究图
- Reviewer 可触发补研或重写
- SSE 实时推送任务状态、来源、摘要、审查结果和最终报告
- DuckDuckGo / Tavily / Perplexity / SearXNG / Advanced 搜索调度
- OpenAI-compatible / Ollama / LMStudio 多模型接入
- 本地语义压缩实现，缺少嵌入能力时自动降级

## 快速开始

后端：

```bash
cd backend
pip install -e .
python src/main.py
```

前端：

```bash
cd frontend
npm install
npm run dev
```

后台启动前后端：

```bash
./scripts/start-all.sh
./scripts/status-all.sh
./scripts/stop-all.sh
```

也可以单独启动：

```bash
./scripts/start-backend.sh
./scripts/start-frontend.sh
```

说明：

- 后端日志：`logs/backend.log`
- 前端日志：`logs/frontend.log`
- PID 文件：`.run/backend.pid`、`.run/frontend.pid`
- 前端后台脚本会监听 `0.0.0.0:5174`，适合远程开发场景下手动做端口转发
- 后端后台脚本使用 `uvicorn` 非 `reload` 模式，适合长期驻留

LangGraph Studio：

```bash
cd backend
python3.11 -m venv .venv-studio
source .venv-studio/bin/activate
pip install -e ".[studio]"
cp .env.example .env
# 填写 LLM_API_KEY 和 LANGSMITH_API_KEY
langgraph dev
```

默认访问：

- 前端：`http://localhost:5174`
- 后端：`http://localhost:8000`

## 默认模型

项目默认按 DeepSeek 配置：

- `LLM_PROVIDER=custom`
- `LLM_BASE_URL=https://api.deepseek.com`
- `SMART_LLM=deepseek-chat`
- `STRATEGIC_LLM=deepseek-reasoner`

你只需要在 `backend/.env` 中填写 `LLM_API_KEY`。

如果要切换到 Ollama 或 LMStudio，直接参考 [backend/.env.example](/home/chenyusiyuan/researcher/helloagents-deepresearch/backend/.env.example) 里的示例即可。

## 文档

- 后端安装与启动：`backend/README.md`
- 环境变量模板：`backend/.env.example`
- 技术说明：`TECHNICAL_REPORT.md`
