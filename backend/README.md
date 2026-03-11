# Backend Setup

这个后端现在可以单独从 `helloagents-deepresearch/backend` 目录安装和运行，不再依赖外层仓库的 `gpt_researcher` 或 `hello_agents`。

## 1. 安装

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

如果你使用 `uv`：

```bash
cd backend
uv venv
source .venv/bin/activate
uv pip install -e .
```

## 2. 配置

复制环境变量模板：

```bash
cp .env.example .env
```

默认模板已经按 DeepSeek 配好了主模型：

- `LLM_PROVIDER=custom`
- `LLM_BASE_URL=https://api.deepseek.com`
- `SMART_LLM=deepseek-chat`
- `STRATEGIC_LLM=deepseek-reasoner`

只需要补上：

- `LLM_API_KEY`
- 可选搜索 API Key：`TAVILY_API_KEY` / `PERPLEXITY_API_KEY`
- 可选嵌入 API Key：`OPENAI_API_KEY`

说明：

- 如果没有配置可用嵌入模型，语义压缩会自动降级到快速路径，不会阻塞主流程。
- 如果使用 Ollama 或 LMStudio，请按 `.env.example` 中的注释切换 `LLM_PROVIDER` 和模型选择器。

## 3. 启动

```bash
cd backend
source .venv/bin/activate
python src/main.py
```

默认监听 `http://127.0.0.1:8000`。

## 4. 使用 LangGraph Studio

如果你要通过 `langgraph dev` 在 LangGraph Studio 里直接观察图流转，建议单独使用 Python 3.11 环境。

```bash
cd backend
python3.11 -m venv .venv-studio
source .venv-studio/bin/activate
pip install -e ".[studio]"
cp .env.example .env
# 在 .env 中至少填写 LLM_API_KEY 和 LANGSMITH_API_KEY
langgraph dev
```

当前仓库已包含：

- `langgraph.json`
- Studio 专用入口图：`src/langgraph_app.py`
- 最小 Studio 输入：`research_topic`
- 可选输入：`search_api`、`research_depth`

说明：

- `langgraph dev` / Studio 路径与 FastAPI 路径并存，互不影响。
- Studio 会读取 `backend/.env`，因此 LangSmith 与模型 Key 都放在同一个环境文件里即可。
- 如果你只想本地跑 API，不需要安装 `.[studio]`。

## 5. 接口

- `GET /healthz`
- `POST /research`
- `POST /research/stream`
