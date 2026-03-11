"""FastAPI entrypoint exposing the LangGraph research workflow via HTTP."""

from __future__ import annotations

import json
import sys
from typing import Any, AsyncIterator, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

from config import Configuration, SearchAPI
from event_mapping import map_langgraph_event, safe_int, serialize_todo_items
from graph.builder import build_graph

# 添加控制台日志处理程序
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <4}</level> | <cyan>using_function:{function}</cyan> | <cyan>{file}:{line}</cyan> | <level>{message}</level>",
    colorize=True,
)


# 添加错误日志文件处理程序
logger.add(
    sink=sys.stderr,
    level="ERROR",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <4}</level> | <cyan>using_function:{function}</cyan> | <cyan>{file}:{line}</cyan> | <level>{message}</level>",
    colorize=True,
)


class ResearchRequest(BaseModel):
    """Payload for triggering a research run."""

    topic: str = Field(..., description="Research topic supplied by the user")
    search_api: SearchAPI | None = Field(
        default=None,
        description="Override the default search backend configured via env",
    )
    research_depth: int | None = Field(
        default=None,
        ge=1,
        le=5,
        description="Override deep_research_depth: 1=light, 2=normal, 3=deep",
    )


class ResearchResponse(BaseModel):
    """HTTP response containing the generated report and structured tasks."""

    report_markdown: str = Field(
        ..., description="Markdown-formatted research report including sections"
    )
    todo_items: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Structured TODO items with summaries and sources",
    )


_GRAPH_APP: Any | None = None


def _mask_secret(value: Optional[str], visible: int = 4) -> str:
    """Mask sensitive tokens while keeping leading and trailing characters."""
    if not value:
        return "unset"

    if len(value) <= visible * 2:
        return "*" * len(value)

    return f"{value[:visible]}...{value[-visible:]}"


def _build_config(payload: ResearchRequest) -> Configuration:
    overrides: Dict[str, Any] = {}

    if payload.search_api is not None:
        overrides["search_api"] = payload.search_api

    if payload.research_depth is not None:
        overrides["deep_research_depth"] = payload.research_depth

    return Configuration.from_env(overrides=overrides)


def _get_graph() -> Any:
    global _GRAPH_APP
    if _GRAPH_APP is None:
        _GRAPH_APP = build_graph()
    return _GRAPH_APP


def _build_initial_state(payload: ResearchRequest, config: Configuration) -> dict[str, Any]:
    runtime_config = config.model_dump(mode="json")
    return {
        "research_topic": payload.topic.strip(),
        "todo_items": [],
        "research_loop_count": 0,
        "structured_report": "",
        "visited_urls": set(),
        "evidence_store": [],
        "research_data": [],
        "review_result": {},
        "revision_count": 0,
        "max_revisions": safe_int(runtime_config.get("max_revisions"), 2),
        "agent_role": "",
        "config": runtime_config,
    }


def create_app() -> FastAPI:
    app = FastAPI(title="Deep Researcher")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    def log_startup_configuration() -> None:
        config = Configuration.from_env()

        if config.llm_provider == "ollama":
            base_url = config.sanitized_ollama_url()
        elif config.llm_provider == "lmstudio":
            base_url = config.lmstudio_base_url
        else:
            base_url = config.llm_base_url or "unset"

        logger.info(
            "DeepResearch configuration loaded: provider=%s model=%s base_url=%s search_api=%s "
            "breadth=%s depth=%s concurrency=%s max_loops=%s max_revisions=%s "
            "smart_llm=%s strategic_llm=%s embedding_model=%s similarity_threshold=%s "
            "fetch_full_page=%s strip_thinking=%s api_key=%s",
            config.llm_provider,
            config.resolved_model() or "unset",
            base_url,
            (config.search_api.value if isinstance(config.search_api, SearchAPI) else config.search_api),
            config.deep_research_breadth,
            config.deep_research_depth,
            config.deep_research_concurrency,
            config.max_web_research_loops,
            config.max_revisions,
            config.smart_llm,
            config.strategic_llm,
            config.embedding_model,
            config.similarity_threshold,
            config.fetch_full_page,
            config.strip_thinking_tokens,
            _mask_secret(config.llm_api_key),
        )

    @app.get("/healthz")
    def health_check() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/research", response_model=ResearchResponse)
    async def run_research(payload: ResearchRequest) -> ResearchResponse:
        try:
            config = _build_config(payload)
            graph = _get_graph()
            result = await graph.ainvoke(_build_initial_state(payload, config))
        except ValueError as exc:  # Likely due to unsupported configuration
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive guardrail
            logger.exception("Research failed")
            raise HTTPException(status_code=500, detail="Research failed") from exc

        return ResearchResponse(
            report_markdown=str(result.get("structured_report") or "").strip(),
            todo_items=serialize_todo_items(result.get("todo_items")),
        )

    @app.post("/research/stream")
    async def stream_research(payload: ResearchRequest) -> StreamingResponse:
        try:
            config = _build_config(payload)
            graph = _get_graph()
            initial_state = _build_initial_state(payload, config)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        async def event_iterator() -> AsyncIterator[str]:
            done_sent = False
            try:
                async for event in graph.astream_events(initial_state, version="v2"):
                    for mapped_event in map_langgraph_event(event):
                        if mapped_event.get("type") == "done":
                            done_sent = True
                        yield f"data: {json.dumps(mapped_event, ensure_ascii=False)}\n\n"
            except Exception as exc:  # pragma: no cover - defensive guardrail
                logger.exception("Streaming research failed")
                error_payload = {"type": "error", "detail": str(exc)}
                yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"
                return

            if not done_sent:
                yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            event_iterator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
