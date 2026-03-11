import os
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field


load_dotenv(Path(__file__).resolve().parents[1] / ".env")


class SearchAPI(Enum):
    PERPLEXITY = "perplexity"
    TAVILY = "tavily"
    DUCKDUCKGO = "duckduckgo"
    SEARXNG = "searxng"
    ADVANCED = "advanced"


class Configuration(BaseModel):
    """Configuration options for the deep research assistant."""

    deep_research_breadth: int = Field(
        default=3,
        title="Research Breadth",
        description="Number of parallel research tasks to explore per planning round",
    )
    deep_research_depth: int = Field(
        default=2,
        title="Research Depth",
        description="Number of recursive deep research rounds to allow per task",
    )
    researcher_max_iterations: int = Field(
        default=3,
        title="Researcher Max Iterations",
        description="Researcher 最大搜索迭代轮数",
    )
    researcher_coverage_threshold: float = Field(
        default=0.75,
        title="Researcher Coverage Threshold",
        description="覆盖度阈值，达到即停止搜索",
    )
    deep_research_concurrency: int = Field(
        default=4,
        title="Research Concurrency",
        description="Maximum number of research tasks to execute concurrently",
    )
    max_web_research_loops: int = Field(
        default=3,
        title="Web Research Loops",
        description="Number of web research iterations to perform",
    )
    similarity_threshold: float = Field(
        default=0.42,
        title="Similarity Threshold",
        description="Semantic similarity threshold used by context compression",
    )
    embedding_model: str = Field(
        default="openai:text-embedding-3-small",
        title="Embedding Model",
        description="Embedding provider and model selector used by compression",
    )
    max_revisions: int = Field(
        default=2,
        title="Max Revisions",
        description="Maximum number of reviewer-driven rewrite rounds",
    )
    smart_llm: str = Field(
        default="deepseek-chat",
        title="Smart LLM",
        description="Default model selector for task summarization and report writing",
    )
    strategic_llm: str = Field(
        default="deepseek-reasoner",
        title="Strategic LLM",
        description="Default model selector for planning and reviewing",
    )
    local_llm: str = Field(
        default="llama3.2",
        title="Local Model Name",
        description="Name of the locally hosted LLM (Ollama/LMStudio)",
    )
    llm_provider: str = Field(
        default="custom",
        title="LLM Provider",
        description="Provider identifier (ollama, lmstudio, or custom)",
    )
    search_api: SearchAPI = Field(
        default=SearchAPI.DUCKDUCKGO,
        title="Search API",
        description="Web search API to use",
    )
    fetch_full_page: bool = Field(
        default=True,
        title="Fetch Full Page",
        description="Include the full page content in the search results",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        title="Ollama Base URL",
        description="Base URL for Ollama API (without /v1 suffix)",
    )
    lmstudio_base_url: str = Field(
        default="http://localhost:1234/v1",
        title="LMStudio Base URL",
        description="Base URL for LMStudio OpenAI-compatible API",
    )
    strip_thinking_tokens: bool = Field(
        default=True,
        title="Strip Thinking Tokens",
        description="Whether to strip <think> tokens from model responses",
    )
    llm_api_key: Optional[str] = Field(
        default=None,
        title="LLM API Key",
        description="Optional API key when using custom OpenAI-compatible services",
    )
    llm_base_url: Optional[str] = Field(
        default="https://api.deepseek.com",
        title="LLM Base URL",
        description="Optional base URL when using custom OpenAI-compatible services",
    )
    llm_timeout_seconds: float = Field(
        default=30.0,
        title="LLM Timeout Seconds",
        description="HTTP timeout in seconds used for OpenAI-compatible LLM requests",
    )
    llm_model_id: Optional[str] = Field(
        default="deepseek-chat",
        title="LLM Model ID",
        description="Optional model identifier for custom OpenAI-compatible services",
    )

    @classmethod
    def from_env(cls, overrides: Optional[dict[str, Any]] = None) -> "Configuration":
        """Create a configuration object using environment variables and overrides."""

        raw_values: dict[str, Any] = {}

        # Load values from environment variables based on field names
        for field_name in cls.model_fields.keys():
            env_key = field_name.upper()
            if env_key in os.environ:
                raw_values[field_name] = os.environ[env_key]

        # Additional mappings for explicit env names
        env_aliases = {
            "deep_research_breadth": os.getenv("DEEP_RESEARCH_BREADTH"),
            "deep_research_depth": os.getenv("DEEP_RESEARCH_DEPTH"),
            "deep_research_concurrency": os.getenv("DEEP_RESEARCH_CONCURRENCY"),
            "local_llm": os.getenv("LOCAL_LLM"),
            "llm_provider": os.getenv("LLM_PROVIDER"),
            "llm_api_key": os.getenv("LLM_API_KEY"),
            "llm_model_id": os.getenv("LLM_MODEL_ID"),
            "llm_base_url": os.getenv("LLM_BASE_URL"),
            "llm_timeout_seconds": os.getenv("LLM_TIMEOUT_SECONDS"),
            "lmstudio_base_url": os.getenv("LMSTUDIO_BASE_URL"),
            "ollama_base_url": os.getenv("OLLAMA_BASE_URL"),
            "max_web_research_loops": os.getenv("MAX_WEB_RESEARCH_LOOPS"),
            "similarity_threshold": os.getenv("SIMILARITY_THRESHOLD"),
            "embedding_model": os.getenv("EMBEDDING_MODEL"),
            "max_revisions": os.getenv("MAX_REVISIONS"),
            "smart_llm": os.getenv("SMART_LLM"),
            "strategic_llm": os.getenv("STRATEGIC_LLM"),
            "fetch_full_page": os.getenv("FETCH_FULL_PAGE"),
            "strip_thinking_tokens": os.getenv("STRIP_THINKING_TOKENS"),
            "search_api": os.getenv("SEARCH_API"),
        }

        for key, value in env_aliases.items():
            if value is not None:
                raw_values.setdefault(key, value)

        if overrides:
            for key, value in overrides.items():
                if value is not None:
                    raw_values[key] = value

        return cls(**raw_values)

    def sanitized_ollama_url(self) -> str:
        """Ensure Ollama base URL includes the /v1 suffix required by OpenAI clients."""

        base = self.ollama_base_url.rstrip("/")
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        return base

    def resolved_model(self) -> Optional[str]:
        """Best-effort resolution of the model identifier to use."""

        return self.llm_model_id or self.local_llm
