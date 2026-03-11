"""Local embedding adapter layer for standalone context compression."""

from __future__ import annotations

import os
from typing import Any, Sequence

import requests
from openai import OpenAI


def _clean_texts(texts: Sequence[str]) -> list[str]:
    cleaned: list[str] = []
    for text in texts:
        value = str(text or "").strip()
        cleaned.append(value or " ")
    return cleaned


class _EmbeddingAdapter:
    def __init__(self, provider: str, model: str, **embedding_kwargs: Any) -> None:
        self.provider = provider
        self.model = model
        self.embedding_kwargs = dict(embedding_kwargs)
        self.timeout = float(self.embedding_kwargs.pop("timeout", 30))

    def embed_query(self, text: str) -> list[float]:
        embeddings = self.embed_documents([text])
        return embeddings[0] if embeddings else []

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        cleaned = _clean_texts(texts)
        if not cleaned:
            return []

        if self.provider in {"openai", "custom", "lmstudio"}:
            return self._embed_via_openai(cleaned)
        if self.provider == "ollama":
            return self._embed_via_ollama(cleaned)

        raise ValueError(f"Unsupported embedding provider: {self.provider}")

    def _embed_via_openai(self, texts: list[str]) -> list[list[float]]:
        client_kwargs: dict[str, Any] = {
            "api_key": self.embedding_kwargs.pop("openai_api_key", None)
            or os.getenv("OPENAI_API_KEY")
            or "EMPTY",
        }
        base_url = self.embedding_kwargs.pop("openai_api_base", None)
        if base_url:
            client_kwargs["base_url"] = base_url
        client_kwargs["timeout"] = self.timeout

        client = OpenAI(**client_kwargs)
        response = client.embeddings.create(
            model=self.model,
            input=texts,
            **self.embedding_kwargs,
        )
        return [list(item.embedding) for item in response.data]

    def _embed_via_ollama(self, texts: list[str]) -> list[list[float]]:
        base_url = (
            self.embedding_kwargs.pop("base_url", None)
            or os.getenv("OLLAMA_BASE_URL")
            or "http://localhost:11434"
        ).rstrip("/")

        response = requests.post(
            f"{base_url}/api/embed",
            json={"model": self.model, "input": texts},
            timeout=self.timeout,
        )
        if response.status_code == 404:
            return [self._embed_via_legacy_ollama(base_url, text) for text in texts]

        response.raise_for_status()
        payload = response.json()
        embeddings = payload.get("embeddings")
        if isinstance(embeddings, list) and embeddings and isinstance(embeddings[0], list):
            return [list(item) for item in embeddings]
        if isinstance(embeddings, list):
            return [list(embeddings)]
        raise ValueError("Ollama embedding response missing 'embeddings'.")

    def _embed_via_legacy_ollama(self, base_url: str, text: str) -> list[float]:
        response = requests.post(
            f"{base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        embedding = payload.get("embedding")
        if isinstance(embedding, list):
            return list(embedding)
        raise ValueError("Legacy Ollama embedding response missing 'embedding'.")


class Memory:
    """Compatibility wrapper matching the old ``Memory(...).get_embeddings()`` API."""

    def __init__(self, embedding_provider: str, model: str, **embedding_kwargs: Any) -> None:
        self._embeddings = _EmbeddingAdapter(embedding_provider, model, **embedding_kwargs)

    def get_embeddings(self) -> _EmbeddingAdapter:
        return self._embeddings
