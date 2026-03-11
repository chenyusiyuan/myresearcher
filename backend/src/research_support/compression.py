"""Local semantic compression helpers for the standalone backend."""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Any

from .prompts import PromptFamily


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    text = str(text or "").strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    step = max(chunk_size - chunk_overlap, 1)
    while start < len(text):
        chunks.append(text[start : start + chunk_size].strip())
        start += step
    return [chunk for chunk in chunks if chunk]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0

    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


@dataclass(slots=True)
class _Chunk:
    page_content: str
    metadata: dict[str, Any]


class ContextCompressor:
    """Compress search pages by embedding similarity without external repo deps."""

    def __init__(
        self,
        documents: list[dict[str, Any]],
        embeddings: Any,
        max_results: int = 5,
        prompt_family: type[PromptFamily] | PromptFamily = PromptFamily,
        similarity_threshold: float = 0.35,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        **_: Any,
    ) -> None:
        self.documents = documents
        self.embeddings = embeddings
        self.max_results = max_results
        self.prompt_family = prompt_family
        self.similarity_threshold = float(similarity_threshold)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _build_chunks(self) -> list[_Chunk]:
        chunks: list[_Chunk] = []
        for document in self.documents:
            raw_content = str(document.get("raw_content") or document.get("content") or "").strip()
            if not raw_content:
                continue

            metadata = {
                "title": str(document.get("title") or "Untitled Source").strip(),
                "source": str(document.get("url") or document.get("source") or "").strip(),
            }
            for chunk in _chunk_text(raw_content, self.chunk_size, self.chunk_overlap):
                chunks.append(_Chunk(page_content=chunk, metadata=metadata))
        return chunks

    def _select_relevant_docs(self, query: str, max_results: int) -> list[_Chunk]:
        chunks = self._build_chunks()
        if not chunks:
            return []

        query_embedding = self.embeddings.embed_query(query)
        chunk_embeddings = self.embeddings.embed_documents([chunk.page_content for chunk in chunks])

        scored_chunks = [
            (_cosine_similarity(query_embedding, embedding), chunk)
            for chunk, embedding in zip(chunks, chunk_embeddings)
        ]
        scored_chunks.sort(key=lambda item: item[0], reverse=True)

        filtered = [item for item in scored_chunks if item[0] >= self.similarity_threshold]
        selected = filtered or scored_chunks[:max_results]
        return [chunk for _, chunk in selected[:max_results]]

    async def async_get_context(self, query: str, max_results: int = 5, cost_callback=None) -> str:
        if cost_callback is not None:
            cost_callback(0)
        return await asyncio.to_thread(self._get_context, query, max_results)

    def _get_context(self, query: str, max_results: int) -> str:
        relevant_docs = self._select_relevant_docs(query, max_results)
        return self.prompt_family.pretty_print_docs(relevant_docs, top_n=max_results)
