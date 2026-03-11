"""Minimal prompt helpers reused by standalone research nodes."""

from __future__ import annotations

from typing import Any


def _coerce_doc(doc: Any) -> tuple[str, dict[str, Any]]:
    if isinstance(doc, dict):
        metadata = doc.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {
                "title": doc.get("title") or "",
                "source": doc.get("url") or doc.get("source") or "",
            }
        content = (
            doc.get("page_content")
            or doc.get("raw_content")
            or doc.get("content")
            or doc.get("snippet")
            or ""
        )
        return str(content).strip(), metadata

    content = getattr(doc, "page_content", "") or ""
    metadata = getattr(doc, "metadata", {}) or {}
    if not isinstance(metadata, dict):
        metadata = {}
    return str(content).strip(), metadata


class PromptFamily:
    """Small subset of GPT Researcher prompt formatting used in this project."""

    @staticmethod
    def pretty_print_docs(docs: list[Any], top_n: int = 5) -> str:
        parts: list[str] = []
        for index, doc in enumerate(docs[: max(top_n, 0)], start=1):
            content, metadata = _coerce_doc(doc)
            if not content:
                continue

            title = str(metadata.get("title") or f"来源 {index}").strip()
            source = str(metadata.get("source") or metadata.get("url") or "").strip()
            block = [f"来源 {index}: {title}"]
            if source:
                block.append(f"URL: {source}")
            block.append(f"内容:\n{content}")
            parts.append("\n".join(block))

        return "\n\n".join(parts).strip()
