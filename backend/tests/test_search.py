from __future__ import annotations

from config import Configuration
from services import search


class _StubDDGSClient:
    def __init__(self, responses: dict[str, object]) -> None:
        self._responses = responses
        self.calls: list[tuple[str, int, str]] = []

    def text(self, query: str, *, max_results: int, backend: str):
        self.calls.append((query, max_results, backend))
        response = self._responses[backend]
        if isinstance(response, Exception):
            raise response
        return response


def test_search_duckduckgo_falls_back_to_next_backend(monkeypatch) -> None:
    client = _StubDDGSClient(
        {
            "primary": RuntimeError("primary backend down"),
            "secondary": [
                {
                    "title": "Ragdoll temperament guide",
                    "href": "https://example.com/ragdoll",
                    "body": "Gentle and affectionate.",
                }
            ],
        }
    )
    monkeypatch.setenv("DDGS_TEXT_BACKENDS", "primary,secondary")
    monkeypatch.setattr(search, "_get_ddgs_client", lambda: client)
    monkeypatch.setattr(search, "_fill_page_content", lambda results, _: results)

    payload = search._search_duckduckgo("ragdoll cat", Configuration(fetch_full_page=False))

    assert payload["backend"] == "duckduckgo:secondary"
    assert [call[2] for call in client.calls] == ["primary", "secondary"]
    assert payload["results"] == [
        {
            "title": "Ragdoll temperament guide",
            "url": "https://example.com/ragdoll",
            "snippet": "Gentle and affectionate.",
            "content": "Gentle and affectionate.",
            "raw_content": "Gentle and affectionate.",
        }
    ]
    assert payload["notices"] == ["ddgs:primary 搜索失败：primary backend down"]


def test_search_duckduckgo_returns_notice_when_all_backends_fail(monkeypatch) -> None:
    client = _StubDDGSClient(
        {
            "primary": RuntimeError("primary backend down"),
            "secondary": RuntimeError("secondary backend down"),
        }
    )
    monkeypatch.setenv("DDGS_TEXT_BACKENDS", "primary,secondary")
    monkeypatch.setattr(search, "_get_ddgs_client", lambda: client)

    payload = search._search_duckduckgo("ragdoll cat", Configuration(fetch_full_page=False))

    assert payload["backend"] == "duckduckgo:fallback_exhausted"
    assert payload["results"] == []
    assert payload["notices"] == [
        "ddgs:primary 搜索失败：primary backend down",
        "ddgs:secondary 搜索失败：secondary backend down",
        "所有 DDGS 搜索后端均失败或未返回可用结果",
    ]
