from __future__ import annotations

import pytest

from graph.nodes import reviewer as reviewer_module


def test_build_review_prompt_renders_json_example() -> None:
    prompt = reviewer_module._build_review_prompt(
        {
            "research_topic": "test",
            "todo_items": [],
            "evidence_store": [],
            "structured_report": "sample report",
        }
    )

    assert '"research_briefs"' in prompt
    assert '"priority": "high"' in prompt
    assert "{\n" in prompt


@pytest.mark.asyncio
async def test_reviewer_node_parses_stubbed_review_response(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyMessage:
        def __init__(self, content: str) -> None:
            self.content = content

    class DummyChoice:
        def __init__(self, content: str) -> None:
            self.message = DummyMessage(content)

    class DummyResponse:
        def __init__(self, content: str) -> None:
            self.choices = [DummyChoice(content)]

    class DummyCompletions:
        async def create(self, **_: object) -> DummyResponse:
            return DummyResponse(
                '{"approved": true, "score": 0.9, "feedback": "ok", '
                '"missing_topics": [], "weak_sections": [], '
                '"research_briefs": [], "section_patch_plan": []}'
            )

    class DummyChat:
        def __init__(self) -> None:
            self.completions = DummyCompletions()

    class DummyClient:
        def __init__(self) -> None:
            self.chat = DummyChat()

    monkeypatch.setattr(
        reviewer_module,
        "_resolve_model_config",
        lambda *args, **kwargs: (None, "custom", "dummy"),
    )
    monkeypatch.setattr(
        reviewer_module,
        "_resolve_client",
        lambda *args, **kwargs: DummyClient(),
    )

    result = await reviewer_module.reviewer_node(
        {
            "research_topic": "test",
            "todo_items": [],
            "evidence_store": [],
            "structured_report": "sample report",
            "config": {},
            "revision_count": 0,
        }
    )

    assert result["review_result"]["approved"] is True
    assert result["review_result"]["feedback"] == "ok"
    assert result["revision_count"] == 1
