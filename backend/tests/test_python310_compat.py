from __future__ import annotations


def test_graph_state_imports_under_python310() -> None:
    from graph import state as state_module

    assert state_module.TodoItem.__name__ == "TodoItem"
    assert state_module.ResearchState.__name__ == "ResearchState"
