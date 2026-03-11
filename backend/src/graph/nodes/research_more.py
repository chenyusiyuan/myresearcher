from __future__ import annotations

from typing import Any


async def research_more_node(state: dict[str, Any]) -> dict[str, Any]:
    """Bump the supplemental research loop counter before fan-out."""
    return {"research_loop_count": int(state.get("research_loop_count", 0)) + 1}
