from __future__ import annotations

from agent_framework.tools.registry import register as register_tool


def _hello_tool(name: str = "world") -> str:
    return f"hello, {name}"


def register() -> None:
    register_tool(
        name="hello_tool",
        description="Return a greeting message for a name.",
        parameters={
            "name": {"type": "string", "description": "Target name. Default is world."},
        },
        required=[],
        handler=_hello_tool,
    )
