from __future__ import annotations

import json
from typing import Any


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def serialize_for_budget(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)


def estimate_payload_tokens(
    system_text: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    provider_type: str = "anthropic",
) -> int:
    parts: list[str] = []
    if provider_type == "openai":
        openai_messages = [{"role": "system", "content": system_text}, *messages]
        parts.extend(serialize_for_budget(message) for message in openai_messages)
        if tools:
            openai_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema", {}),
                    },
                }
                for tool in tools
            ]
            parts.extend(serialize_for_budget(tool) for tool in openai_tools)
        return estimate_tokens("\n".join(part for part in parts if part))

    parts.append(system_text)
    parts.extend(serialize_for_budget(message) for message in messages)
    if tools:
        parts.extend(serialize_for_budget(tool) for tool in tools)
    return estimate_tokens("\n".join(part for part in parts if part))


def should_compact(estimated_tokens: int, soft_limit: int) -> bool:
    return estimated_tokens >= soft_limit
