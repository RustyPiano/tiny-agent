from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass
class ReactDecision:
    thought: str
    action: str
    action_input: dict | str


def parse_react_json(raw: str, allowed_actions: set[str]) -> ReactDecision:
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("react decision must be valid JSON") from exc

    if not isinstance(payload, dict):
        raise ValueError("react decision must be a JSON object")

    expected_keys = {"thought", "action", "action_input"}
    if set(payload.keys()) != expected_keys:
        raise ValueError("react decision keys must equal {'thought', 'action', 'action_input'}")

    thought = payload["thought"]
    if not isinstance(thought, str) or not thought.strip():
        raise ValueError("react decision thought must be a non-empty string")

    action = payload["action"]
    if not isinstance(action, str) or action not in allowed_actions:
        raise ValueError("react decision action is not allowed")

    action_input = payload["action_input"]
    if not isinstance(action_input, dict) and action_input != "NONE":
        raise ValueError("react decision action_input must be dict or 'NONE'")

    return ReactDecision(thought=thought, action=action, action_input=action_input)
