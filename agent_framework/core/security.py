from __future__ import annotations


class SecurityGuard:
    def __init__(self, allowed_tools: set[str]):
        self.allowed_tools = {name for name in allowed_tools if isinstance(name, str) and name}

    def validate_tool_call(self, name: str, action_input) -> tuple[bool, str]:
        _ = action_input
        if not self.allowed_tools:
            return False, "工具白名单为空，拒绝所有工具调用"
        if name not in self.allowed_tools:
            return False, f"工具未在白名单中: {name}"
        return True, "ok"
