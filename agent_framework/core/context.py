# core/context.py


class Context:
    """管理单次 Agent 运行的 messages 数组。"""

    def __init__(self, initial_messages: list[dict] | None = None) -> None:
        self._messages: list[dict] = initial_messages or []

    def add_user(self, text: str) -> None:
        self._messages.append({"role": "user", "content": text})

    def add_assistant(self, assistant_message: dict) -> None:
        """追加 provider 序列化好的 assistant 消息（含 tool_use block）。"""
        self._messages.append(assistant_message)

    def add_tool_results(self, results: list[dict]) -> None:
        """
        追加工具结果。
        Anthropic：results 是 content list，打包成一条 user 消息。
        OpenAI：results 是多条独立的 role=tool 消息，逐条追加。
        由调用方（agent）根据 provider 类型决定传入形式。
        """
        for item in results:
            self._messages.append(item)

    def get(self) -> list[dict]:
        return self._messages

    def snapshot(self) -> list[dict]:
        """返回可 JSON 序列化的深拷贝（用于持久化）。"""
        import json
        from typing import Any

        result: list[dict[str, Any]] = json.loads(json.dumps(self._messages, default=str))
        return result

    def __len__(self) -> int:
        return len(self._messages)
