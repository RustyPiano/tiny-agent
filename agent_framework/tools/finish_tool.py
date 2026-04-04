import json

from agent_framework.tools.registry import register


def finish(response: str) -> str:
    return json.dumps({"response": response}, ensure_ascii=False)


def register_finish_tool() -> None:
    register(
        name="finish",
        description="返回最终响应的 JSON payload 字符串。",
        parameters={
            "response": {"type": "string", "description": "最终响应文本"},
        },
        required=["response"],
        handler=finish,
    )
