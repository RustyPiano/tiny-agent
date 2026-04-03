# tools/registry.py
from collections.abc import Callable

_TOOLS: dict[str, dict] = {}


def register(
    name: str,
    description: str,
    parameters: dict,
    handler: Callable,
    required: list[str] | None = None,
) -> None:
    _TOOLS[name] = {
        "schema": {
            "name": name,
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": parameters,
                "required": required or list(parameters.keys()),
            },
        },
        "handler": handler,
    }


def get_schemas() -> list[dict]:
    return [t["schema"] for t in _TOOLS.values()]


def execute(name: str, inputs: dict) -> str:
    if name not in _TOOLS:
        return f"[error] 未知工具: {name}"
    try:
        return str(_TOOLS[name]["handler"](**inputs))
    except Exception as e:
        return f"[error] 工具 {name} 执行失败: {e}"


def list_tools() -> list[str]:
    return list(_TOOLS.keys())
