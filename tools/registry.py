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
                "required": required if required is not None else list(parameters.keys()),
            },
        },
        "handler": handler,
    }


def get_schemas() -> list[dict]:
    return [t["schema"] for t in _TOOLS.values()]


def execute(name: str, inputs: dict) -> str:
    if name not in _TOOLS:
        return f"[error] 未知工具: {name}"

    if not isinstance(inputs, dict):
        return f"[error] 工具参数必须是 object，实际: {type(inputs).__name__}"

    tool_info = _TOOLS[name]
    parse_error = inputs.get("_tool_parse_error")
    raw_arguments = inputs.get("_tool_raw_arguments")

    if parse_error:
        preview = str(raw_arguments)[:200] if raw_arguments is not None else ""
        return (
            "[error] 工具参数解析失败: "
            f"{parse_error}; raw_arguments={preview!r}"
        )

    schema = tool_info.get("schema", {})
    input_schema = schema.get("input_schema", {})
    required = input_schema.get("required", []) or []
    handler_inputs = {k: v for k, v in inputs.items() if not k.startswith("_tool_")}
    missing = [k for k in required if k not in handler_inputs]
    if missing:
        return f"[error] 缺少必填参数: {', '.join(missing)}"

    try:
        return str(tool_info["handler"](**handler_inputs))
    except Exception as e:
        return f"[error] 工具 {name} 执行失败: {e}"


def list_tools() -> list[str]:
    return list(_TOOLS.keys())
