import json
import pathlib

from agent_framework import _config as config
from agent_framework.tools.registry import register


_MAX_LIST_DIR_ENTRIES = 50


def _json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


def list_dir(
    path: str,
    offset: int = 0,
    limit: int | None = None,
    *,
    workspace_root: pathlib.Path | None = None,
) -> str:
    target = pathlib.Path(path).resolve()
    workspace_root = (
        workspace_root.resolve()
        if isinstance(workspace_root, pathlib.Path)
        else pathlib.Path(config.WORKSPACE_ROOT).resolve()
    )
    if not isinstance(offset, int) or offset < 0:
        return "[error] offset must be an integer >= 0"
    if limit is not None and (not isinstance(limit, int) or limit < 1):
        return "[error] limit must be an integer >= 1 or null"

    try:
        target.relative_to(workspace_root)
    except ValueError:
        return f"[error] 路径超出工作空间范围: {target}"

    if not target.exists():
        return f"[error] 路径不存在: {path}"
    if not target.is_dir():
        return f"[error] 不是目录: {path}"

    entries = sorted(p.name for p in target.iterdir())
    effective_limit = _MAX_LIST_DIR_ENTRIES if limit is None else min(limit, _MAX_LIST_DIR_ENTRIES)
    preview = entries[offset : offset + effective_limit]
    truncated = (offset + len(preview)) < len(entries)

    if offset == 0 and limit is None and not truncated:
        return "\n".join(entries)

    next_offset = offset + len(preview)
    payload = {
        "ok": True,
        "tool": "list_dir",
        "path": str(target),
        "offset": offset,
        "limit": effective_limit,
        "returned": len(preview),
        "next_offset": next_offset,
        "truncated": truncated,
        "preview": preview,
    }
    if truncated:
        payload["continuation"] = {
            "tool": "list_dir",
            "path": str(target),
            "offset": next_offset,
            "limit": effective_limit,
        }
    return _json(payload)


def register_list_dir_tool(workspace_root: pathlib.Path | None = None) -> None:
    def _handler(path: str, offset: int = 0, limit: int | None = None) -> str:
        return list_dir(path, offset=offset, limit=limit, workspace_root=workspace_root)

    register(
        name="list_dir",
        description="列出目录中的文件和子目录名称。",
        parameters={
            "path": {"type": "string", "description": "目录路径"},
            "offset": {
                "type": "integer",
                "description": "从第几个目录项开始返回，默认 0",
            },
            "limit": {
                "type": ["integer", "null"],
                "description": "单次返回的最大目录项数，默认值会自动截断",
            },
        },
        required=["path"],
        handler=_handler,
    )
