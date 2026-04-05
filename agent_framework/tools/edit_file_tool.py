import pathlib

from agent_framework.tools.file_tools import _validate_path
from agent_framework.tools.registry import register


def edit_file(
    path: str,
    old_str: str,
    new_str: str,
    replace_all: bool = False,
    *,
    workspace_root: pathlib.Path | None = None,
) -> str:
    p, err = _validate_path(path, workspace_root=workspace_root)
    if err:
        return f"[error] {err}"

    if not p.exists():
        return f"[error] 文件不存在: {path}"
    if not p.is_file():
        return f"[error] 不是文件: {path}"

    if old_str == "":
        return "[error] old_str 不能为空"

    try:
        content = p.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        return f"[error] 文件不是有效的 UTF-8 文本: {exc}"

    count = content.count(old_str)

    if count == 0:
        return f"[error] old_str 未找到: {old_str}"
    if not replace_all and count > 1:
        return "[error] old_str 匹配多处，请设置 replace_all=true"

    updated = (
        content.replace(old_str, new_str) if replace_all else content.replace(old_str, new_str, 1)
    )
    try:
        p.write_text(updated, encoding="utf-8")
    except OSError as exc:
        return f"[error] 写入文件失败: {exc}"

    return f"[ok] 已更新文件: {path}"


def register_edit_file_tool(workspace_root: pathlib.Path | None = None) -> None:
    def _handler(path: str, old_str: str, new_str: str, replace_all: bool = False) -> str:
        return edit_file(
            path,
            old_str,
            new_str,
            replace_all=replace_all,
            workspace_root=workspace_root,
        )

    register(
        name="edit_file",
        description="在文件中查找 old_str 并替换为 new_str。",
        parameters={
            "path": {"type": "string", "description": "文件路径"},
            "old_str": {"type": "string", "description": "要查找的原始文本"},
            "new_str": {"type": "string", "description": "替换后的文本"},
            "replace_all": {
                "type": "boolean",
                "description": "是否替换所有匹配（默认 false 仅允许唯一匹配）",
            },
        },
        required=["path", "old_str", "new_str"],
        handler=_handler,
    )
