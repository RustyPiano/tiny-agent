import pathlib

import config
from tools.registry import register


def list_dir(path: str) -> str:
    target = pathlib.Path(path).resolve()
    workspace_root = config.WORKSPACE_ROOT
    try:
        target.relative_to(workspace_root)
    except ValueError:
        return f"[error] 路径超出工作空间范围: {target}"

    if not target.exists():
        return f"[error] 路径不存在: {path}"
    if not target.is_dir():
        return f"[error] 不是目录: {path}"

    entries = sorted(p.name for p in target.iterdir())
    return "\n".join(entries)


def register_list_dir_tool() -> None:
    register(
        name="list_dir",
        description="列出目录中的文件和子目录名称。",
        parameters={
            "path": {"type": "string", "description": "目录路径"},
        },
        required=["path"],
        handler=list_dir,
    )
