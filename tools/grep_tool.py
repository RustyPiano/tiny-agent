import os
import pathlib
import re

import config
from tools.registry import register


def grep(pattern: str, path: str) -> str:
    root = pathlib.Path(path).resolve()
    workspace_root = config.WORKSPACE_ROOT

    def _is_within_workspace(candidate: pathlib.Path) -> bool:
        try:
            candidate.relative_to(workspace_root)
            return True
        except ValueError:
            return False

    if not _is_within_workspace(root):
        return f"[error] 路径超出工作空间范围: {root}"

    if not root.exists():
        return f"[error] 路径不存在: {path}"

    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"[error] 非法正则: {e}"

    matches: list[str] = []

    if root.is_file():
        files = [root]
    else:
        files = []
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            dir_path = pathlib.Path(dirpath)
            dirnames[:] = [
                name
                for name in sorted(dirnames)
                if not (dir_path / name).is_symlink()
            ]
            for name in sorted(filenames):
                files.append(dir_path / name)

    for file in files:
        try:
            resolved_file = file.resolve()
        except OSError:
            continue

        if not _is_within_workspace(resolved_file):
            continue

        try:
            with file.open("r", encoding="utf-8") as f:
                for idx, line in enumerate(f, start=1):
                    text = line.rstrip("\n")
                    if regex.search(text):
                        matches.append(f"{file}:{idx}:{text}")
        except UnicodeDecodeError:
            continue
        except OSError:
            continue

    return "\n".join(matches)


def register_grep_tool() -> None:
    register(
        name="grep",
        description="在文件或目录中按正则搜索文本并返回匹配行。",
        parameters={
            "pattern": {"type": "string", "description": "正则表达式"},
            "path": {"type": "string", "description": "文件或目录路径"},
        },
        required=["pattern", "path"],
        handler=grep,
    )
