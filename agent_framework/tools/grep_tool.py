import json
import os
import pathlib
import re

from agent_framework import _config as config
from agent_framework.tools.registry import register

_MAX_GREP_MATCHES = 20
_MAX_GREP_LINE_CHARS = 200


def _json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _truncate_line(text: str) -> tuple[str, bool]:
    if len(text) <= _MAX_GREP_LINE_CHARS:
        return text, False
    return text[:_MAX_GREP_LINE_CHARS] + f"...[截断 {len(text)} 字符]", True


def _resolve_workspace_root(workspace_root: pathlib.Path | str | None = None) -> pathlib.Path:
    root_value = config.WORKSPACE_ROOT if workspace_root is None else workspace_root
    return pathlib.Path(root_value).resolve()


def grep(
    pattern: str,
    path: str,
    offset: int = 0,
    limit: int | None = None,
    *,
    workspace_root: pathlib.Path | str | None = None,
) -> str:
    root = pathlib.Path(path).resolve()
    workspace_root = _resolve_workspace_root(workspace_root)

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

    if not isinstance(offset, int) or offset < 0:
        return "[error] offset must be an integer >= 0"
    if limit is not None and (not isinstance(limit, int) or limit < 1):
        return "[error] limit must be an integer >= 1 or null"

    effective_limit = _MAX_GREP_MATCHES if limit is None else min(limit, _MAX_GREP_MATCHES)
    matches: list[str] = []
    matched_count = 0
    truncated = False
    preview_truncated = False

    if root.is_file():
        files = [root]
    else:
        files = []
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            dir_path = pathlib.Path(dirpath)
            dirnames[:] = [name for name in sorted(dirnames) if not (dir_path / name).is_symlink()]
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
                        preview_text, line_truncated = _truncate_line(text)
                        if matched_count < offset:
                            matched_count += 1
                            continue
                        if len(matches) < effective_limit:
                            matches.append(f"{file}:{idx}:{preview_text}")
                            matched_count += 1
                            preview_truncated = preview_truncated or line_truncated
                            continue
                        truncated = True
                        break
                if truncated:
                    break
        except UnicodeDecodeError:
            continue
        except OSError:
            continue

    if offset == 0 and limit is None and not truncated and not preview_truncated:
        return "\n".join(matches)

    next_offset = offset + len(matches)
    payload = {
        "ok": True,
        "tool": "grep",
        "pattern": pattern,
        "path": str(root),
        "offset": offset,
        "limit": effective_limit,
        "returned": len(matches),
        "next_offset": next_offset,
        "truncated": truncated,
        "preview_truncated": preview_truncated,
        "preview": matches,
    }
    if truncated:
        payload["continuation"] = {
            "tool": "grep",
            "pattern": pattern,
            "path": str(root),
            "offset": next_offset,
            "limit": effective_limit,
        }
    return _json(payload)


def register_grep_tool(workspace_root: pathlib.Path | str | None = None) -> None:
    def _handler(pattern: str, path: str, offset: int = 0, limit: int | None = None) -> str:
        return grep(
            pattern,
            path,
            offset=offset,
            limit=limit,
            workspace_root=workspace_root,
        )

    register(
        name="grep",
        description="在文件或目录中按正则搜索文本并返回匹配行。",
        parameters={
            "pattern": {"type": "string", "description": "正则表达式"},
            "path": {"type": "string", "description": "文件或目录路径"},
            "offset": {
                "type": "integer",
                "description": "从第几个匹配结果开始返回，默认 0",
            },
            "limit": {
                "type": ["integer", "null"],
                "description": "单次返回的最大匹配数，默认值会自动截断",
            },
        },
        required=["pattern", "path"],
        handler=_handler,
    )
