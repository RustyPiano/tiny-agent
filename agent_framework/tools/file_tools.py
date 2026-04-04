# tools/file_tools.py
import pathlib

from agent_framework import _config as config
from agent_framework.tools.registry import register


def _validate_path(path: str) -> tuple[pathlib.Path, str | None]:
    """验证路径是否在工作空间内，返回 (resolved_path, error_message)"""
    p = pathlib.Path(path)
    try:
        resolved = p.resolve()
    except Exception as e:
        return p, f"路径解析失败: {e}"

    # 运行时动态获取 WORKSPACE_ROOT（支持测试 monkeypatch）
    workspace_root = config.WORKSPACE_ROOT

    # 检查是否在工作空间内
    try:
        resolved.relative_to(workspace_root)
    except ValueError:
        return resolved, f"路径超出工作空间范围: {resolved}\n工作空间根目录: {workspace_root}"

    return resolved, None


def read_file(path: str, start_line: int | None = None, end_line: int | None = None) -> str:
    p, err = _validate_path(path)
    if err:
        return f"[error] {err}"

    if not p.exists():
        return f"[error] 文件不存在: {path}"
    if not p.is_file():
        return f"[error] 不是文件: {path}"
    try:
        content = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"[error] 无法以 UTF-8 读取（可能是二进制文件）: {path}"

    if start_line is not None or end_line is not None:
        lines = content.splitlines()
        if start_line is not None and start_line <= 0:
            return f"[error] start_line 必须大于 0，实际: {start_line}"
        if end_line is not None and end_line <= 0:
            return f"[error] end_line 必须大于 0，实际: {end_line}"
        if start_line is not None and end_line is not None and end_line < start_line:
            return (
                "[error] end_line 不能小于 start_line，"
                f"实际: start_line={start_line}, end_line={end_line}"
            )
        s = (start_line or 1) - 1
        e = end_line or len(lines)
        return "\n".join(f"{s + i + 1}\t{line}" for i, line in enumerate(lines[s:e]))

    lines_with_endings = content.splitlines(keepends=True)
    if len(lines_with_endings) > config.MAX_FILE_READ_LINES:
        return "".join(lines_with_endings[: config.MAX_FILE_READ_LINES])

    return content


def write_file(path: str, content: str, mode: str = "overwrite") -> str:
    p, err = _validate_path(path)
    if err:
        return f"[error] {err}"

    content_len = len(content)
    if content_len > config.MAX_WRITE_FILE_CHARS:
        return (
            "[error] write_file 内容过大: "
            f"{content_len} 字符，超过上限 {config.MAX_WRITE_FILE_CHARS} 字符。"
            "请改用 edit_file 做局部修改，或将内容分块（chunking）写入。"
        )

    p.parent.mkdir(parents=True, exist_ok=True)
    if mode == "append":
        with p.open("a", encoding="utf-8") as f:
            f.write(content)
        return f"[ok] 已追加 {len(content)} 字符到 {path}"
    else:
        p.write_text(content, encoding="utf-8")
        return f"[ok] 已写入 {len(content)} 字符到 {path}"


def register_file_tools() -> None:
    register(
        name="read_file",
        description=(
            "读取本地文件内容。可通过 start_line / end_line 截取片段，适合大文件避免 token 超限。"
        ),
        parameters={
            "path": {"type": "string", "description": "文件路径"},
            "start_line": {"type": "integer", "description": "起始行号（含，从 1 开始），可省略"},
            "end_line": {"type": "integer", "description": "结束行号（含），可省略"},
        },
        required=["path"],
        handler=read_file,
    )
    register(
        name="write_file",
        description=(
            "将文本内容写入本地文件。"
            "mode=overwrite 覆盖（默认），mode=append 追加。自动创建中间目录。"
        ),
        parameters={
            "path": {"type": "string", "description": "文件路径"},
            "content": {"type": "string", "description": "要写入的文本内容"},
            "mode": {"type": "string", "description": "'overwrite'（默认）或 'append'"},
        },
        required=["path", "content"],
        handler=write_file,
    )
