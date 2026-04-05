# tools/bash_tool.py
import os
import pathlib
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile

from agent_framework import _config as config
from agent_framework._config import OUTPUT_TRUNCATE
from agent_framework.tools.registry import register

_BLOCKED_PATTERNS = [
    # 基础危险命令
    "mkfs",
    "dd if=",
    ":(){:|:&};:",
    "> /dev/sda",
    "> /dev/sdb",
    # fork bomb 变体
    ".(){.:|:&};.",
    # 磁盘分区
    "fdisk",
    "parted",
    # 下载内容直接管道到 shell（常见供应链攻击手法）
    "| sh",
    "| bash",
    "| zsh",
    "|sh",
    "|bash",
    "|zsh",
]

_BLOCKED_TOKENS = ["sudo"]

_BLOCKED_TOKEN_PATTERNS = [
    (token, re.compile(rf"(^|[;&|\s]){token}(\s|$)", re.IGNORECASE)) for token in _BLOCKED_TOKENS
]

# 使用正则匹配 rm -rf 的各种变体
_RM_RF_PATTERNS = [
    r"rm\s+.*-[^-]*r[^-]*f.*\s+/",  # rm -rf /
    r"rm\s+.*-[^-]*f[^-]*r.*\s+/",  # rm -fr /
    r"rm\s+--recursive\s+--force\s+/",  # rm --recursive --force /
    r"rm\s+--force\s+--recursive\s+/",  # rm --force --recursive /
    r"rm\s+-rf\s+~",  # rm -rf ~
    r"rm\s+-rf\s+\$HOME",  # rm -rf $HOME
]

_RM_RF_COMPILED = [re.compile(p) for p in _RM_RF_PATTERNS]
_DEFAULT_BASH_TIMEOUT_SEC = 30


def _tokenize_command(command: str) -> list[str] | None:
    try:
        return shlex.split(command, posix=True)
    except ValueError:
        return None


def _is_blocked(command: str) -> str | None:
    """检查命令是否被阻止，返回阻止原因或 None"""
    cmd_lower = command.lower().strip()

    # 检查基础模式
    for pattern in _BLOCKED_PATTERNS:
        if pattern.lower() in cmd_lower:
            return f"包含危险模式: '{pattern}'"

    for token, compiled_pattern in _BLOCKED_TOKEN_PATTERNS:
        if compiled_pattern.search(command):
            return f"包含受限命令: {token}"

    tokens = _tokenize_command(command)
    if tokens is not None:
        for raw_token in tokens:
            normalized = os.path.basename(raw_token).lower()
            if normalized in _BLOCKED_TOKENS:
                return f"包含受限命令: {normalized}"

    # 检查 rm -rf 变体
    for compiled_pattern in _RM_RF_COMPILED:
        if compiled_pattern.search(command):
            return f"匹配危险删除模式: {compiled_pattern.pattern}"

    # 检查对关键系统目录的删除
    critical_dirs = ["/", "/etc", "/usr", "/bin", "/sbin", "/boot", "/dev", "/proc", "/sys"]
    if re.search(r"rm\s+", command):
        for d in critical_dirs:
            # rm 后面跟关键目录（允许 ls 等只读操作）
            if re.search(rf"rm\s+.*\s+{re.escape(d)}(\s|$)", command):
                return f"尝试删除关键系统目录: {d}"

    return None


def _is_detached_command(command: str) -> bool:
    in_single_quote = False
    in_double_quote = False
    escaped = False

    for i, ch in enumerate(command):
        if escaped:
            escaped = False
            continue

        if ch == "\\":
            escaped = True
            continue

        if ch == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            continue

        if ch == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
            continue

        if in_single_quote or in_double_quote:
            continue

        if ch == "&":
            prev_char = command[i - 1] if i > 0 else ""
            next_char = command[i + 1] if i + 1 < len(command) else ""

            # 允许 &&
            if prev_char == "&" or next_char == "&":
                continue

            # 允许重定向相关形式：2>&1、>&2、&>file、<&0、|&
            if prev_char in {">", "<", "|"}:
                continue
            if next_char == ">":
                continue

            return True

    return False


def _select_timeout(command: str, timeout: int | float | str | None) -> int:
    _ = command
    if timeout is not None:
        try:
            requested = int(float(timeout))
            return min(requested, _DEFAULT_BASH_TIMEOUT_SEC)
        except (ValueError, TypeError):
            pass

    return _DEFAULT_BASH_TIMEOUT_SEC


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return

    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait(timeout=2.0)


def _check_timeout_binary(command: str) -> str | None:
    tokens = _tokenize_command(command)
    if not tokens:
        return None
    if os.path.basename(tokens[0]).lower() != "timeout":
        return None
    if shutil.which("timeout"):
        return None

    if sys.platform == "darwin":
        return (
            "[error] 未找到 `timeout` 命令。"
            "macOS 可先执行 `brew install coreutils`，然后改用 `gtimeout`。"
        )
    return "[error] 未找到 `timeout` 命令。请先安装 coreutils 或使用系统可用的超时工具。"


def _resolve_workspace_root(workspace_root: pathlib.Path | str | None = None) -> pathlib.Path:
    root_value = config.WORKSPACE_ROOT if workspace_root is None else workspace_root
    return pathlib.Path(root_value).resolve()


def _resolve_workdir(
    workdir: str | None, workspace_root: pathlib.Path | str | None = None
) -> tuple[str | None, str | None]:
    root = _resolve_workspace_root(workspace_root)
    if not root.exists() or not root.is_dir():
        return None, f"工作空间根目录不存在或不是目录: {root}"

    target = root if workdir is None else pathlib.Path(workdir).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        return None, f"工作目录超出工作空间范围: {target}\n工作空间根目录: {root}"
    if not target.exists() or not target.is_dir():
        return None, f"工作目录不存在或不是目录: {target}"
    return str(target), None


def _truncate_output(output: str) -> str:
    if len(output) <= OUTPUT_TRUNCATE:
        return output
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".log", prefix="agent_framework_bash_", delete=False
        ) as fp:
            fp.write(output)
            artifact_path = fp.name
        return (
            output[:OUTPUT_TRUNCATE]
            + f"\n...[输出已截断，共 {len(output)} 字符]"
            + f"\n完整输出已保存到: {artifact_path}"
        )
    except Exception:
        return output[:OUTPUT_TRUNCATE] + f"\n...[输出已截断，共 {len(output)} 字符]"


def run_bash(
    command: str,
    timeout: int | None = None,
    workdir: str | None = None,
    *,
    workspace_root: pathlib.Path | str | None = None,
) -> str:
    block_reason = _is_blocked(command)
    if block_reason:
        return f"[blocked] 命令被拒绝: {block_reason}\n原始命令: {command}"

    if not command.strip():
        return "[error] 空命令，未执行"

    if _is_detached_command(command):
        return (
            "[blocked] 不支持以 '&' 启动后台分离命令。"
            "请使用任务工具管理长任务：`run_job`（如当前环境已启用）。"
        )

    resolved_workdir, workdir_error = _resolve_workdir(workdir, workspace_root=workspace_root)
    if workdir_error is not None:
        return f"[error] {workdir_error}"

    timeout_error = _check_timeout_binary(command)
    if timeout_error:
        return timeout_error

    selected_timeout = _select_timeout(command, timeout)

    try:
        popen_kwargs: dict = {
            "shell": True,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "cwd": resolved_workdir,
        }
        if os.name == "posix":
            popen_kwargs["start_new_session"] = True

        process = subprocess.Popen(
            command,
            **popen_kwargs,
        )
        stdout, stderr = process.communicate(timeout=selected_timeout)
        output = stdout
        if stderr:
            output += f"\n[stderr]\n{stderr}"
        if not output.strip():
            output = f"[ok] 命令执行完毕，退出码 {process.returncode}，无输出"
        return _truncate_output(output)
    except subprocess.TimeoutExpired:
        _terminate_process_tree(process)
        return (
            f"[timeout] 命令在 {selected_timeout}s 内未完成: {command}\n"
            "该命令超出了前台短任务窗口，请改用 `run_job`。"
        )
    except Exception as e:
        return f"[error] {e}"


def register_bash_tool(workspace_root: pathlib.Path | str | None = None) -> None:
    def _handler(command: str, timeout: int | None = None, workdir: str | None = None) -> str:
        return run_bash(
            command,
            timeout=timeout,
            workdir=workdir,
            workspace_root=workspace_root,
        )

    register(
        name="run_bash",
        description=(
            "在本地 shell 中执行前台短命令并返回输出（stdout + stderr）。"
            "适合快速查看目录结构、执行短检查、运行短脚本。"
            "预计可能超过 30 秒的命令不要使用本工具，应改用 run_job。"
        ),
        parameters={
            "command": {"type": "string", "description": "要执行的 shell 命令"},
            "workdir": {"type": "string", "description": "工作目录路径，默认当前目录"},
        },
        required=["command"],
        handler=_handler,
    )
