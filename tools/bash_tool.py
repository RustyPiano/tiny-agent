# tools/bash_tool.py
import os
import re
import shlex
import subprocess

from config import OUTPUT_TRUNCATE
from tools.registry import register

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
]

_BLOCKED_TOKENS = ["sudo", "curl", "wget"]

_BLOCKED_TOKEN_PATTERNS = [
    (token, re.compile(rf"(^|[;&|\s]){token}(\s|$)", re.IGNORECASE))
    for token in _BLOCKED_TOKENS
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


def run_bash(command: str, timeout: int = 30, workdir: str | None = None) -> str:
    block_reason = _is_blocked(command)
    if block_reason:
        return f"[blocked] 命令被拒绝: {block_reason}\n原始命令: {command}"

    argv = _tokenize_command(command)
    if argv is None:
        return f"[error] 命令解析失败，可能包含不受支持的 shell 语法: {command}"
    if not argv:
        return "[error] 空命令，未执行"

    try:
        result = subprocess.run(
            argv,
            shell=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=workdir,
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr}"
        if not output.strip():
            output = f"[ok] 命令执行完毕，退出码 {result.returncode}，无输出"
        if len(output) > OUTPUT_TRUNCATE:
            output = output[:OUTPUT_TRUNCATE] + f"\n...[输出已截断，共 {len(output)} 字符]"
        return output
    except subprocess.TimeoutExpired:
        return f"[timeout] 命令在 {timeout}s 内未完成: {command}"
    except Exception as e:
        return f"[error] {e}"


def register_bash_tool() -> None:
    register(
        name="run_bash",
        description=(
            "在本地 shell 中执行命令并返回输出（stdout + stderr）。"
            "适合查看目录结构、运行脚本、执行测试、安装依赖等。"
            "不要用于不可逆的危险操作。"
        ),
        parameters={
            "command": {"type": "string", "description": "要执行的 shell 命令"},
            "timeout": {"type": "integer", "description": "超时秒数，默认 30"},
            "workdir": {"type": "string", "description": "工作目录路径，默认当前目录"},
        },
        required=["command"],
        handler=run_bash,
    )
