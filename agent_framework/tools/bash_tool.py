# tools/bash_tool.py
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

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
]

_BLOCKED_TOKENS = ["sudo", "curl", "wget"]

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
            if prev_char != "&" and next_char != "&":
                return True

    return False


def _select_timeout(command: str, timeout: int | None) -> int:
    if timeout is not None:
        return timeout

    cmd = command.strip().lower()

    build_install_patterns = [
        r"^cargo\s+(build|check|test)(\s|$)",
        r"^(npm|pnpm|yarn)\s+install(\s|$)",
        r"^pip\s+install(\s|$)",
    ]
    for pattern in build_install_patterns:
        if re.search(pattern, cmd):
            return 300

    test_patterns = [
        r"^pytest(\s|$)",
        r"^(npm|pnpm|yarn)\s+test(\s|$)",
        r"^go\s+test(\s|$)",
        r"^mvn\s+test(\s|$)",
        r"^gradle\s+test(\s|$)",
    ]
    for pattern in test_patterns:
        if re.search(pattern, cmd):
            return 180

    return 30


def _check_timeout_binary(command: str) -> str | None:
    tokens = _tokenize_command(command)
    if not tokens:
        return None
    if os.path.basename(tokens[0]).lower() != "timeout":
        return None
    if shutil.which("timeout"):
        return None

    if sys.platform == "darwin":
        return "[error] 未找到 `timeout` 命令。macOS 可先执行 `brew install coreutils`，然后改用 `gtimeout`。"
    return "[error] 未找到 `timeout` 命令。请先安装 coreutils 或使用系统可用的超时工具。"


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


def run_bash(command: str, timeout: int | None = None, workdir: str | None = None) -> str:
    block_reason = _is_blocked(command)
    if block_reason:
        return f"[blocked] 命令被拒绝: {block_reason}\n原始命令: {command}"

    if not command.strip():
        return "[error] 空命令，未执行"

    if _is_detached_command(command):
        return (
            "[blocked] 不支持以 '&' 启动后台分离命令。"
            "请使用任务工具管理长任务：`start_job`、`poll_job`、`read_job_log`、`cancel_job`（如当前环境已启用）。"
        )

    timeout_error = _check_timeout_binary(command)
    if timeout_error:
        return timeout_error

    selected_timeout = _select_timeout(command, timeout)

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=selected_timeout,
            cwd=workdir,
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr}"
        if not output.strip():
            output = f"[ok] 命令执行完毕，退出码 {result.returncode}，无输出"
        return _truncate_output(output)
    except subprocess.TimeoutExpired:
        return f"[timeout] 命令在 {selected_timeout}s 内未完成: {command}"
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
            "timeout": {
                "type": ["integer", "null"],
                "description": "可选超时秒数；不传时自动按命令类型选择",
            },
            "workdir": {"type": "string", "description": "工作目录路径，默认当前目录"},
        },
        required=["command"],
        handler=run_bash,
    )
