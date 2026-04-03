# config.py
import os
import pathlib
from dataclasses import dataclass, field

# --- 常量 ---
MAX_TOKENS = 4096
MAX_TURNS = 20
OUTPUT_TRUNCATE = 8000
SESSIONS_DIR = "sessions"

# 工作空间根目录（向后兼容，测试和 tools 仍在引用）
WORKSPACE_ROOT = pathlib.Path(os.getenv("AGENT_WORKSPACE", os.getcwd())).resolve()

BASE_SYSTEM_PROMPT = """你是一个能力强大的本地 Agent。
你可以读写文件、执行 bash 命令来完成用户交代的任务。
执行有副作用的操作前，先通过 read_file 或 run_bash('ls') 了解现状，再动手。
每次工具调用后，根据结果决定下一步，不要盲目连续操作。"""


@dataclass
class AgentSettings:
    """运行时配置，由 main.py 构建并注入 core/agent.py"""

    provider_type: str = "anthropic"
    model: str = "claude-opus-4-6"
    base_url: str | None = None
    api_key: str | None = None
    workspace_root: pathlib.Path = field(default_factory=lambda: WORKSPACE_ROOT)
    sessions_dir: pathlib.Path = field(default_factory=lambda: pathlib.Path(SESSIONS_DIR))
    max_tokens: int = MAX_TOKENS
    max_turns: int = MAX_TURNS
    output_truncate: int = OUTPUT_TRUNCATE

    def validate(self) -> list[str]:
        """校验配置，返回错误列表"""
        errors = []
        if self.provider_type not in ("anthropic", "openai"):
            errors.append(f"不支持的 provider: {self.provider_type}")
        if not self.model:
            errors.append("model 不能为空")
        if self.max_tokens < 1:
            errors.append(f"max_tokens 必须 >= 1, 当前: {self.max_tokens}")
        if self.max_turns < 1:
            errors.append(f"max_turns 必须 >= 1, 当前: {self.max_turns}")
        if not self.workspace_root.exists():
            errors.append(f"workspace_root 不存在: {self.workspace_root}")
        return errors

    @classmethod
    def from_env(cls) -> "AgentSettings":
        """从环境变量构建"""
        return cls(
            provider_type=os.getenv("AGENT_PROVIDER", cls.provider_type),
            model=os.getenv("AGENT_MODEL", cls.model),
            base_url=os.getenv("AGENT_BASE_URL"),
            api_key=os.getenv("AGENT_API_KEY"),
            workspace_root=pathlib.Path(
                os.getenv("AGENT_WORKSPACE", str(WORKSPACE_ROOT))
            ).resolve(),
        )

    def to_provider_config(self) -> dict:
        """转换为 provider factory 所需的 dict"""
        cfg = {"type": self.provider_type, "model": self.model}
        if self.base_url:
            cfg["base_url"] = self.base_url
        if self.api_key:
            cfg["api_key"] = self.api_key
        return cfg
