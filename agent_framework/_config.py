# config.py
import os
import pathlib
from dataclasses import dataclass, field

# --- 常量 ---
MAX_TOKENS = 16000
MAX_TURNS = 50
OUTPUT_TRUNCATE = 8000
SESSIONS_DIR = "sessions"
SYSTEM_PROMPT_DYNAMIC_BOUNDARY = "__SYSTEM_PROMPT_DYNAMIC_BOUNDARY__"
MAX_FILE_READ_LINES = 2000
MAX_WRITE_FILE_CHARS = 64000
MAX_HISTORY_RECORDS = 15
MAX_MEMORY_LINES = 200
CONTEXT_SOFT_LIMIT_TOKENS = 160000
MAX_COMPACT_HISTORY_MESSAGES = 8

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def _parse_bool_env(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    return default


# 工作空间根目录（向后兼容，测试和 tools 仍在引用）
WORKSPACE_ROOT = pathlib.Path(os.getenv("AGENT_WORKSPACE", os.getcwd())).resolve()
PROJECT_SKILLS_DIR = WORKSPACE_ROOT / ".agents" / "skills"
GLOBAL_SKILLS_DIR = pathlib.Path.home() / ".agents" / "skills"

BASE_SYSTEM_PROMPT = """你是一个能力强大的本地 Agent。必须严格遵循以下静态契约：

## 输出协议（严格 ReAct JSON）
1. 你每一轮只允许输出一个 JSON object，不允许输出 JSON 之外的任何文字。
2. JSON 只能包含且必须包含以下三个键："thought"、"action"、"action_input"。
3. 键的约束：
   - "thought": string，简短说明当前决策依据。
   - "action": string，必须是当前运行时允许的工具名之一，或 "NONE"。
   - "action_input": object | "NONE"（联合类型）。
     - 当 action 为运行时允许的工具名时，"action_input" 必须为 object。
     - 当 action 为 "NONE" 时，"action_input" 必须且只能为 "NONE"。
4. 任何额外键、Markdown 包裹、代码块、解释性前后缀都视为协议违规。

## 安全规则
1. 禁止调用当前运行时允许集合之外的工具或伪造工具。
2. 禁止越权访问工作空间之外路径；文件操作必须遵守工具侧路径校验。
3. 对有副作用的操作，先最小化探查现状（例如 read_file、list_dir、run_bash("ls")）。
4. 禁止执行高风险、不可逆或提权意图命令；遇到风险时改为解释风险并给出安全替代方案。

## 上下文纪律
1. 严格控制上下文窗口：最多 50 轮、最多 15 条压缩历史、最多 200 行记忆。
2. 单次读取遵守上限（例如文件读取最多 2000 行），避免无边界展开。
3. 每次工具调用后必须基于观察结果再决定下一步，禁止盲目连续调用。

## 动态上下文边界
系统会在静态 prompt 后追加动态区域，边界标记为 __SYSTEM_PROMPT_DYNAMIC_BOUNDARY__。
你必须将该标记之后的内容视为可变运行时上下文，并在决策时优先利用最新动态信息。"""


@dataclass
class AgentSettings:
    """运行时配置，由 main.py 构建并注入 core/agent.py"""

    provider_type: str = "anthropic"
    model: str = "claude-opus-4-6"
    base_url: str | None = None
    api_key: str | None = None
    workspace_root: pathlib.Path = field(default_factory=lambda: WORKSPACE_ROOT)
    project_skills_dir: pathlib.Path = field(
        default_factory=lambda: WORKSPACE_ROOT / ".agents" / "skills"
    )
    global_skills_dir: pathlib.Path = field(
        default_factory=lambda: pathlib.Path.home() / ".agents" / "skills"
    )
    # session store 默认使用该路径；运行时可通过 settings 覆盖
    sessions_dir: pathlib.Path = field(default_factory=lambda: pathlib.Path(SESSIONS_DIR))
    max_tokens: int = MAX_TOKENS
    max_turns: int = MAX_TURNS
    output_truncate: int = OUTPUT_TRUNCATE
    context_soft_limit_tokens: int = CONTEXT_SOFT_LIMIT_TOKENS
    enable_subagent_flow: bool = False

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
        if self.project_skills_dir.exists() and not self.project_skills_dir.is_dir():
            errors.append(f"project_skills_dir 不是目录: {self.project_skills_dir}")
        if self.global_skills_dir.exists() and not self.global_skills_dir.is_dir():
            errors.append(f"global_skills_dir 不是目录: {self.global_skills_dir}")
        return errors

    @classmethod
    def from_env(cls) -> "AgentSettings":
        """从环境变量构建"""
        workspace_root = pathlib.Path(os.getenv("AGENT_WORKSPACE", str(WORKSPACE_ROOT))).resolve()
        project_default = workspace_root / ".agents" / "skills"
        global_default = pathlib.Path.home() / ".agents" / "skills"
        return cls(
            provider_type=os.getenv("AGENT_PROVIDER", cls.provider_type),
            model=os.getenv("AGENT_MODEL", cls.model),
            base_url=os.getenv("AGENT_BASE_URL"),
            api_key=os.getenv("AGENT_API_KEY"),
            workspace_root=workspace_root,
            project_skills_dir=pathlib.Path(
                os.getenv("AGENT_PROJECT_SKILLS_DIR", str(project_default))
            )
            .expanduser()
            .resolve(),
            global_skills_dir=pathlib.Path(
                os.getenv("AGENT_GLOBAL_SKILLS_DIR", str(global_default))
            )
            .expanduser()
            .resolve(),
            enable_subagent_flow=_parse_bool_env(
                os.getenv("AGENT_ENABLE_SUBAGENT_FLOW"),
                default=False,
            ),
        )

    def to_provider_config(self) -> dict:
        """转换为 provider factory 所需的 dict"""
        cfg = {"type": self.provider_type, "model": self.model}
        if self.base_url:
            cfg["base_url"] = self.base_url
        if self.api_key:
            cfg["api_key"] = self.api_key
        return cfg
