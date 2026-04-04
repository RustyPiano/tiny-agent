# llm/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    """一次工具调用请求（从 LLM 响应中解析出来）"""

    id: str
    name: str
    inputs: dict
    parse_error: str | None = None
    raw_arguments: str | None = None


@dataclass
class LLMResponse:
    """统一的 LLM 响应，屏蔽各家 API 差异"""

    text: str  # 最终文本（stop_reason=end_turn 时有值）
    tool_calls: list[ToolCall]  # 工具调用列表（有值时进入下一轮）
    stop_reason: str  # "end_turn" | "tool_use"
    # assistant 消息的可序列化形式，用于追加进 messages 历史
    assistant_message: dict = field(default_factory=dict)


class BaseLLMProvider(ABC):
    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        system: str,
        tools: list[dict],
        max_tokens: int = 16000,
    ) -> LLMResponse:
        """
        发送对话，返回统一响应。
        messages 由 Context 维护，provider 在调用前自行转换格式。
        tools 为 Anthropic 格式的 schema list，provider 自行转换。
        """
        ...

    @abstractmethod
    def format_tool_result(self, tool_call_id: str, content: str) -> dict:
        """
        把工具执行结果格式化成该 provider messages 数组需要的格式。
        Anthropic 和 OpenAI 格式不同，各自实现。
        """
        ...

    @abstractmethod
    def tool_results_as_message(self, results: list[dict]) -> list[dict]:
        """
        把多条 tool_result 包装成可追加到 messages 的条目列表。
        Anthropic：返回 [{"role":"user","content":[...results]}]
        OpenAI：每条 result 是独立消息，直接返回 results 列表。
        """
        ...
