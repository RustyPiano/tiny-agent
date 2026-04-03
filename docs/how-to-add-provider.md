# 如何新增 Provider

本文档说明如何为 Agent Framework 添加新的 LLM Provider。

## 快速开始

### 1. 创建 Provider 类

在 `llm/` 目录下创建新文件，例如 `llm/gemini_provider.py`:

```python
# llm/gemini_provider.py
from llm.base import BaseLLMProvider, LLMResponse, ToolCall


class GeminiProvider(BaseLLMProvider):
    def __init__(self, model: str, api_key: str | None = None):
        self.model = model
        # 初始化 SDK 客户端
        # self.client = genai.Client(api_key=api_key)

    def chat(
        self,
        messages: list[dict],
        system: str,
        tools: list[dict],
    ) -> LLMResponse:
        """
        发送对话，返回统一响应。

        Args:
            messages: 对话历史，格式取决于 provider
            system: System prompt
            tools: 工具 schema 列表（Anthropic 格式）

        Returns:
            LLMResponse 包含:
            - text: 最终文本（stop_reason=end_turn 时有值）
            - tool_calls: 工具调用列表
            - stop_reason: "end_turn" 或 "tool_use"
            - assistant_message: 可序列化的 assistant 消息
        """
        # 1. 转换 messages 格式（如果需要）
        # 2. 转换 tools 格式（如果需要）
        # 3. 调用 API
        # 4. 解析响应，构建 LLMResponse

        # 示例返回
        return LLMResponse(
            text="回答文本",
            tool_calls=[],
            stop_reason="end_turn",
            assistant_message={"role": "assistant", "content": "回答文本"},
        )

    def format_tool_result(self, tool_call_id: str, content: str) -> dict:
        """
        把工具执行结果格式化成该 provider 需要的格式。

        不同 provider 格式不同:
        - Anthropic: {"type": "tool_result", "tool_use_id": id, "content": content}
        - OpenAI: {"role": "tool", "tool_call_id": id, "content": content}
        """
        # 根据你的 provider 格式返回
        return {"tool_call_id": tool_call_id, "content": content}

    def tool_results_as_message(self, results: list[dict]) -> list[dict]:
        """
        把多条 tool_result 包装成可追加到 messages 的条目列表。

        - Anthropic: 返回 [{"role":"user","content":[...results]}]
        - OpenAI: 每条 result 是独立消息，直接返回 results 列表
        """
        return results  # 或 [{"role": "user", "content": results}]
```

### 2. 注册到 Factory

在 `llm/factory.py` 中添加新 provider:

```python
def create_provider(cfg: dict) -> BaseLLMProvider:
    t = cfg.get("type", "anthropic")

    if t == "anthropic":
        from llm.anthropic_provider import AnthropicProvider
        return AnthropicProvider(model=cfg["model"], api_key=cfg.get("api_key"))

    if t == "openai":
        from llm.openai_provider import OpenAIProvider
        return OpenAIProvider(
            model=cfg["model"],
            base_url=cfg.get("base_url"),
            api_key=cfg.get("api_key"),
        )

    # 添加新 provider
    if t == "gemini":
        from llm.gemini_provider import GeminiProvider
        return GeminiProvider(model=cfg["model"], api_key=cfg.get("api_key"))

    raise ValueError(f"未知 provider 类型: {t!r}，支持: anthropic, openai, gemini")
```

### 3. 更新配置校验

在 `config.py` 的 `AgentSettings.validate()` 中添加支持的 provider:

```python
def validate(self) -> list[str]:
    errors = []
    if self.provider_type not in ("anthropic", "openai", "gemini"):  # 添加 gemini
        errors.append(f"不支持的 provider: {self.provider_type}")
    # ...
```

## 详细说明

### BaseLLMProvider 接口

```python
class BaseLLMProvider(ABC):
    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        system: str,
        tools: list[dict],
    ) -> LLMResponse:
        """发送对话，返回统一响应"""
        ...

    @abstractmethod
    def format_tool_result(self, tool_call_id: str, content: str) -> dict:
        """格式化单条工具执行结果"""
        ...

    @abstractmethod
    def tool_results_as_message(self, results: list[dict]) -> list[dict]:
        """把多条 tool_result 包装成 messages 条目"""
        ...
```

### LLMResponse 结构

```python
@dataclass
class LLMResponse:
    text: str                    # 最终文本
    tool_calls: list[ToolCall]   # 工具调用列表
    stop_reason: str             # "end_turn" 或 "tool_use"
    assistant_message: dict      # 可序列化的 assistant 消息
```

### ToolCall 结构

```python
@dataclass
class ToolCall:
    id: str       # 工具调用 ID
    name: str     # 工具名称
    inputs: dict  # 工具输入参数
```

## 格式转换参考

### Anthropic 格式

```python
# messages
[{"role": "user", "content": "你好"}]

# tools
[{
    "name": "tool_name",
    "description": "描述",
    "input_schema": {
        "type": "object",
        "properties": {"param": {"type": "string"}},
        "required": ["param"]
    }
}]

# tool_result
{"type": "tool_result", "tool_use_id": "id", "content": "结果"}

# tool_results message
[{"role": "user", "content": [tool_result1, tool_result2]}]
```

### OpenAI 格式

```python
# messages
[{"role": "user", "content": "你好"}]

# tools
[{
    "type": "function",
    "function": {
        "name": "tool_name",
        "description": "描述",
        "parameters": {
            "type": "object",
            "properties": {"param": {"type": "string"}},
            "required": ["param"]
        }
    }
}]

# tool_result
{"role": "tool", "tool_call_id": "id", "content": "结果"}

# tool_results message
[tool_result1, tool_result2]  # 每条是独立消息
```

## 完整示例：Ollama Provider

Ollama 兼容 OpenAI API，所以可以直接使用 OpenAI provider:

```bash
python main.py --provider openai --model qwen2.5:14b --base-url http://localhost:11434/v1
```

如果需要单独实现，可以创建:

```python
# llm/ollama_provider.py
from llm.openai_provider import OpenAIProvider


class OllamaProvider(OpenAIProvider):
    """Ollama 兼容 OpenAI API，继承 OpenAIProvider"""

    def __init__(self, model: str, base_url: str = "http://localhost:11434/v1", **kwargs):
        super().__init__(model=model, base_url=base_url, api_key="ollama")
```

## 测试 Provider

创建测试文件 `tests/test_gemini_provider.py`:

```python
from llm.gemini_provider import GeminiProvider
from llm.base import LLMResponse


def test_gemini_chat():
    provider = GeminiProvider(model="gemini-pro")
    response = provider.chat(
        messages=[{"role": "user", "content": "你好"}],
        system="你是一个助手",
        tools=[],
    )
    assert isinstance(response, LLMResponse)
    assert response.stop_reason in ("end_turn", "tool_use")


def test_gemini_format_tool_result():
    provider = GeminiProvider(model="gemini-pro")
    result = provider.format_tool_result("call_123", "执行结果")
    assert result["tool_call_id"] == "call_123"
    assert result["content"] == "执行结果"
```

## 最佳实践

1. **格式转换**: 在 `chat()` 内部完成消息和工具格式的转换
2. **错误处理**: 捕获 API 异常，返回有意义的错误信息
3. **重试机制**: 考虑添加 API 调用的重试逻辑
4. **Token 计算**: 如果 provider 支持，可以添加 token 使用量统计
5. **流式响应**: 如需支持流式，可扩展 BaseLLMProvider 接口

## 环境变量

新 provider 的 API Key 应遵循命名规范:

```bash
export GEMINI_API_KEY=xxx
```

在 `config.py` 中添加:

```python
@classmethod
def from_env(cls) -> "AgentSettings":
    return cls(
        # ...
        api_key=os.getenv("AGENT_API_KEY") or os.getenv("GEMINI_API_KEY"),
    )
```
