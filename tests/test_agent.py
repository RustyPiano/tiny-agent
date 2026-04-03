# tests/test_agent.py
"""Agent 主循环测试，使用 mock provider"""

import pytest

from config import AgentSettings
from core.agent import run
from core.context import Context
from core.logging import RunContext
from core.policies import RuntimePolicy, Step
from core.runtime import AgentRuntime
from llm.base import BaseLLMProvider, LLMResponse, ToolCall


class MockProvider(BaseLLMProvider):
    """模拟 LLM provider，返回预定义响应"""

    def __init__(self, responses: list[LLMResponse]):
        self._responses = responses
        self._call_index = 0
        self.messages_received: list[list[dict]] = []

    def chat(self, messages, system, tools) -> LLMResponse:
        self.messages_received.append(messages)
        if self._call_index >= len(self._responses):
            raise IndexError("MockProvider: 响应已用完")
        resp = self._responses[self._call_index]
        self._call_index += 1
        return resp

    def format_tool_result(self, tool_call_id: str, content: str) -> dict:
        return {"type": "tool_result", "tool_use_id": tool_call_id, "content": content}

    def tool_results_as_message(self, results: list[dict]) -> list[dict]:
        return [{"role": "user", "content": results}]


class MockOpenAIProvider(BaseLLMProvider):
    """模拟 OpenAI provider"""

    def __init__(self, responses: list[LLMResponse]):
        self._responses = responses
        self._call_index = 0

    def chat(self, messages, system, tools) -> LLMResponse:
        if self._call_index >= len(self._responses):
            raise IndexError("MockOpenAIProvider: 响应已用完")
        resp = self._responses[self._call_index]
        self._call_index += 1
        return resp

    def format_tool_result(self, tool_call_id: str, content: str) -> dict:
        return {"role": "tool", "tool_call_id": tool_call_id, "content": content}

    def tool_results_as_message(self, results: list[dict]) -> list[dict]:
        return results


def test_simple_text_response():
    """测试简单的文本响应（无工具调用）"""
    provider = MockProvider(
        [
            LLMResponse(
                text="你好！有什么可以帮助你的？",
                tool_calls=[],
                stop_reason="end_turn",
                assistant_message={"role": "assistant", "content": "你好！有什么可以帮助你的？"},
            )
        ]
    )
    result = run("你好", provider=provider)
    assert result == "你好！有什么可以帮助你的？"


def test_single_tool_call():
    """测试单个工具调用"""
    provider = MockProvider(
        [
            # 第一轮：LLM 请求执行工具
            LLMResponse(
                text="",
                tool_calls=[
                    ToolCall(id="call_1", name="run_bash", inputs={"command": "echo hello"})
                ],
                stop_reason="tool_use",
                assistant_message={
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_1",
                            "name": "run_bash",
                            "input": {"command": "echo hello"},
                        }
                    ],
                },
            ),
            # 第二轮：LLM 根据工具结果回复
            LLMResponse(
                text="命令执行成功，输出为: hello",
                tool_calls=[],
                stop_reason="end_turn",
                assistant_message={"role": "assistant", "content": "命令执行成功，输出为: hello"},
            ),
        ]
    )
    result = run("执行 echo hello", provider=provider, verbose=True)
    assert "hello" in result


def test_multiple_tool_calls():
    """测试多个工具调用（一轮内多个）"""
    provider = MockProvider(
        [
            # 第一轮：LLM 同时请求执行两个工具
            LLMResponse(
                text="",
                tool_calls=[
                    ToolCall(id="call_1", name="run_bash", inputs={"command": "echo first"}),
                    ToolCall(id="call_2", name="run_bash", inputs={"command": "echo second"}),
                ],
                stop_reason="tool_use",
                assistant_message={
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_1",
                            "name": "run_bash",
                            "input": {"command": "echo first"},
                        },
                        {
                            "type": "tool_use",
                            "id": "call_2",
                            "name": "run_bash",
                            "input": {"command": "echo second"},
                        },
                    ],
                },
            ),
            # 第二轮：LLM 回复
            LLMResponse(
                text="两个命令都执行完毕",
                tool_calls=[],
                stop_reason="end_turn",
                assistant_message={"role": "assistant", "content": "两个命令都执行完毕"},
            ),
        ]
    )
    result = run("执行两个 echo 命令", provider=provider)
    assert "执行完毕" in result


def test_multiple_tool_calls_openai():
    """测试 OpenAI provider 的多工具调用"""
    provider = MockOpenAIProvider(
        [
            LLMResponse(
                text="",
                tool_calls=[
                    ToolCall(id="call_1", name="run_bash", inputs={"command": "echo first"}),
                    ToolCall(id="call_2", name="run_bash", inputs={"command": "echo second"}),
                ],
                stop_reason="tool_use",
                assistant_message={
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "run_bash",
                                "arguments": '{"command": "echo first"}',
                            },
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "run_bash",
                                "arguments": '{"command": "echo second"}',
                            },
                        },
                    ],
                },
            ),
            LLMResponse(
                text="完成",
                tool_calls=[],
                stop_reason="end_turn",
                assistant_message={"role": "assistant", "content": "完成"},
            ),
        ]
    )
    result = run("执行两个命令", provider=provider)
    assert result == "完成"


def test_max_turns_warning():
    """测试达到最大轮次时的警告"""
    # 创建一个会持续请求工具调用的 provider
    provider = MockProvider(
        [
            LLMResponse(
                text="",
                tool_calls=[
                    ToolCall(id=f"call_{i}", name="run_bash", inputs={"command": f"echo {i}"})
                ],
                stop_reason="tool_use",
                assistant_message={"role": "assistant", "content": []},
            )
            for i in range(25)  # 超过 MAX_TURNS=20
        ]
    )
    result = run("无限循环测试", provider=provider)
    assert "[warn]" in result
    assert "最大轮次" in result


def test_session_persistence(tmp_path, monkeypatch):
    """测试会话持久化"""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "sessions").mkdir(exist_ok=True)

    provider = MockProvider(
        [
            LLMResponse(
                text="记住：我的项目名叫 AgentX",
                tool_calls=[],
                stop_reason="end_turn",
                assistant_message={"role": "assistant", "content": "记住：我的项目名叫 AgentX"},
            )
        ]
    )
    result = run("记住：我的项目名叫 AgentX", provider=provider, session_id="test_session")
    assert "AgentX" in result

    # 检查会话文件是否创建
    assert (tmp_path / "sessions" / "test_session.json").exists()


def test_runtime_methods_are_override_friendly():
    """测试 runtime 子类可覆盖阶段行为"""

    class FailingProvider(BaseLLMProvider):
        def chat(self, messages, system, tools) -> LLMResponse:
            raise AssertionError("should not call provider.chat when call_llm is overridden")

        def format_tool_result(self, tool_call_id: str, content: str) -> dict:
            return {"type": "tool_result", "tool_use_id": tool_call_id, "content": content}

        def tool_results_as_message(self, results: list[dict]) -> list[dict]:
            return [{"role": "user", "content": results}]

    class OverrideRuntime(AgentRuntime):
        def call_llm(self) -> LLMResponse:
            return LLMResponse(
                text="override-ok",
                tool_calls=[],
                stop_reason="end_turn",
                assistant_message={"role": "assistant", "content": "override-ok"},
            )

    runtime = OverrideRuntime(
        provider=FailingProvider(),
        settings=AgentSettings(),
        ctx=Context(),
        tool_registry=None,
        session_store=None,
        system="test-system",
        run_ctx=RunContext(),
        session_id=None,
        provider_type="unknown",
    )

    assert runtime.run() == "override-ok"


def test_runtime_override_execute_tools_and_persist_session():
    """测试 execute_tools/persist_session 也可覆盖"""

    class NoopProvider(BaseLLMProvider):
        def chat(self, messages, system, tools) -> LLMResponse:
            raise AssertionError("should not call provider.chat when call_llm is overridden")

        def format_tool_result(self, tool_call_id: str, content: str) -> dict:
            return {"type": "tool_result", "tool_use_id": tool_call_id, "content": content}

        def tool_results_as_message(self, results: list[dict]) -> list[dict]:
            return [{"role": "user", "content": results}]

    class OverrideRuntime(AgentRuntime):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._call_count = 0
            self.execute_called = False
            self.persist_calls = 0

        def call_llm(self) -> LLMResponse:
            self._call_count += 1
            if self._call_count == 1:
                return LLMResponse(
                    text="",
                    tool_calls=[ToolCall(id="call_1", name="x", inputs={})],
                    stop_reason="tool_use",
                    assistant_message={"role": "assistant", "content": []},
                )
            return LLMResponse(
                text="done",
                tool_calls=[],
                stop_reason="end_turn",
                assistant_message={"role": "assistant", "content": "done"},
            )

        def execute_tools(self, response: LLMResponse) -> list[dict]:
            self.execute_called = True
            return [{"type": "tool_result", "tool_use_id": "call_1", "content": "ok"}]

        def persist_session(self) -> None:
            self.persist_calls += 1

    runtime = OverrideRuntime(
        provider=NoopProvider(),
        settings=AgentSettings(),
        ctx=Context(),
        tool_registry=None,
        session_store=None,
        system="test-system",
        run_ctx=RunContext(),
        session_id=None,
        provider_type="unknown",
    )

    assert runtime.run() == "done"
    assert runtime.execute_called is True
    assert runtime.persist_calls == 1


def test_runtime_custom_policy_persist_path():
    """测试自定义策略可以走 PERSIST 分支"""

    class PersistPolicy(RuntimePolicy):
        def next_step(self, turn: int, max_turns: int, response: LLMResponse | None) -> Step:
            if response is None:
                return Step.CALL_LLM
            return Step.PERSIST

    class NoopProvider(BaseLLMProvider):
        def chat(self, messages, system, tools) -> LLMResponse:
            raise AssertionError("should not call provider.chat when call_llm is overridden")

        def format_tool_result(self, tool_call_id: str, content: str) -> dict:
            return {"type": "tool_result", "tool_use_id": tool_call_id, "content": content}

        def tool_results_as_message(self, results: list[dict]) -> list[dict]:
            return [{"role": "user", "content": results}]

    class PersistRuntime(AgentRuntime):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.persist_calls = 0

        def call_llm(self) -> LLMResponse:
            return LLMResponse(
                text="persisted",
                tool_calls=[],
                stop_reason="custom",
                assistant_message={"role": "assistant", "content": "persisted"},
            )

        def persist_session(self) -> None:
            self.persist_calls += 1

    runtime = PersistRuntime(
        provider=NoopProvider(),
        settings=AgentSettings(),
        ctx=Context(),
        tool_registry=None,
        session_store=None,
        system="test-system",
        run_ctx=RunContext(),
        session_id=None,
        provider_type="unknown",
        policy=PersistPolicy(),
    )

    assert runtime.run() == "persisted"
    assert runtime.persist_calls == 1


def test_runtime_unknown_step_raises():
    """测试未知步骤会抛错而不是静默结束"""

    class UnknownPolicy(RuntimePolicy):
        def next_step(self, turn: int, max_turns: int, response: LLMResponse | None):
            return "UNKNOWN_STEP"

    class NoopProvider(BaseLLMProvider):
        def chat(self, messages, system, tools) -> LLMResponse:
            raise AssertionError("should not call provider.chat")

        def format_tool_result(self, tool_call_id: str, content: str) -> dict:
            return {"type": "tool_result", "tool_use_id": tool_call_id, "content": content}

        def tool_results_as_message(self, results: list[dict]) -> list[dict]:
            return [{"role": "user", "content": results}]

    runtime = AgentRuntime(
        provider=NoopProvider(),
        settings=AgentSettings(),
        ctx=Context(),
        tool_registry=None,
        session_store=None,
        system="test-system",
        run_ctx=RunContext(),
        session_id=None,
        provider_type="unknown",
        policy=UnknownPolicy(),
    )

    with pytest.raises(RuntimeError, match="Unhandled runtime step"):
        runtime.run()
