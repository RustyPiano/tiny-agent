# tests/test_agent.py
"""Agent 主循环测试，使用 mock provider"""

from pathlib import Path

import pytest

from agent_framework._config import AgentSettings
from agent_framework.core.agent import _load_history_with_provider_check, run
from agent_framework.core.context import Context
from agent_framework.core.logging import RunContext
from agent_framework.core.policies import RuntimePolicy, Step
from agent_framework.core.prompt_builder import build_system_prompt
from agent_framework.core.runtime import AgentRuntime
from agent_framework.llm.base import BaseLLMProvider, LLMResponse, ToolCall


class MockProvider(BaseLLMProvider):
    """模拟 LLM provider，返回预定义响应"""

    def __init__(self, responses: list[LLMResponse]):
        self._responses = responses
        self._call_index = 0
        self.messages_received: list[list[dict]] = []

    def chat(self, messages, system, tools, max_tokens=16000) -> LLMResponse:
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

    def chat(self, messages, system, tools, max_tokens=16000) -> LLMResponse:
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
    result = run("执行 echo hello", provider=provider)
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
            for i in range(10)  # 超过 max_turns=5
        ]
    )
    result = run("无限循环测试", provider=provider, settings=AgentSettings(max_turns=5))
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
        def chat(self, messages, system, tools, max_tokens=16000) -> LLMResponse:
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

    class PersistPolicy(RuntimePolicy):
        def next_step(self, turn: int, max_turns: int, response: LLMResponse | None) -> Step:
            if response is None:
                return Step.CALL_LLM
            return Step.PERSIST

    class NoopProvider(BaseLLMProvider):
        def chat(self, messages, system, tools, max_tokens=16000) -> LLMResponse:
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
        def chat(self, messages, system, tools, max_tokens=16000) -> LLMResponse:
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


def test_runtime_forwards_tool_parse_error_metadata():
    class NoopProvider(BaseLLMProvider):
        def chat(self, messages, system, tools, max_tokens=16000) -> LLMResponse:
            raise AssertionError("should not call provider.chat")

        def format_tool_result(self, tool_call_id: str, content: str) -> dict:
            return {"role": "tool", "tool_call_id": tool_call_id, "content": content}

        def tool_results_as_message(self, results: list[dict]) -> list[dict]:
            return results

    class RecordingRegistry:
        def __init__(self):
            self.calls: list[tuple[str, dict]] = []

        def execute(self, name: str, inputs: dict) -> str:
            self.calls.append((name, dict(inputs)))
            return "[ok]"

        def get_schemas(self) -> list[dict]:
            return [{"name": "write_file"}]

    class NoopStore:
        def save(self, session_id, messages, provider_type):
            _ = (session_id, messages, provider_type)

    registry = RecordingRegistry()

    runtime = AgentRuntime(
        provider=NoopProvider(),
        settings=AgentSettings(),
        ctx=Context(),
        tool_registry=registry,
        session_store=NoopStore(),
        system="test-system",
        run_ctx=RunContext(),
        session_id=None,
        provider_type="openai",
    )

    response = LLMResponse(
        text="",
        tool_calls=[
            ToolCall(
                id="call_1",
                name="write_file",
                inputs={},
                parse_error="arguments JSON 解析失败: Unterminated string",
                raw_arguments='{"path": "a", "content": "x"',
            )
        ],
        stop_reason="tool_use",
        assistant_message={"role": "assistant", "content": ""},
    )

    runtime.execute_tools(response)

    assert len(registry.calls) == 1
    called_name, called_inputs = registry.calls[0]
    assert called_name == "write_file"
    assert called_inputs["_tool_parse_error"].startswith("arguments JSON 解析失败")
    assert called_inputs["_tool_raw_arguments"].startswith('{"path"')


def test_runtime_handles_non_dict_tool_inputs():
    class NoopProvider(BaseLLMProvider):
        def chat(self, messages, system, tools, max_tokens=16000) -> LLMResponse:
            raise AssertionError("should not call provider.chat")

        def format_tool_result(self, tool_call_id: str, content: str) -> dict:
            return {"role": "tool", "tool_call_id": tool_call_id, "content": content}

        def tool_results_as_message(self, results: list[dict]) -> list[dict]:
            return results

    class RecordingRegistry:
        def __init__(self):
            self.calls: list[tuple[str, dict]] = []

        def execute(self, name: str, inputs: dict) -> str:
            self.calls.append((name, dict(inputs)))
            return "[ok]"

        def get_schemas(self) -> list[dict]:
            return [{"name": "write_file"}]

    class NoopStore:
        def save(self, session_id, messages, provider_type):
            _ = (session_id, messages, provider_type)

    registry = RecordingRegistry()

    runtime = AgentRuntime(
        provider=NoopProvider(),
        settings=AgentSettings(),
        ctx=Context(),
        tool_registry=registry,
        session_store=NoopStore(),
        system="test-system",
        run_ctx=RunContext(),
        session_id=None,
        provider_type="openai",
    )

    response = LLMResponse(
        text="",
        tool_calls=[
            ToolCall(
                id="call_2",
                name="write_file",
                inputs=[("path", "a.txt")],
            )
        ],
        stop_reason="tool_use",
        assistant_message={"role": "assistant", "content": ""},
    )

    runtime.execute_tools(response)

    assert len(registry.calls) == 1
    called_name, called_inputs = registry.calls[0]
    assert called_name == "write_file"
    assert called_inputs["_tool_parse_error"].startswith("tool inputs 必须是 object")
    assert called_inputs["_tool_raw_arguments"] == "[('path', 'a.txt')]"


def test_system_prompt_contains_available_skills():
    from agent_framework.skills import discover_skills

    discover_skills(
        project_dir=Path("/nonexistent/project"),
        global_dir=Path("/nonexistent/global"),
    )
    prompt = build_system_prompt()
    assert "## Available Skills" not in prompt


def test_load_history_treats_empty_stored_provider_as_unknown(monkeypatch):
    def fake_load(session_id: str):
        _ = session_id
        return [{"role": "user", "content": "legacy"}], ""

    monkeypatch.setattr("agent_framework.core.agent.session_store.load", fake_load)

    history = _load_history_with_provider_check(
        session_id="legacy-session",
        provider_type="openai",
        run_ctx=RunContext(),
    )

    assert history == [{"role": "user", "content": "legacy"}]


def test_load_history_treats_missing_stored_provider_as_unknown(monkeypatch):
    def fake_load(session_id: str):
        _ = session_id
        return [{"role": "user", "content": "legacy"}], None

    monkeypatch.setattr("agent_framework.core.agent.session_store.load", fake_load)

    history = _load_history_with_provider_check(
        session_id="legacy-session",
        provider_type="anthropic",
        run_ctx=RunContext(),
    )

    assert history == [{"role": "user", "content": "legacy"}]


def test_run_passes_subagent_flow_flag_to_prompt_builder(monkeypatch):
    provider = MockProvider(
        [
            LLMResponse(
                text="ok",
                tool_calls=[],
                stop_reason="end_turn",
                assistant_message={"role": "assistant", "content": "ok"},
            )
        ]
    )
    captured: dict = {}

    def fake_build_system_prompt(
        skill_names=None,
        enable_subagent_flow=False,
        tool_schemas=None,
    ):
        captured["skill_names"] = skill_names
        captured["enable_subagent_flow"] = enable_subagent_flow
        captured["tool_schemas"] = tool_schemas
        return "test-system"

    monkeypatch.setattr("agent_framework.core.agent.build_system_prompt", fake_build_system_prompt)

    settings = AgentSettings(enable_subagent_flow=True)
    result = run("hello", provider=provider, settings=settings)

    assert result == "ok"
    assert captured["enable_subagent_flow"] is True


def test_run_activates_runtime_subagent_flow_with_default_seed(monkeypatch):
    provider = MockProvider(
        [
            LLMResponse(
                text="ok",
                tool_calls=[],
                stop_reason="end_turn",
                assistant_message={"role": "assistant", "content": "ok"},
            )
        ]
    )

    class RecordingRuntime:
        last_instance = None

        def __init__(self, **kwargs):
            _ = kwargs
            self.seed_tasks = None
            RecordingRuntime.last_instance = self

        def enable_subagent_flow(self, tasks: list[str]) -> None:
            self.seed_tasks = list(tasks)

        def run(self) -> str:
            return "ok"

    monkeypatch.setattr("agent_framework.core.agent.AgentRuntime", RecordingRuntime)
    settings = AgentSettings(enable_subagent_flow=True)

    result = run("hello", provider=provider, settings=settings)

    assert result == "ok"
    assert RecordingRuntime.last_instance is not None
    assert RecordingRuntime.last_instance.seed_tasks == ["task_001"]


def test_run_path_handle_flow_result_not_disabled_when_flag_enabled(monkeypatch):
    provider = MockProvider(
        [
            LLMResponse(
                text="unused",
                tool_calls=[],
                stop_reason="end_turn",
                assistant_message={"role": "assistant", "content": "unused"},
            )
        ]
    )

    class FlowCheckRuntime(AgentRuntime):
        def run(self) -> str:
            result = self.handle_flow_result(
                {
                    "task_id": "task_001",
                    "phase": "implement",
                    "status": "NEEDS_CONTEXT",
                    "details": "flow ok",
                }
            )
            return result.get("message", "")

    monkeypatch.setattr("agent_framework.core.agent.AgentRuntime", FlowCheckRuntime)
    settings = AgentSettings(enable_subagent_flow=True)

    result = run("hello", provider=provider, settings=settings)

    assert result == "flow ok"
