from config import AgentSettings
from core.context import Context
from core.logging import RunContext
from core.runtime import AgentRuntime
from llm.base import BaseLLMProvider, LLMResponse


class SequenceProvider(BaseLLMProvider):
    def __init__(self, responses: list[LLMResponse]):
        self._responses = responses
        self._call_index = 0

    def chat(self, messages, system, tools) -> LLMResponse:
        _ = (messages, system, tools)
        if self._call_index >= len(self._responses):
            raise IndexError("SequenceProvider: 响应已用完")
        response = self._responses[self._call_index]
        self._call_index += 1
        return response

    def format_tool_result(self, tool_call_id: str, content: str) -> dict:
        return {"type": "tool_result", "tool_use_id": tool_call_id, "content": content}

    def tool_results_as_message(self, results: list[dict]) -> list[dict]:
        return [{"role": "user", "content": results}]


class RecordingRegistry:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def execute(self, name: str, inputs: dict) -> str:
        self.calls.append((name, dict(inputs)))
        return "ok"

    def get_schemas(self) -> list[dict]:
        return [{"name": "run_bash"}]


class NoopStore:
    def save(self, session_id, messages, provider_type):
        _ = (session_id, messages, provider_type)


def make_runtime(provider: BaseLLMProvider, registry: RecordingRegistry) -> AgentRuntime:
    ctx = Context()
    ctx.add_user("执行任务")
    return AgentRuntime(
        provider=provider,
        settings=AgentSettings(),
        ctx=ctx,
        tool_registry=registry,
        session_store=NoopStore(),
        system="test-system",
        run_ctx=RunContext(),
        session_id=None,
        provider_type="unknown",
    )


def test_self_healing_retries_on_repeated_none_action_then_returns_deterministic_payload() -> None:
    none_action = LLMResponse(
        text='{"thought":"no-op","action":"NONE","action_input":"NONE"}',
        tool_calls=[],
        stop_reason="tool_use",
        assistant_message={
            "role": "assistant",
            "content": '{"thought":"no-op","action":"NONE","action_input":"NONE"}',
        },
    )
    registry = RecordingRegistry()
    provider = SequenceProvider([none_action, none_action, none_action, none_action])
    runtime = make_runtime(provider, registry)

    result = runtime.run()

    assert result == '{"thought":"format_retry_exhausted","action":"NONE","action_input":"NONE"}'
    assert provider._call_index == 4
    assert registry.calls == []


def test_self_healing_recovers_after_none_action_then_executes_valid_tool() -> None:
    none_action = LLMResponse(
        text='{"thought":"not yet","action":"NONE","action_input":"NONE"}',
        tool_calls=[],
        stop_reason="tool_use",
        assistant_message={
            "role": "assistant",
            "content": '{"thought":"not yet","action":"NONE","action_input":"NONE"}',
        },
    )
    valid_action = LLMResponse(
        text='{"thought":"need shell","action":"run_bash","action_input":{"command":"echo hi"}}',
        tool_calls=[],
        stop_reason="tool_use",
        assistant_message={
            "role": "assistant",
            "content": '{"thought":"need shell","action":"run_bash","action_input":{"command":"echo hi"}}',
        },
    )
    final = LLMResponse(
        text="done",
        tool_calls=[],
        stop_reason="end_turn",
        assistant_message={"role": "assistant", "content": "done"},
    )

    registry = RecordingRegistry()
    provider = SequenceProvider([none_action, valid_action, final])
    runtime = make_runtime(provider, registry)

    result = runtime.run()

    assert result == "done"
    assert provider._call_index == 3
    assert registry.calls == [("run_bash", {"command": "echo hi"})]
