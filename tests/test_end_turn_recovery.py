from agent_framework._config import AgentSettings
from agent_framework.core.context import Context
from agent_framework.core.logging import RunContext
from agent_framework.core.runtime import AgentRuntime
from agent_framework.llm.base import BaseLLMProvider, LLMResponse


class SequenceProvider(BaseLLMProvider):
    def __init__(self, responses: list[LLMResponse]):
        self._responses = responses
        self._call_index = 0

    def chat(self, messages, system, tools, max_tokens=16000) -> LLMResponse:
        _ = (messages, system, tools)
        if self._call_index >= len(self._responses):
            raise IndexError("SequenceProvider: responses exhausted")
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
    ctx.add_user("run task")
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


def test_end_turn_recovery_retry_exhaustion() -> None:
    malformed = LLMResponse(
        text='{"thought":"oops","action":"run_bash","action_input":123}',
        tool_calls=[],
        stop_reason="end_turn",
        assistant_message={"role": "assistant", "content": "bad react json"},
    )
    registry = RecordingRegistry()
    provider = SequenceProvider([malformed, malformed, malformed])
    runtime = make_runtime(provider, registry)

    result = runtime.run()

    assert result == (
        '{"thought":"end_turn_parse_retry_exhausted","action":"NONE","action_input":"NONE"}'
    )
    assert provider._call_index == 3
    assert registry.calls == []
    messages = runtime.ctx.get()
    observations = [
        msg.get("content", "")
        for msg in messages
        if msg.get("role") == "user"
        and isinstance(msg.get("content"), str)
        and "ReAct output parse failed at end_turn" in msg.get("content", "")
    ]
    assert len(observations) == 2


def test_end_turn_recovery_executes_tool() -> None:
    malformed = LLMResponse(
        text='{"thought":"oops","action":"run_bash","action_input":123}',
        tool_calls=[],
        stop_reason="end_turn",
        assistant_message={"role": "assistant", "content": "bad react json"},
    )
    valid_action = LLMResponse(
        text='{"thought":"need shell","action":"run_bash","action_input":{"command":"echo hi"}}',
        tool_calls=[],
        stop_reason="end_turn",
        assistant_message={"role": "assistant", "content": "run tool"},
    )
    final = LLMResponse(
        text="done",
        tool_calls=[],
        stop_reason="end_turn",
        assistant_message={"role": "assistant", "content": "done"},
    )

    registry = RecordingRegistry()
    provider = SequenceProvider([malformed, valid_action, final])
    runtime = make_runtime(provider, registry)

    result = runtime.run()

    assert result == "done"
    assert provider._call_index == 3
    assert registry.calls == [("run_bash", {"command": "echo hi"})]
