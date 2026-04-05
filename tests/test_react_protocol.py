import pytest

from agent_framework._config import AgentSettings
from agent_framework.core.context import Context
from agent_framework.core.logging import RunContext
from agent_framework.core.react_protocol import ReactDecision, parse_react_json
from agent_framework.core.runtime import AgentRuntime
from agent_framework.llm.base import BaseLLMProvider, LLMResponse, ToolCall


class NoopProvider(BaseLLMProvider):
    def chat(self, messages, system, tools, max_tokens=16000) -> LLMResponse:
        raise AssertionError("should not call provider.chat")

    def format_tool_result(self, tool_call_id: str, content: str) -> dict:
        return {"role": "tool", "tool_call_id": tool_call_id, "content": content}

    def tool_results_as_message(self, results: list[dict]) -> list[dict]:
        return results


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


def make_runtime(registry: RecordingRegistry) -> AgentRuntime:
    return AgentRuntime(
        provider=NoopProvider(),
        settings=AgentSettings(),
        ctx=Context(),
        tool_registry=registry,
        session_store=NoopStore(),
        system="test-system",
        run_ctx=RunContext(),
        session_id=None,
        provider_type="unknown",
    )


def test_parse_react_json_valid_payload() -> None:
    raw = (
        '{"thought":"need shell output",'
        '"action":"run_bash",'
        '"action_input":{"command":"echo hello"}}'
    )

    result = parse_react_json(raw, allowed_actions={"run_bash", "NONE"})

    assert result == ReactDecision(
        thought="need shell output",
        action="run_bash",
        action_input={"command": "echo hello"},
    )


def test_parse_react_json_rejects_non_json_text() -> None:
    with pytest.raises(ValueError, match="must be valid JSON"):
        parse_react_json(
            "I will now run a command",
            allowed_actions={"run_bash", "NONE"},
        )


def test_parse_react_json_rejects_missing_required_fields() -> None:
    raw = '{"thought":"ok","action":"run_bash"}'

    with pytest.raises(ValueError, match="keys"):
        parse_react_json(raw, allowed_actions={"run_bash", "NONE"})


def test_parse_react_json_rejects_invalid_action() -> None:
    raw = '{"thought":"ok","action":"unknown_tool","action_input":{"x":1}}'

    with pytest.raises(ValueError, match="action is not allowed"):
        parse_react_json(raw, allowed_actions={"run_bash", "NONE"})


def test_parse_react_json_rejects_invalid_action_input_type() -> None:
    raw = '{"thought":"ok","action":"run_bash","action_input":123}'

    with pytest.raises(ValueError, match="action_input"):
        parse_react_json(raw, allowed_actions={"run_bash", "NONE"})


def test_parse_react_json_accepts_extra_keys() -> None:
    raw = '{"thought":"ok","action":"run_bash","action_input":{"command":"echo hi"},"extra":"x"}'

    result = parse_react_json(raw, allowed_actions={"run_bash", "NONE"})
    assert result.thought == "ok"
    assert result.action == "run_bash"
    assert result.action_input == {"command": "echo hi"}


def test_runtime_execute_tools_uses_react_json_parser_path_without_list_tools() -> None:
    registry = RecordingRegistry()
    runtime = make_runtime(registry)

    response = LLMResponse(
        text=('{"thought":"need shell","action":"run_bash","action_input":{"command":"echo hi"}}'),
        tool_calls=[],
        stop_reason="tool_use",
        assistant_message={"role": "assistant", "content": ""},
    )

    runtime.execute_tools(response)

    assert registry.calls == [("run_bash", {"command": "echo hi"})]


def test_runtime_execute_tools_react_none_does_not_create_tool_call() -> None:
    registry = RecordingRegistry()
    runtime = make_runtime(registry)

    response = LLMResponse(
        text='{"thought":"done","action":"NONE","action_input":"NONE"}',
        tool_calls=[],
        stop_reason="end_turn",
        assistant_message={"role": "assistant", "content": ""},
    )

    results = runtime.execute_tools(response)

    assert registry.calls == []
    assert results == []


def test_runtime_execute_tools_ignores_react_json_when_tool_calls_present() -> None:
    registry = RecordingRegistry()
    runtime = make_runtime(registry)

    response = LLMResponse(
        text=(
            '{"thought":"try override","action":"run_bash",'
            '"action_input":{"command":"echo from-json"}}'
        ),
        tool_calls=[
            ToolCall(
                id="call_1",
                name="run_bash",
                inputs={"command": "echo from-tool-call"},
            )
        ],
        stop_reason="tool_use",
        assistant_message={"role": "assistant", "content": ""},
    )

    runtime.execute_tools(response)

    assert registry.calls == [("run_bash", {"command": "echo from-tool-call"})]
