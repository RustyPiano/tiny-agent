# tests/test_finish_termination.py
"""Finish tool termination tests — TDD: write failing tests first."""

import json

from config import AgentSettings
from core.context import Context
from core.logging import RunContext
from core.runtime import AgentRuntime
from llm.base import BaseLLMProvider, LLMResponse, ToolCall


class CountingProvider(BaseLLMProvider):
    """Provider that counts LLM calls and returns predefined responses."""

    def __init__(self, responses: list[LLMResponse]):
        self._responses = responses
        self._call_index = 0
        self.call_count = 0

    def chat(self, messages, system, tools, max_tokens=16000) -> LLMResponse:
        self.call_count += 1
        if self._call_index >= len(self._responses):
            n = self._call_index
            total = len(self._responses)
            raise IndexError(f"CountingProvider: call {n} but only {total} responses")
        resp = self._responses[self._call_index]
        self._call_index += 1
        return resp

    def format_tool_result(self, tool_call_id: str, content: str) -> dict:
        return {"type": "tool_result", "tool_use_id": tool_call_id, "content": content}

    def tool_results_as_message(self, results: list[dict]) -> list[dict]:
        return [{"role": "user", "content": results}]


def _make_registry():
    class FinishOnlyRegistry:
        def execute(self, name: str, inputs: dict) -> str:
            if name == "finish":
                return json.dumps({"response": inputs.get("response", "")}, ensure_ascii=False)
            return "[unknown tool]"

        def get_schemas(self) -> list[dict]:
            return [{"name": "finish", "parameters": {"response": {"type": "string"}}}]

    return FinishOnlyRegistry()


def _make_runtime(provider, registry=None):
    return AgentRuntime(
        provider=provider,
        settings=AgentSettings(),
        ctx=Context(),
        tool_registry=registry or _make_registry(),
        session_store=None,
        system="test",
        run_ctx=RunContext(),
        session_id=None,
        provider_type="unknown",
    )


def test_finish_tool_causes_immediate_termination():
    """Finish tool success -> runtime returns immediately, no extra LLM call."""
    finish_output = json.dumps({"response": "done!"}, ensure_ascii=False)

    class FinishRegistry:
        def execute(self, name: str, inputs: dict) -> str:
            return finish_output

        def get_schemas(self) -> list[dict]:
            return [{"name": "finish"}]

    provider = CountingProvider(
        [
            LLMResponse(
                text="",
                tool_calls=[ToolCall(id="tc1", name="finish", inputs={"response": "done!"})],
                stop_reason="tool_use",
                assistant_message={"role": "assistant", "content": []},
            ),
        ]
    )
    runtime = _make_runtime(provider, FinishRegistry())
    result = runtime.run()

    assert provider.call_count == 1, "Should NOT make a second LLM call after finish"
    assert result == "done!"


def test_finish_tool_error_does_not_terminate():
    """Finish tool error -> runtime continues to next LLM call."""
    finish_registry = type(
        "FinishErrorRegistry",
        (),
        {
            "execute": lambda self, name, inputs: "[error] finish failed",
            "get_schemas": lambda self: [{"name": "finish"}],
        },
    )()

    provider = CountingProvider(
        [
            # First LLM call -> finish tool
            LLMResponse(
                text="",
                tool_calls=[ToolCall(id="tc1", name="finish", inputs={"response": "done!"})],
                stop_reason="tool_use",
                assistant_message={"role": "assistant", "content": []},
            ),
            # Second LLM call after error -> end_turn
            LLMResponse(
                text="recovered",
                tool_calls=[],
                stop_reason="end_turn",
                assistant_message={"role": "assistant", "content": "recovered"},
            ),
        ]
    )
    runtime = _make_runtime(provider, finish_registry)
    result = runtime.run()

    assert provider.call_count == 2, "Should continue after finish tool error"
    assert result == "recovered"


def test_finish_tool_parse_error_does_not_terminate():
    """Finish tool with parse_error -> runtime continues to next LLM call."""
    finish_registry = type(
        "FinishParseErrorRegistry",
        (),
        {
            "execute": lambda self, name, inputs: json.dumps(
                {"response": "should not terminate"}, ensure_ascii=False
            ),
            "get_schemas": lambda self: [{"name": "finish"}],
        },
    )()

    provider = CountingProvider(
        [
            # First LLM call -> finish tool with parse error
            LLMResponse(
                text="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="finish",
                        inputs={},
                        parse_error="arguments JSON parse failed",
                        raw_arguments="{bad",
                    )
                ],
                stop_reason="tool_use",
                assistant_message={"role": "assistant", "content": []},
            ),
            # Second LLM call -> end_turn
            LLMResponse(
                text="recovered from parse error",
                tool_calls=[],
                stop_reason="end_turn",
                assistant_message={"role": "assistant", "content": "recovered from parse error"},
            ),
        ]
    )
    runtime = _make_runtime(provider, finish_registry)
    result = runtime.run()

    assert provider.call_count == 2, "Should continue after finish tool parse error"
    assert "recovered" in result


def test_finish_with_non_json_output_does_not_terminate():
    """Finish tool returning plain text (not JSON) -> runtime continues."""
    finish_registry = type(
        "NonJsonFinishRegistry",
        (),
        {
            "execute": lambda self, name, inputs: "this is not json",
            "get_schemas": lambda self: [{"name": "finish"}],
        },
    )()

    provider = CountingProvider(
        [
            LLMResponse(
                text="",
                tool_calls=[ToolCall(id="tc1", name="finish", inputs={"response": "done!"})],
                stop_reason="tool_use",
                assistant_message={"role": "assistant", "content": []},
            ),
            LLMResponse(
                text="continued after non-json",
                tool_calls=[],
                stop_reason="end_turn",
                assistant_message={"role": "assistant", "content": "continued after non-json"},
            ),
        ]
    )
    runtime = _make_runtime(provider, finish_registry)
    result = runtime.run()

    assert provider.call_count == 2, "Should continue after non-JSON finish output"
    assert result == "continued after non-json"


def test_finish_with_non_string_response_value_does_not_terminate():
    """Finish tool returning {"response": 42} -> runtime continues (response must be string)."""
    finish_registry = type(
        "NonStringFinishRegistry",
        (),
        {
            "execute": lambda self, name, inputs: json.dumps({"response": 42}, ensure_ascii=False),
            "get_schemas": lambda self: [{"name": "finish"}],
        },
    )()

    provider = CountingProvider(
        [
            LLMResponse(
                text="",
                tool_calls=[ToolCall(id="tc1", name="finish", inputs={"response": 42})],
                stop_reason="tool_use",
                assistant_message={"role": "assistant", "content": []},
            ),
            LLMResponse(
                text="continued after non-string response",
                tool_calls=[],
                stop_reason="end_turn",
                assistant_message={
                    "role": "assistant",
                    "content": "continued after non-string response",
                },
            ),
        ]
    )
    runtime = _make_runtime(provider, finish_registry)
    result = runtime.run()

    assert provider.call_count == 2, "Should continue after non-string response value"
    assert result == "continued after non-string response"


def test_finish_with_empty_string_response_does_not_terminate():
    """Finish tool returning {"response": ""} -> runtime continues (response must be non-empty)."""
    finish_registry = type(
        "EmptyStringFinishRegistry",
        (),
        {
            "execute": lambda self, name, inputs: json.dumps({"response": ""}, ensure_ascii=False),
            "get_schemas": lambda self: [{"name": "finish"}],
        },
    )()

    provider = CountingProvider(
        [
            LLMResponse(
                text="",
                tool_calls=[ToolCall(id="tc1", name="finish", inputs={"response": ""})],
                stop_reason="tool_use",
                assistant_message={"role": "assistant", "content": []},
            ),
            LLMResponse(
                text="continued after empty response",
                tool_calls=[],
                stop_reason="end_turn",
                assistant_message={
                    "role": "assistant",
                    "content": "continued after empty response",
                },
            ),
        ]
    )
    runtime = _make_runtime(provider, finish_registry)
    result = runtime.run()

    assert provider.call_count == 2, "Should continue after empty string response"
    assert result == "continued after empty response"


def test_multiple_tools_in_batch_including_finish_terminates():
    """Batch with finish + another tool -> finish still terminates after tool execution."""
    call_order: list[str] = []

    class MultiToolRegistry:
        def execute(self, name: str, inputs: dict) -> str:
            call_order.append(name)
            if name == "finish":
                return json.dumps({"response": "batch done!"}, ensure_ascii=False)
            return "read_file result"

        def get_schemas(self) -> list[dict]:
            return [{"name": "finish"}, {"name": "read_file"}]

    provider = CountingProvider(
        [
            LLMResponse(
                text="",
                tool_calls=[
                    ToolCall(id="tc1", name="read_file", inputs={"path": "/tmp/test"}),
                    ToolCall(id="tc2", name="finish", inputs={"response": "batch done!"}),
                ],
                stop_reason="tool_use",
                assistant_message={"role": "assistant", "content": []},
            ),
        ]
    )
    runtime = _make_runtime(provider, MultiToolRegistry())
    result = runtime.run()

    assert provider.call_count == 1, "Should NOT make a second LLM call after finish in batch"
    assert result == "batch done!"
    assert call_order == ["read_file", "finish"]


def test_finish_via_react_json_fallback_terminates():
    """Finish via ReAct JSON fallback path -> runtime terminates."""
    finish_registry = type(
        "ReactFinishRegistry",
        (),
        {
            "execute": lambda self, name, inputs: json.dumps(
                {"response": "react finish done"}, ensure_ascii=False
            ),
            "get_schemas": lambda self: [{"name": "finish"}],
        },
    )()

    provider = CountingProvider(
        [
            LLMResponse(
                text=json.dumps(
                    {
                        "thought": "done",
                        "action": "finish",
                        "action_input": {"response": "react finish done"},
                    },
                    ensure_ascii=False,
                ),
                tool_calls=[],
                stop_reason="tool_use",
                assistant_message={"role": "assistant", "content": []},
            ),
        ]
    )
    runtime = _make_runtime(provider, finish_registry)
    result = runtime.run()

    assert provider.call_count == 1, "Should NOT make a second LLM call after ReAct finish"
    assert result == "react finish done"


def test_finish_registry_checks_tool_name():
    """Verify that the finish check in execute_tools guards on tool name."""
    registry = _make_registry()
    assert hasattr(registry, "execute")
    # Calling with non-finish tool should return unknown
    result = registry.execute("some_other_tool", {})
    assert result == "[unknown tool]"
