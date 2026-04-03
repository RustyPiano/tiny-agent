from __future__ import annotations

import builtins

from config import AgentSettings
from core.context import Context
from core.logging import RunContext
from core.runtime import AgentRuntime
from llm.base import BaseLLMProvider, LLMResponse


class _NoopProvider(BaseLLMProvider):
    def chat(self, messages, system, tools, max_tokens=16000) -> LLMResponse:
        raise AssertionError("chat should be overridden in this test")

    def format_tool_result(self, tool_call_id: str, content: str) -> dict:
        return {"type": "tool_result", "tool_use_id": tool_call_id, "content": content}

    def tool_results_as_message(self, results: list[dict]) -> list[dict]:
        return [{"role": "user", "content": results}]


class _NoopRegistry:
    def get_schemas(self) -> list[dict]:
        return []

    def list_tools(self) -> list[str]:
        return []


class _NoopStore:
    def save(self, session_id, messages, provider_type):
        _ = (session_id, messages, provider_type)


class _SequenceProvider(BaseLLMProvider):
    def __init__(self, responses: list[LLMResponse]):
        self._responses = responses
        self._index = 0

    def chat(self, messages, system, tools, max_tokens=16000) -> LLMResponse:
        _ = (messages, system, tools)
        if self._index >= len(self._responses):
            raise IndexError("SequenceProvider: responses exhausted")
        response = self._responses[self._index]
        self._index += 1
        return response

    def format_tool_result(self, tool_call_id: str, content: str) -> dict:
        return {"type": "tool_result", "tool_use_id": tool_call_id, "content": content}

    def tool_results_as_message(self, results: list[dict]) -> list[dict]:
        return [{"role": "user", "content": results}]


class _RecordingRegistry:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def execute(self, name: str, inputs: dict) -> str:
        self.calls.append((name, dict(inputs)))
        return "ok"

    def get_schemas(self) -> list[dict]:
        return [{"name": "run_bash"}]

    def list_tools(self) -> list[str]:
        return ["run_bash"]


def test_runtime_emits_assistant_decision_event(monkeypatch):
    events: list[tuple[str, dict]] = []

    def fake_log_event(event: str, ctx: RunContext, **kwargs):
        _ = ctx
        events.append((event, kwargs))

    monkeypatch.setattr("core.runtime.log_event", fake_log_event)

    class _Runtime(AgentRuntime):
        def call_llm(self) -> LLMResponse:
            return LLMResponse(
                text="ok",
                tool_calls=[],
                stop_reason="end_turn",
                assistant_message={"role": "assistant", "content": "ok"},
            )

    runtime = _Runtime(
        provider=_NoopProvider(),
        settings=AgentSettings(),
        ctx=Context(),
        tool_registry=_NoopRegistry(),
        session_store=_NoopStore(),
        system="test-system",
        run_ctx=RunContext(),
        session_id=None,
        provider_type="unknown",
    )

    assert runtime.run() == "ok"
    assert ("assistant_decision", {"stop_reason": "end_turn", "tool_calls_count": 0}) in events


def test_main_show_turns_flag_passes_output_pathway(monkeypatch, capsys):
    import main

    class DummySettings:
        provider_type = "openai"
        model = "dummy-model"
        base_url = None

        def validate(self):
            return []

        def to_provider_config(self):
            return {}

    dummy_settings = DummySettings()
    monkeypatch.setattr(main.AgentSettings, "from_env", staticmethod(lambda: dummy_settings))
    monkeypatch.setattr(main, "bootstrap", lambda settings: None)
    monkeypatch.setattr(main, "create_provider", lambda cfg: object())

    captured_kwargs: dict = {}

    def fake_run(user_input: str, **kwargs):
        _ = user_input
        captured_kwargs.update(kwargs)
        kwargs["turn_printer"]("[t2] stop=end_turn tools=0")
        return "done"

    monkeypatch.setattr(main, "run", fake_run)

    inputs = iter(["hello"])

    def fake_input(prompt: str) -> str:
        _ = prompt
        try:
            return next(inputs)
        except StopIteration:
            raise KeyboardInterrupt from None

    monkeypatch.setattr(builtins, "input", fake_input)
    monkeypatch.setattr("sys.argv", ["main.py", "--show-turns"])

    main.main()
    out = capsys.readouterr().out

    assert captured_kwargs["show_turns"] is True
    assert callable(captured_kwargs["turn_printer"])
    assert "[t2] stop=end_turn tools=0" in out


def test_runtime_turn_summary_counts_react_fallback_tool_and_does_not_leak_suffixes():
    turn_lines: list[str] = []
    provider = _SequenceProvider(
        [
            LLMResponse(
                text=(
                    '{"thought":"need shell","action":"run_bash",'
                    '"action_input":{"command":"echo hi"}}'
                ),
                tool_calls=[],
                stop_reason="end_turn",
                assistant_message={"role": "assistant", "content": "run tool"},
            ),
            LLMResponse(
                text="done",
                tool_calls=[],
                stop_reason="end_turn",
                assistant_message={"role": "assistant", "content": "done"},
            ),
        ]
    )
    registry = _RecordingRegistry()
    ctx = Context()
    ctx.add_user("run task")

    runtime = AgentRuntime(
        provider=provider,
        settings=AgentSettings(),
        ctx=ctx,
        tool_registry=registry,
        session_store=_NoopStore(),
        system="test-system",
        run_ctx=RunContext(),
        session_id=None,
        provider_type="unknown",
        show_turns=True,
        turn_printer=turn_lines.append,
    )

    assert runtime.run() == "done"
    assert registry.calls == [("run_bash", {"command": "echo hi"})]
    assert turn_lines[0].startswith("[t1] stop=end_turn tools=1")
    assert "run_bash:ok" in turn_lines[0]
    assert turn_lines[1] == "[t2] stop=end_turn tools=0"
