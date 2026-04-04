# tests/test_network_resilience.py
"""Network resilience tests: LLM retry logic and REPL exception handling."""

import io
import time
from unittest.mock import MagicMock, patch

import httpx

from agent_framework._config import AgentSettings
from agent_framework.core.context import Context
from agent_framework.core.logging import RunContext
from agent_framework.core.runtime import AgentRuntime
from agent_framework.llm.base import BaseLLMProvider, LLMResponse


def _make_api_connection_error(msg: str = "Connection error.") -> Exception:
    import openai

    req = httpx.Request("POST", "https://example.com")
    return openai.APIConnectionError(message=msg, request=req)


def _make_rate_limit_error(msg: str = "Rate limit exceeded.") -> Exception:
    import openai

    resp = httpx.Response(429, request=httpx.Request("POST", "https://example.com"))
    return openai.RateLimitError(msg, response=resp, body=None)


class FakeFailingThenRecoveringProvider(BaseLLMProvider):
    """Provider that raises network error on first N calls, then returns a valid response."""

    def __init__(self, fail_count: int, error_factory):
        self.fail_count = fail_count
        self.error_factory = error_factory
        self._call_index = 0

    def chat(self, messages, system, tools, max_tokens=16000) -> LLMResponse:
        if self._call_index < self.fail_count:
            self._call_index += 1
            raise self.error_factory()
        self._call_index += 1
        return LLMResponse(
            text="recovered response",
            tool_calls=[],
            stop_reason="end_turn",
            assistant_message={"role": "assistant", "content": "recovered response"},
        )

    def format_tool_result(self, tool_call_id: str, content: str) -> dict:
        return {"type": "tool_result", "tool_use_id": tool_call_id, "content": content}

    def tool_results_as_message(self, results: list[dict]) -> list[dict]:
        return [{"role": "user", "content": results}]


class FakeAlwaysFailingProvider(BaseLLMProvider):
    """Provider that always raises a network error."""

    def __init__(self, error_factory):
        self.error_factory = error_factory

    def chat(self, messages, system, tools, max_tokens=16000) -> LLMResponse:
        raise self.error_factory()

    def format_tool_result(self, tool_call_id: str, content: str) -> dict:
        return {"type": "tool_result", "tool_use_id": tool_call_id, "content": content}

    def tool_results_as_message(self, results: list[dict]) -> list[dict]:
        return [{"role": "user", "content": results}]


class _DummyToolRegistry:
    def get_schemas(self) -> list[dict]:
        return []


def _make_runtime(provider: BaseLLMProvider) -> AgentRuntime:
    return AgentRuntime(
        provider=provider,
        settings=AgentSettings(),
        ctx=Context(),
        tool_registry=_DummyToolRegistry(),
        session_store=None,
        system="test-system",
        run_ctx=RunContext(),
        session_id=None,
        provider_type="openai",
    )


# --- Task 1: call_llm network error retry ---


def test_call_llm_retries_on_connection_error_and_recovers(monkeypatch):
    """call_llm retries on APIConnectionError and returns response after recovery."""
    monkeypatch.setattr(time, "sleep", lambda s: None)

    provider = FakeFailingThenRecoveringProvider(
        fail_count=1, error_factory=_make_api_connection_error
    )
    runtime = _make_runtime(provider)
    runtime.ctx.add_user("hello")

    response = runtime.call_llm()

    assert response.stop_reason == "end_turn"
    assert "recovered response" in response.text
    assert response.tool_calls == []


def test_call_llm_returns_error_response_after_max_retries(monkeypatch):
    """call_llm returns structured error LLMResponse after exhausting retries."""
    monkeypatch.setattr(time, "sleep", lambda s: None)

    provider = FakeAlwaysFailingProvider(error_factory=_make_api_connection_error)
    runtime = _make_runtime(provider)
    runtime.ctx.add_user("hello")

    response = runtime.call_llm()

    assert response.stop_reason == "end_turn"
    assert response.tool_calls == []
    assert "[network_error]" in response.text
    assert "3 attempts" in response.text


def test_call_llm_retries_on_rate_limit_error(monkeypatch):
    """call_llm retries on RateLimitError and returns response after recovery."""
    monkeypatch.setattr(time, "sleep", lambda s: None)

    provider = FakeFailingThenRecoveringProvider(fail_count=1, error_factory=_make_rate_limit_error)
    runtime = _make_runtime(provider)
    runtime.ctx.add_user("hello")

    response = runtime.call_llm()

    assert response.stop_reason == "end_turn"
    assert "recovered response" in response.text


# --- Task 2: REPL exception handling ---


def test_main_repl_continues_after_run_exception(monkeypatch):
    """REPL loop continues after run() raises an exception, does not crash."""
    monkeypatch.setattr(time, "sleep", lambda s: None)

    call_count = 0

    def fake_run(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise _make_api_connection_error("boom")
        return "second result"

    inputs = iter(["first task", "second task"])

    def fake_input(prompt=""):
        try:
            return next(inputs)
        except StopIteration:
            raise KeyboardInterrupt from None

    captured = io.StringIO()

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr("sys.stdout", captured)
    monkeypatch.setattr("sys.argv", ["main.py"])

    with patch("agent_framework.main.run", side_effect=fake_run):
        with patch("agent_framework.main.create_provider"):
            with patch("agent_framework.main.bootstrap"):
                with patch("agent_framework.main.setup_logging"):
                    with patch("agent_framework.main.AgentSettings") as mock_settings:
                        mock_instance = MagicMock()
                        mock_instance.validate.return_value = []
                        mock_instance.provider_type = "openai"
                        mock_instance.model = "gpt-4"
                        mock_instance.to_provider_config.return_value = MagicMock()
                        mock_settings.from_env.return_value = mock_instance
                        mock_settings.return_value = mock_instance

                        from agent_framework.main import main as main

                        try:
                            main()
                        except SystemExit:
                            pass

    output = captured.getvalue()
    assert "second result" in output
    assert call_count == 2


def test_call_llm_sleep_called_exactly_twice_on_total_failure(monkeypatch):
    """time.sleep is called exactly 2 times (between 3 attempts) on total failure."""
    mock_sleep = MagicMock()
    monkeypatch.setattr(time, "sleep", mock_sleep)

    provider = FakeAlwaysFailingProvider(error_factory=_make_api_connection_error)
    runtime = _make_runtime(provider)
    runtime.ctx.add_user("hello")

    response = runtime.call_llm()

    assert response.stop_reason == "end_turn"
    assert "[network_error]" in response.text
    assert mock_sleep.call_count == 2
    mock_sleep.assert_any_call(1)
    mock_sleep.assert_any_call(2)
