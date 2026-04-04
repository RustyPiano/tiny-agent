# tests/test_anthropic_network_resilience.py
"""Anthropic network resilience tests: call_llm retry logic for Anthropic provider errors."""

import time

import httpx

from config import AgentSettings
from core.context import Context
from core.logging import RunContext
from core.runtime import AgentRuntime
from llm.base import BaseLLMProvider, LLMResponse


def _make_api_connection_error(msg: str = "Connection error.") -> Exception:
    import anthropic

    req = httpx.Request("POST", "https://example.com")
    return anthropic.APIConnectionError(message=msg, request=req)


def _make_rate_limit_error(msg: str = "Rate limit exceeded.") -> Exception:
    import anthropic

    resp = httpx.Response(429, request=httpx.Request("POST", "https://example.com"))
    return anthropic.RateLimitError(msg, response=resp, body=None)


class FakeFailingThenRecoveringProvider(BaseLLMProvider):
    """Raise Anthropic network errors for N calls, then recover."""

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
    """Provider that always raises an Anthropic network error."""

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
        provider_type="anthropic",
    )


def test_call_llm_retries_on_anthropic_connection_error_and_recovers(monkeypatch):
    """call_llm retries on Anthropic APIConnectionError and returns response after recovery."""
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


def test_call_llm_returns_error_response_after_max_anthropic_retries(monkeypatch):
    """call_llm returns structured error after Anthropic retries are exhausted."""
    monkeypatch.setattr(time, "sleep", lambda s: None)

    provider = FakeAlwaysFailingProvider(error_factory=_make_api_connection_error)
    runtime = _make_runtime(provider)
    runtime.ctx.add_user("hello")

    response = runtime.call_llm()

    assert response.stop_reason == "end_turn"
    assert response.tool_calls == []
    assert "[network_error]" in response.text
    assert "3 attempts" in response.text


def test_call_llm_retries_on_anthropic_rate_limit_error(monkeypatch):
    """call_llm retries on Anthropic RateLimitError and returns response after recovery."""
    monkeypatch.setattr(time, "sleep", lambda s: None)

    provider = FakeFailingThenRecoveringProvider(fail_count=1, error_factory=_make_rate_limit_error)
    runtime = _make_runtime(provider)
    runtime.ctx.add_user("hello")

    response = runtime.call_llm()

    assert response.stop_reason == "end_turn"
    assert "recovered response" in response.text
