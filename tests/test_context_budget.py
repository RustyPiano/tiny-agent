from __future__ import annotations

import config
from config import MAX_FILE_READ_LINES
from core.context import Context
from core.context_budget import estimate_tokens, should_compact
from core.runtime import AgentRuntime
from llm.base import BaseLLMProvider, LLMResponse
from tools.file_tools import read_file, write_file


def test_read_file_default_caps_to_max_lines(tmp_path):
    path = str(tmp_path / "many_lines.txt")
    content = "\n".join(f"line-{i}" for i in range(MAX_FILE_READ_LINES + 50))
    write_file(path, content)

    result = read_file(path)
    lines = result.splitlines()

    assert len(lines) <= MAX_FILE_READ_LINES
    assert lines[0] == "line-0"


def test_estimate_tokens_returns_positive_for_non_empty_text():
    assert estimate_tokens("hello world") > 0


def test_should_compact_behavior():
    assert should_compact(estimated_tokens=160000, soft_limit=160000) is True
    assert should_compact(estimated_tokens=159999, soft_limit=160000) is False


class _CaptureProvider(BaseLLMProvider):
    def __init__(self):
        self.last_system: str = ""

    def chat(self, messages, system, tools, max_tokens=16000) -> LLMResponse:
        _ = (messages, tools)
        self.last_system = system
        return LLMResponse(
            text="ok",
            tool_calls=[],
            stop_reason="end_turn",
            assistant_message={"role": "assistant", "content": "ok"},
        )

    def format_tool_result(self, tool_call_id: str, content: str) -> dict:
        return {"type": "tool_result", "tool_use_id": tool_call_id, "content": content}

    def tool_results_as_message(self, results: list[dict]) -> list[dict]:
        return [{"role": "user", "content": results}]


class _NoopRegistry:
    def get_schemas(self) -> list[dict]:
        return []


class _NoopStore:
    def save(self, session_id, messages, provider_type):
        _ = (session_id, messages, provider_type)


def test_runtime_only_compacts_near_soft_limit(monkeypatch):
    provider = _CaptureProvider()
    runtime = AgentRuntime(
        provider=provider,
        settings=type("S", (), {"max_turns": 20, "max_tokens": 16000})(),
        ctx=Context(initial_messages=[{"role": "user", "content": "short task"}]),
        tool_registry=_NoopRegistry(),
        session_store=_NoopStore(),
        system="STATIC",
        run_ctx=type("R", (), {"turn": 0})(),
        session_id=None,
        provider_type="anthropic",
    )

    monkeypatch.setattr(config, "CONTEXT_SOFT_LIMIT_TOKENS", 10**9)
    runtime.call_llm()
    assert "## Compacted History\n[none]" in provider.last_system

    monkeypatch.setattr(config, "CONTEXT_SOFT_LIMIT_TOKENS", 1)
    runtime.call_llm()
    assert "user: short task" in provider.last_system


def test_runtime_budget_counts_structured_tool_result_payload(monkeypatch):
    provider = _CaptureProvider()
    large_payload = "x" * 5000
    runtime = AgentRuntime(
        provider=provider,
        settings=type("S", (), {"max_turns": 20, "max_tokens": 16000})(),
        ctx=Context(
            initial_messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_1",
                            "content": {
                                "stdout": large_payload,
                                "meta": {"exit_code": 0},
                            },
                        }
                    ],
                }
            ]
        ),
        tool_registry=_NoopRegistry(),
        session_store=_NoopStore(),
        system="STATIC",
        run_ctx=type("R", (), {"turn": 0})(),
        session_id=None,
        provider_type="anthropic",
    )

    monkeypatch.setattr(config, "CONTEXT_SOFT_LIMIT_TOKENS", 200)
    runtime.call_llm()

    assert "user: [content blocks: tool_result:call_1]" in provider.last_system
