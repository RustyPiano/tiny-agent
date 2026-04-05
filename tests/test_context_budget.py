from __future__ import annotations

from agent_framework import _config as config
from agent_framework._config import MAX_FILE_READ_LINES
from agent_framework.core.context import Context
from agent_framework.core.context_budget import estimate_tokens, serialize_for_budget, should_compact
from agent_framework.core.runtime import AgentRuntime
from agent_framework.llm.base import BaseLLMProvider, LLMResponse
from agent_framework.tools.file_tools import read_file, write_file


def test_read_file_default_caps_to_max_lines(tmp_path):
    path = str(tmp_path / "many_lines.txt")
    content = "\n".join(f"line-{i}" for i in range(MAX_FILE_READ_LINES + 50))
    write_file(path, content)

    result = read_file(path)

    assert "[truncated]" in result
    assert "line-0" in result


def test_read_file_truncation_includes_total_line_count(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MAX_FILE_READ_LINES", 5)
    path = str(tmp_path / "big.txt")
    content = "\n".join(f"line-{i}" for i in range(20))
    write_file(path, content)

    result = read_file(path)

    assert "[truncated]" in result
    assert "共 20 行" in result
    assert "start_line/end_line" in result
    # 截断前的内容完整
    assert "line-0" in result
    assert "line-4" in result


def test_estimate_tokens_returns_positive_for_non_empty_text():
    assert estimate_tokens("hello world") > 0


def test_should_compact_behavior():
    assert should_compact(estimated_tokens=160000, soft_limit=160000) is True
    assert should_compact(estimated_tokens=159999, soft_limit=160000) is False


class _CaptureProvider(BaseLLMProvider):
    def __init__(self):
        self.last_system: str = ""
        self.last_messages: list[dict] = []

    def chat(self, messages, system, tools, max_tokens=16000) -> LLMResponse:
        _ = (messages, tools)
        self.last_system = system
        self.last_messages = list(messages)
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


class _RichRegistry(_NoopRegistry):
    def get_schemas(self) -> list[dict]:
        return [
            {
                "name": "run_bash",
                "description": "tool " + ("d" * 4000),
                "input_schema": {"type": "object"},
            }
        ]


def test_runtime_only_compacts_near_soft_limit(monkeypatch):
    provider = _CaptureProvider()
    messages = [{"role": "user", "content": f"message-{i} " + ("x" * 200)} for i in range(12)]
    runtime = AgentRuntime(
        provider=provider,
        settings=type("S", (), {"max_turns": 20, "max_tokens": 16000})(),
        ctx=Context(initial_messages=messages),
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
    assert provider.last_messages == messages

    monkeypatch.setattr(config, "CONTEXT_SOFT_LIMIT_TOKENS", 1)
    runtime.call_llm()
    assert "## Compacted History\n[sent as compacted provider message]" in provider.last_system
    assert len(provider.last_messages) < len(messages)


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
    baseline_runtime = AgentRuntime(
        provider=_CaptureProvider(),
        settings=type("S", (), {"max_turns": 20, "max_tokens": 16000})(),
        ctx=Context(initial_messages=[{"role": "user", "content": "tiny"}]),
        tool_registry=_NoopRegistry(),
        session_store=_NoopStore(),
        system="STATIC",
        run_ctx=type("R", (), {"turn": 0})(),
        session_id=None,
        provider_type="anthropic",
    )

    monkeypatch.setattr(runtime, "_load_memory_text", lambda: "")
    monkeypatch.setattr(baseline_runtime, "_load_memory_text", lambda: "")

    assert runtime._estimate_context_tokens(runtime.ctx.get()) > baseline_runtime._estimate_context_tokens(
        baseline_runtime.ctx.get()
    )


def test_runtime_estimate_context_tokens_includes_dynamic_system_and_tools(monkeypatch):
    base_provider = _CaptureProvider()
    rich_provider = _CaptureProvider()

    base_runtime = AgentRuntime(
        provider=base_provider,
        settings=type("S", (), {"max_turns": 20, "max_tokens": 16000})(),
        ctx=Context(initial_messages=[{"role": "user", "content": "short task"}]),
        tool_registry=_NoopRegistry(),
        session_store=_NoopStore(),
        system="STATIC",
        run_ctx=type("R", (), {"turn": 0})(),
        session_id=None,
        provider_type="anthropic",
    )
    rich_runtime = AgentRuntime(
        provider=rich_provider,
        settings=type("S", (), {"max_turns": 20, "max_tokens": 16000})(),
        ctx=Context(initial_messages=[{"role": "user", "content": "short task"}]),
        tool_registry=_RichRegistry(),
        session_store=_NoopStore(),
        system="STATIC",
        run_ctx=type("R", (), {"turn": 0})(),
        session_id=None,
        provider_type="anthropic",
    )

    monkeypatch.setattr(base_runtime, "_load_memory_text", lambda: "")
    monkeypatch.setattr(rich_runtime, "_load_memory_text", lambda: "memory " + ("m" * 4000))

    base_tokens = base_runtime._estimate_context_tokens(base_runtime.ctx.get())
    rich_tokens = rich_runtime._estimate_context_tokens(rich_runtime.ctx.get())

    assert rich_tokens > base_tokens


def test_runtime_estimate_context_tokens_matches_openai_payload_shape(monkeypatch):
    runtime = AgentRuntime(
        provider=_CaptureProvider(),
        settings=type("S", (), {"max_turns": 20, "max_tokens": 16000})(),
        ctx=Context(
            initial_messages=[
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "run_bash",
                                "arguments": '{"command":"' + ("x" * 4000) + '"}',
                            },
                        }
                    ],
                }
            ]
        ),
        tool_registry=_RichRegistry(),
        session_store=_NoopStore(),
        system="STATIC",
        run_ctx=type("R", (), {"turn": 0})(),
        session_id=None,
        provider_type="openai",
    )

    monkeypatch.setattr(runtime, "_load_memory_text", lambda: "memory " + ("m" * 4000))

    estimated = runtime._estimate_context_tokens(runtime.ctx.get())
    messages, system_text, tools, _ = runtime._build_provider_payload(runtime.ctx.get())

    openai_messages = [{"role": "system", "content": system_text}, *messages]
    openai_tools = [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {}),
            },
        }
        for tool in tools
    ]
    expected = estimate_tokens(
        "\n".join(
            [
                *[serialize_for_budget(message) for message in openai_messages],
                *[serialize_for_budget(tool) for tool in openai_tools],
            ]
        )
    )

    assert estimated == expected
