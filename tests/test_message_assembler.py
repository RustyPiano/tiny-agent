from pathlib import Path

import config
from config import MAX_COMPACT_HISTORY_MESSAGES, SYSTEM_PROMPT_DYNAMIC_BOUNDARY
from core.context import Context
from core.message_assembler import assemble_messages
from core.runtime import AgentRuntime
from llm.base import BaseLLMProvider, LLMResponse


class _CaptureProvider(BaseLLMProvider):
    def __init__(self):
        self.last_system: str = ""

    def chat(self, messages, system, tools) -> LLMResponse:
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


def _build_runtime(messages: list[dict]) -> tuple[AgentRuntime, _CaptureProvider]:
    provider = _CaptureProvider()
    runtime = AgentRuntime(
        provider=provider,
        settings=type("S", (), {"max_turns": 20})(),
        ctx=Context(initial_messages=messages),
        tool_registry=_NoopRegistry(),
        session_store=_NoopStore(),
        system="STATIC-SYSTEM",
        run_ctx=type("R", (), {"turn": 0})(),
        session_id=None,
        provider_type="anthropic",
    )
    return runtime, provider


def test_assemble_messages_contains_dynamic_boundary():
    result = assemble_messages(
        static_system_prompt="BASE",
        memory_text="M",
        compacted_history="H",
        last_observation="O",
        current_task="T",
    )

    assert SYSTEM_PROMPT_DYNAMIC_BOUNDARY in result


def test_assemble_messages_contains_required_sections():
    result = assemble_messages(
        static_system_prompt="STATIC",
        memory_text="memory-content",
        compacted_history="history-content",
        last_observation="observation-content",
        current_task="task-content",
    )

    assert "Current MEMORY" in result
    assert "Compacted History" in result
    assert "Current Task + Last Observation" in result
    assert "memory-content" in result
    assert "history-content" in result
    assert "task-content" in result
    assert "observation-content" in result


def test_runtime_system_prompt_contains_required_sections():
    runtime, provider = _build_runtime(messages=[{"role": "user", "content": "hello"}])

    runtime.call_llm()

    assert SYSTEM_PROMPT_DYNAMIC_BOUNDARY in provider.last_system
    assert "## Current MEMORY" in provider.last_system
    assert "## Compacted History" in provider.last_system
    assert "## Current Task + Last Observation" in provider.last_system


def test_runtime_current_task_skips_anthropic_tool_result_user_payload():
    runtime, provider = _build_runtime(
        messages=[
            {"role": "user", "content": "original task"},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "call_1", "name": "run_bash", "input": {}}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "done"}],
            },
        ]
    )

    runtime.call_llm()

    assert "Current Task:\noriginal task" in provider.last_system


def test_runtime_current_task_is_none_when_only_synthetic_tool_result_user_payload_exists():
    runtime, provider = _build_runtime(
        messages=[
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "call_1", "name": "run_bash", "input": {}}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "done"}],
            },
        ]
    )

    runtime.call_llm()

    assert "Current Task:\n[none]" in provider.last_system


def test_runtime_prompt_uses_fallback_markers_for_empty_fields():
    runtime, provider = _build_runtime(messages=[])

    runtime.call_llm()

    assert "## Current MEMORY\n[none]" in provider.last_system
    assert "## Compacted History\n[none]" in provider.last_system
    assert "Current Task:\n[none]" in provider.last_system
    assert "Last Observation:\n[none]" in provider.last_system


def test_runtime_compact_history_is_deterministic_and_summarized():
    messages = [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "name": "read_file", "id": "a", "input": {}}],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": "ok"}]},
        {"role": "assistant", "content": {"z": 1, "a": 2}},
    ]
    for i in range(MAX_COMPACT_HISTORY_MESSAGES + 3):
        messages.append({"role": "user", "content": f"m{i}"})

    runtime, _provider = _build_runtime(messages=messages)

    first = runtime._compact_history(messages)
    second = runtime._compact_history(messages)

    assert first == second
    assert len(first.splitlines()) == MAX_COMPACT_HISTORY_MESSAGES
    assert "{'type':" not in first


def test_runtime_loads_memory_md_from_workspace_root(tmp_path: Path, monkeypatch):
    memory_path = tmp_path / "MEMORY.md"
    memory_path.write_text("project memory line", encoding="utf-8")
    monkeypatch.setattr(config, "WORKSPACE_ROOT", tmp_path)

    runtime, provider = _build_runtime(messages=[{"role": "user", "content": "hello"}])

    runtime.call_llm()

    assert "## Current MEMORY\nproject memory line" in provider.last_system


def test_runtime_prefers_settings_workspace_memory_before_global(tmp_path: Path, monkeypatch):
    settings_root = tmp_path / "settings-root"
    global_root = tmp_path / "global-root"
    settings_root.mkdir(parents=True)
    global_root.mkdir(parents=True)
    (settings_root / "MEMORY.md").write_text("settings memory", encoding="utf-8")
    (global_root / "MEMORY.md").write_text("global memory", encoding="utf-8")
    monkeypatch.setattr(config, "WORKSPACE_ROOT", global_root)

    provider = _CaptureProvider()
    runtime = AgentRuntime(
        provider=provider,
        settings=type("S", (), {"max_turns": 20, "workspace_root": settings_root})(),
        ctx=Context(initial_messages=[{"role": "user", "content": "hello"}]),
        tool_registry=_NoopRegistry(),
        session_store=_NoopStore(),
        system="STATIC-SYSTEM",
        run_ctx=type("R", (), {"turn": 0})(),
        session_id=None,
        provider_type="anthropic",
    )

    runtime.call_llm()

    assert "## Current MEMORY\nsettings memory" in provider.last_system


def test_runtime_uses_history_compactor_module_output_in_prompt(monkeypatch):
    runtime, provider = _build_runtime(messages=[{"role": "user", "content": "hello"}])

    monkeypatch.setattr(config, "CONTEXT_SOFT_LIMIT_TOKENS", 1)
    monkeypatch.setattr(
        "core.runtime.compact_history",
        lambda history, max_records, summarize_fn: ["module-compact-line"],
    )

    runtime.call_llm()

    assert "## Compacted History\nmodule-compact-line" in provider.last_system
