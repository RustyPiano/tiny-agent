from pathlib import Path

from agent_framework import _config as config
from agent_framework._config import MAX_COMPACT_HISTORY_MESSAGES, SYSTEM_PROMPT_DYNAMIC_BOUNDARY
from agent_framework.core.context import Context
from agent_framework.core.history_compactor import compact_message_history
from agent_framework.core.message_assembler import assemble_messages
from agent_framework.core.runtime import AgentRuntime
from agent_framework.llm.base import BaseLLMProvider, LLMResponse


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


def _build_runtime(messages: list[dict]) -> tuple[AgentRuntime, _CaptureProvider]:
    provider = _CaptureProvider()
    runtime = AgentRuntime(
        provider=provider,
        settings=type("S", (), {"max_turns": 20, "max_tokens": 16000})(),
        ctx=Context(initial_messages=messages),
        tool_registry=_NoopRegistry(),
        session_store=_NoopStore(),
        system="STATIC-SYSTEM",
        run_ctx=type("R", (), {"turn": 0})(),
        session_id=None,
        provider_type="anthropic",
    )
    return runtime, provider


def _build_runtime_with_provider_type(
    messages: list[dict],
    provider_type: str,
) -> tuple[AgentRuntime, _CaptureProvider]:
    provider = _CaptureProvider()
    runtime = AgentRuntime(
        provider=provider,
        settings=type("S", (), {"max_turns": 20, "max_tokens": 16000})(),
        ctx=Context(initial_messages=messages),
        tool_registry=_NoopRegistry(),
        session_store=_NoopStore(),
        system="STATIC-SYSTEM",
        run_ctx=type("R", (), {"turn": 0})(),
        session_id=None,
        provider_type=provider_type,
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
    assert "[sent as compacted provider message]" in result
    assert "task-content" in result
    assert "observation-content" in result


def test_runtime_system_prompt_contains_required_sections():
    runtime, provider = _build_runtime(messages=[{"role": "user", "content": "hello"}])

    runtime.call_llm()

    assert SYSTEM_PROMPT_DYNAMIC_BOUNDARY in provider.last_system
    assert "## Current MEMORY" in provider.last_system
    assert "## Compacted History" in provider.last_system
    assert "## Current Task + Last Observation" in provider.last_system
    assert provider.last_messages == [{"role": "user", "content": "hello"}]


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


def test_runtime_current_task_is_summarized_in_dynamic_system_but_preserved_in_messages():
    long_task = "task-" + ("x" * 400)
    runtime, provider = _build_runtime(messages=[{"role": "user", "content": long_task}])

    runtime.call_llm()

    assert provider.last_messages == [{"role": "user", "content": long_task}]
    assert long_task not in provider.last_system
    assert "Current Task:\n" in provider.last_system


def test_runtime_last_observation_is_summarized_in_dynamic_system():
    long_observation = "obs-" + ("y" * 400)
    runtime, provider = _build_runtime(
        messages=[
            {"role": "user", "content": "short task"},
            {"role": "assistant", "content": long_observation},
        ]
    )

    runtime.call_llm()

    assert long_observation not in provider.last_system
    assert "Last Observation:\n" in provider.last_system


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
    assert len(first.splitlines()) == len(messages) - MAX_COMPACT_HISTORY_MESSAGES - 1
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
        settings=type(
            "S",
            (),
            {
                "max_turns": 20,
                "max_tokens": 16000,
                "workspace_root": settings_root,
            },
        )(),
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


def test_runtime_does_not_fallback_to_global_memory_when_settings_root_has_no_memory(
    tmp_path: Path, monkeypatch
):
    settings_root = tmp_path / "settings-root"
    global_root = tmp_path / "global-root"
    settings_root.mkdir(parents=True)
    global_root.mkdir(parents=True)
    (global_root / "MEMORY.md").write_text("global memory", encoding="utf-8")
    monkeypatch.setattr(config, "WORKSPACE_ROOT", global_root)

    provider = _CaptureProvider()
    runtime = AgentRuntime(
        provider=provider,
        settings=type(
            "S",
            (),
            {
                "max_turns": 20,
                "max_tokens": 16000,
                "workspace_root": settings_root,
            },
        )(),
        ctx=Context(initial_messages=[{"role": "user", "content": "hello"}]),
        tool_registry=_NoopRegistry(),
        session_store=_NoopStore(),
        system="STATIC-SYSTEM",
        run_ctx=type("R", (), {"turn": 0})(),
        session_id=None,
        provider_type="anthropic",
    )

    runtime.call_llm()

    assert "global memory" not in provider.last_system


def test_runtime_uses_history_compactor_module_output_in_prompt(monkeypatch):
    messages = [
        {"role": "user", "content": f"m{i} " + ("x" * 200)}
        for i in range(MAX_COMPACT_HISTORY_MESSAGES + 4)
    ]
    runtime, provider = _build_runtime(messages=messages)

    monkeypatch.setattr(config, "CONTEXT_SOFT_LIMIT_TOKENS", 1)
    monkeypatch.setattr(
        "agent_framework.core.runtime.compact_message_history",
        lambda history, recent_window_size, summarize_fn, should_keep_with_next: (
            "module-compact-line",
            history[-recent_window_size:],
        ),
    )

    runtime.call_llm()

    assert "## Compacted History\n[sent as compacted provider message]" in provider.last_system
    assert provider.last_messages[0]["content"] == "Earlier conversation summary:\nmodule-compact-line"


def test_runtime_compaction_reduces_provider_messages(monkeypatch):
    messages = [{"role": "user", "content": f"m{i} " + ("x" * 200)} for i in range(12)]
    runtime, provider = _build_runtime(messages=messages)

    monkeypatch.setattr(config, "CONTEXT_SOFT_LIMIT_TOKENS", 1)

    runtime.call_llm()

    assert len(provider.last_messages) < len(messages)
    assert provider.last_messages[-1] == messages[-1]
    assert provider.last_messages[0]["role"] == "assistant"
    assert "Earlier conversation summary" in provider.last_messages[0]["content"]


def test_runtime_compaction_falls_back_when_summary_is_larger(monkeypatch):
    messages = [{"role": "user", "content": f"m{i} " + ("x" * 200)} for i in range(12)]
    runtime, provider = _build_runtime(messages=messages)

    monkeypatch.setattr(config, "CONTEXT_SOFT_LIMIT_TOKENS", 1)
    monkeypatch.setattr(
        runtime,
        "_build_compacted_payload_messages",
        lambda all_messages: (
            "summary " + ("y" * 5000),
            [{"role": "assistant", "content": "huge " + ("z" * 20000)}],
        ),
    )

    runtime.call_llm()

    assert provider.last_messages == messages


def test_compact_message_history_keeps_tool_use_with_tool_result_boundary():
    history = [{"role": "user", "content": f"old-{i}"} for i in range(6)]
    history.append(
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "call_1", "name": "run_bash", "input": {}}],
        }
    )
    history.append(
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "done"}],
        }
    )
    history.extend({"role": "assistant", "content": f"after-{i}"} for i in range(7))

    summary, recent_window = compact_message_history(
        history=history,
        recent_window_size=MAX_COMPACT_HISTORY_MESSAGES,
        summarize_fn=lambda records: "summary",
        should_keep_with_next=lambda left, right: left["role"] == "assistant"
        and right["role"] == "user"
        and "tool_use" in str(left.get("content"))
        and "tool_result" in str(right.get("content")),
    )

    assert summary == "summary"
    assert recent_window[0] == history[6]
    assert recent_window[1] == history[7]


def test_runtime_compaction_preserves_tool_use_and_tool_result_boundary(monkeypatch):
    messages = [{"role": "user", "content": f"old-{i} " + ("x" * 200)} for i in range(6)]
    tool_use_message = {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": "call_1", "name": "run_bash", "input": {}}],
    }
    tool_result_message = {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "done"}],
    }
    messages.append(tool_use_message)
    messages.append(tool_result_message)
    messages.extend(
        {"role": "assistant", "content": f"after-{i} " + ("y" * 200)} for i in range(7)
    )
    messages.append({"role": "user", "content": "current turn"})

    runtime, provider = _build_runtime(messages=messages)

    monkeypatch.setattr(config, "CONTEXT_SOFT_LIMIT_TOKENS", 1)

    runtime.call_llm()

    assert tool_result_message in provider.last_messages
    assert tool_use_message in provider.last_messages
    assert provider.last_messages.index(tool_use_message) < provider.last_messages.index(
        tool_result_message
    )


def test_runtime_compaction_preserves_openai_multi_tool_result_batch(monkeypatch):
    messages = [{"role": "user", "content": f"old-{i} " + ("x" * 200)} for i in range(2)]
    assistant_tool_call_message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "run_bash", "arguments": "{}"},
            },
            {
                "id": "call_2",
                "type": "function",
                "function": {"name": "read_file", "arguments": "{}"},
            },
        ],
    }
    tool_result_1 = {"role": "tool", "tool_call_id": "call_1", "content": "done-1"}
    tool_result_2 = {"role": "tool", "tool_call_id": "call_2", "content": "done-2"}
    messages.append(assistant_tool_call_message)
    messages.append(tool_result_1)
    messages.append(tool_result_2)
    messages.extend(
        {"role": "assistant", "content": f"after-{i} " + ("y" * 200)} for i in range(7)
    )
    messages.append({"role": "user", "content": "current turn"})

    runtime, provider = _build_runtime_with_provider_type(messages=messages, provider_type="openai")

    monkeypatch.setattr(config, "CONTEXT_SOFT_LIMIT_TOKENS", 1)

    runtime.call_llm()

    assert assistant_tool_call_message in provider.last_messages
    assert tool_result_1 in provider.last_messages
    assert tool_result_2 in provider.last_messages
    assert provider.last_messages.index(assistant_tool_call_message) < provider.last_messages.index(
        tool_result_1
    )
    assert provider.last_messages.index(tool_result_1) < provider.last_messages.index(tool_result_2)
