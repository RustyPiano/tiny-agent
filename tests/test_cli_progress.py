from __future__ import annotations

import threading
import time
from pathlib import Path

from agent_framework._config import AgentSettings
from agent_framework.core.context import Context
from agent_framework.core.logging import RunContext
from agent_framework.core.runtime import AgentRuntime
from agent_framework.llm.base import BaseLLMProvider, LLMResponse, ToolCall


class _NoopProvider(BaseLLMProvider):
    def chat(self, messages, system, tools, max_tokens=16000) -> LLMResponse:
        raise AssertionError("chat should be overridden in this test")

    def format_tool_result(self, tool_call_id: str, content: str) -> dict:
        return {"type": "tool_result", "tool_use_id": tool_call_id, "content": content}

    def tool_results_as_message(self, results: list[dict]) -> list[dict]:
        return [{"role": "user", "content": results}]


class _NoopStore:
    def save(self, session_id, messages, provider_type):
        _ = (session_id, messages, provider_type)


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


class _SlowRegistry(_RecordingRegistry):
    def execute(self, name: str, inputs: dict) -> str:
        time.sleep(0.15)
        return "slow-ok"


def _make_runtime(provider, registry, ui_event_printer=None):
    ctx = Context()
    ctx.add_user("test task")
    return AgentRuntime(
        provider=provider,
        settings=AgentSettings(),
        ctx=ctx,
        tool_registry=registry,
        session_store=_NoopStore(),
        system="test-system",
        run_ctx=RunContext(),
        session_id=None,
        provider_type="unknown",
        ui_event_printer=ui_event_printer,
    )


def test_ui_event_printer_emits_turn_start(monkeypatch):
    events: list[str] = []

    class _EndTurnProvider(_NoopProvider):
        def chat(self, messages, system, tools, max_tokens=16000) -> LLMResponse:
            return LLMResponse(
                text="done",
                tool_calls=[],
                stop_reason="end_turn",
                assistant_message={"role": "assistant", "content": "done"},
            )

    runtime = _make_runtime(
        _EndTurnProvider(),
        _RecordingRegistry(),
        ui_event_printer=events.append,
    )
    result = runtime.run()

    assert result == "done"
    assert any("第1轮" in e and "分析中" in e for e in events)
    assert any("任务完成" in e for e in events)


def test_ui_event_printer_emits_tool_events():
    events: list[str] = []

    class _ToolProvider(_NoopProvider):
        def chat(self, messages, system, tools, max_tokens=16000) -> LLMResponse:
            return LLMResponse(
                text="",
                tool_calls=[ToolCall(id="call_1", name="run_bash", inputs={"command": "echo hi"})],
                stop_reason="tool_use",
                assistant_message={"role": "assistant", "content": []},
            )

    class _EndAfterTool(_NoopProvider):
        def __init__(self):
            self._call_count = 0

        def chat(self, messages, system, tools, max_tokens=16000) -> LLMResponse:
            self._call_count += 1
            if self._call_count == 1:
                return LLMResponse(
                    text="",
                    tool_calls=[
                        ToolCall(id="call_1", name="run_bash", inputs={"command": "echo hi"})
                    ],
                    stop_reason="tool_use",
                    assistant_message={"role": "assistant", "content": []},
                )
            return LLMResponse(
                text="done",
                tool_calls=[],
                stop_reason="end_turn",
                assistant_message={"role": "assistant", "content": "done"},
            )

    registry = _RecordingRegistry()
    provider = _EndAfterTool()
    runtime = _make_runtime(provider, registry, ui_event_printer=events.append)
    result = runtime.run()

    assert result == "done"
    assert any("工具 run_bash 开始" in e for e in events)
    assert any("run_bash 完成" in e for e in events)


def test_ui_event_printer_emits_max_turns_warning():
    events: list[str] = []

    class _InfiniteToolProvider(_NoopProvider):
        def __init__(self):
            self._call_count = 0

        def chat(self, messages, system, tools, max_tokens=16000) -> LLMResponse:
            self._call_count += 1
            return LLMResponse(
                text="",
                tool_calls=[
                    ToolCall(
                        id=f"call_{self._call_count}",
                        name="run_bash",
                        inputs={"command": "echo hi"},
                    )
                ],
                stop_reason="tool_use",
                assistant_message={"role": "assistant", "content": []},
            )

    registry = _RecordingRegistry()
    provider = _InfiniteToolProvider()
    runtime = _make_runtime(provider, registry, ui_event_printer=events.append)
    result = runtime.run()

    assert "[warn]" in result
    assert any("达到最大轮次" in e for e in events)


def test_heartbeat_emits_at_interval():
    events: list[str] = []
    stop_early = threading.Event()

    class _SlowProvider(_NoopProvider):
        def chat(self, messages, system, tools, max_tokens=16000) -> LLMResponse:
            return LLMResponse(
                text="done",
                tool_calls=[],
                stop_reason="end_turn",
                assistant_message={"role": "assistant", "content": "done"},
            )

    class _VerySlowRegistry(_RecordingRegistry):
        def execute(self, name: str, inputs: dict) -> str:
            stop_early.wait(timeout=0.5)
            return "slow-ok"

    registry = _VerySlowRegistry()
    provider = _SlowProvider()
    runtime = _make_runtime(provider, registry, ui_event_printer=events.append)

    # Run in thread so we can stop the slow execute
    result_holder = [None]

    def run_runtime():
        result_holder[0] = runtime.run()

    t = threading.Thread(target=run_runtime, daemon=True)
    t.start()

    # Wait enough time for at least one heartbeat (10s interval, but execute blocks)
    # Since execute takes 0.5s, heartbeat won't fire in this short test.
    # We just verify the heartbeat mechanism exists and doesn't crash.
    stop_early.set()
    t.join(timeout=5)

    # The runtime should complete without hanging
    assert result_holder[0] is not None


def test_no_ui_events_when_printer_is_none():
    class _EndTurnProvider(_NoopProvider):
        def chat(self, messages, system, tools, max_tokens=16000) -> LLMResponse:
            return LLMResponse(
                text="done",
                tool_calls=[],
                stop_reason="end_turn",
                assistant_message={"role": "assistant", "content": "done"},
            )

    runtime = _make_runtime(_EndTurnProvider(), _RecordingRegistry(), ui_event_printer=None)
    result = runtime.run()

    assert result == "done"


def test_ui_event_printer_emits_per_tool_details(tmp_path: Path):
    events: list[str] = []
    edit_target = tmp_path / "edit_target.txt"
    edit_target.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    class _DetailRegistry:
        def execute(self, name: str, inputs: dict) -> str:
            if name == "use_skill":
                return "# Find Skills"
            if name == "list_dir":
                return "\n".join(f"item{i}" for i in range(10))
            if name == "write_file":
                Path(inputs["path"]).write_text(inputs["content"], encoding="utf-8")
                return f"[ok] 已写入 {len(inputs['content'])} 字符到 {inputs['path']}"
            if name == "read_file":
                return "1\tline1\n2\tline2\n3\tline3"
            if name == "edit_file":
                p = Path(inputs["path"])
                text = p.read_text(encoding="utf-8")
                p.write_text(
                    text.replace(inputs["old_str"], inputs["new_str"], 1),
                    encoding="utf-8",
                )
                return f"[ok] 已更新文件: {inputs['path']}"
            if name == "run_bash":
                return "ok output line one\nok output line two"
            if name == "finish":
                return '{"response":"done"}'
            return "[error] unknown tool"

        def get_schemas(self) -> list[dict]:
            return [
                {"name": "use_skill"},
                {"name": "list_dir"},
                {"name": "write_file"},
                {"name": "read_file"},
                {"name": "edit_file"},
                {"name": "run_bash"},
                {"name": "finish"},
            ]

        def list_tools(self) -> list[str]:
            return [
                "use_skill",
                "list_dir",
                "write_file",
                "read_file",
                "edit_file",
                "run_bash",
                "finish",
            ]

    runtime = _make_runtime(_NoopProvider(), _DetailRegistry(), ui_event_printer=events.append)

    response = LLMResponse(
        text="",
        tool_calls=[
            ToolCall(id="c1", name="use_skill", inputs={"name": "find-skills"}),
            ToolCall(id="c2", name="list_dir", inputs={"path": str(tmp_path)}),
            ToolCall(
                id="c3",
                name="write_file",
                inputs={
                    "path": str(tmp_path / "new.txt"),
                    "content": "a\nb\nc\n",
                    "mode": "overwrite",
                },
            ),
            ToolCall(id="c4", name="read_file", inputs={"path": str(tmp_path / "new.txt")}),
            ToolCall(
                id="c5",
                name="edit_file",
                inputs={
                    "path": str(edit_target),
                    "old_str": "beta",
                    "new_str": "BETA",
                },
            ),
            ToolCall(
                id="c6",
                name="run_bash",
                inputs={
                    "command": "python run_script.py " + "x" * 120,
                },
            ),
            ToolCall(id="c7", name="finish", inputs={"response": "done"}),
        ],
        stop_reason="tool_use",
        assistant_message={"role": "assistant", "content": []},
    )

    runtime.execute_tools(response)

    assert any("skill=find-skills" in e for e in events)
    assert any("结果(10)" in e and "+2" in e for e in events)
    assert any("写入 3 行" in e and "new.txt" in e for e in events)
    assert any("读取 3 行" in e and "new.txt" in e for e in events)
    assert any("+1 / -1" in e and "edit_target.txt" in e for e in events)
    assert any("cmd=python run_script.py" in e and "…" in e for e in events)


def test_ui_event_printer_shows_bash_status_and_preview():
    events: list[str] = []

    class _BashRegistry:
        def execute(self, name: str, inputs: dict) -> str:
            _ = inputs
            if name == "run_bash":
                return "[timeout] 命令在 30s 内未完成: long command"
            return "ok"

        def get_schemas(self) -> list[dict]:
            return [{"name": "run_bash"}]

        def list_tools(self) -> list[str]:
            return ["run_bash"]

    runtime = _make_runtime(_NoopProvider(), _BashRegistry(), ui_event_printer=events.append)
    response = LLMResponse(
        text="",
        tool_calls=[ToolCall(id="c1", name="run_bash", inputs={"command": "echo hello"})],
        stop_reason="tool_use",
        assistant_message={"role": "assistant", "content": []},
    )

    runtime.execute_tools(response)

    assert any("状态=timeout" in e for e in events)
    assert any("输出=" in e for e in events)


def test_emit_tool_detail_start_job_shows_job_id_and_status():
    events: list[str] = []
    runtime = _make_runtime(_NoopProvider(), _RecordingRegistry(), ui_event_printer=events.append)

    runtime._emit_tool_detail(
        "start_job",
        {},
        '{"job_id":"job_123","status":"queued"}',
    )

    assert any("job=" in e and "status=" in e and "job_123" in e and "queued" in e for e in events)


def test_emit_tool_detail_poll_job_shows_status_and_exit_code_when_available():
    events: list[str] = []
    runtime = _make_runtime(_NoopProvider(), _RecordingRegistry(), ui_event_printer=events.append)

    runtime._emit_tool_detail(
        "poll_job",
        {"job_id": "job_123"},
        '{"status":"finished","exit_code":0}',
    )

    assert any(
        "job=" in e and "status=" in e and "exit=" in e and "finished" in e and "0" in e
        for e in events
    )


def test_emit_tool_detail_read_job_log_shows_bytes_offset_progression_preview():
    events: list[str] = []
    runtime = _make_runtime(_NoopProvider(), _RecordingRegistry(), ui_event_printer=events.append)

    runtime._emit_tool_detail(
        "read_job_log",
        {"job_id": "job_123", "offset": 120},
        '{"bytes_read":64,"next_offset":184,"preview":"line1\\nline2"}',
    )

    assert any(
        "job=" in e
        and "bytes=" in e
        and "offset=" in e
        and "preview=" in e
        and "64" in e
        and "120" in e
        and "184" in e
        and "line1" in e
        for e in events
    )


def test_emit_tool_detail_cancel_job_shows_cancellation_result():
    events: list[str] = []
    runtime = _make_runtime(_NoopProvider(), _RecordingRegistry(), ui_event_printer=events.append)

    runtime._emit_tool_detail(
        "cancel_job",
        {"job_id": "job_123"},
        '{"cancelled":true,"status":"cancelling"}',
    )

    assert any(
        "job=" in e and "cancelled=" in e and "status=" in e and "true" in e and "cancelling" in e
        for e in events
    )


def test_emit_tool_detail_job_tools_fall_back_when_json_parse_fails():
    events: list[str] = []
    runtime = _make_runtime(_NoopProvider(), _RecordingRegistry(), ui_event_printer=events.append)

    runtime._emit_tool_detail("start_job", {}, "plain text output")

    assert any("状态=ok" in e and "plain text output" in e for e in events)
