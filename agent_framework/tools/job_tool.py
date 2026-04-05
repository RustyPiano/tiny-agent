import json
import os
import pathlib
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass

from agent_framework import _config as config
from agent_framework.tools.bash_tool import _is_blocked, _is_detached_command
from agent_framework.tools.registry import register


@dataclass
class _JobRecord:
    job_id: str
    process: subprocess.Popen
    started_at: float
    log_path: str
    cancelled: bool = False
    cancel_signal: str | None = None
    finished_at: float | None = None
    last_output_at: float | None = None
    last_observed_size: int = 0
    tail_cache: str = ""


_MAX_JOB_RECORDS = 256
_TERMINAL_JOB_TTL_SEC = 3600.0
_MAX_RUNNING_JOBS = 16
_MAX_OUTPUT_TAIL_BYTES = 2000
_JOB_QUIET_THRESHOLD_SEC = 5.0
_JOB_STALLED_THRESHOLD_SEC = 30.0


_JOBS: dict[str, _JobRecord] = {}
_LOCK = threading.RLock()


def _ok(payload: dict) -> str:
    data = {"ok": True}
    data.update(payload)
    return json.dumps(data, ensure_ascii=False)


def _err(error: str, message: str) -> str:
    return json.dumps({"ok": False, "error": error, "message": message}, ensure_ascii=False)


def _resolve_workspace_root(workspace_root: pathlib.Path | str | None = None) -> pathlib.Path:
    root_value = config.WORKSPACE_ROOT if workspace_root is None else workspace_root
    return pathlib.Path(root_value).resolve()


def _resolve_workdir(
    workdir: str | None, workspace_root: pathlib.Path | str | None = None
) -> tuple[str | None, str | None]:
    root = _resolve_workspace_root(workspace_root)
    if not root.exists() or not root.is_dir():
        return None, f"workspace root must be an existing directory: {root}"

    try:
        resolved = root if workdir is None else pathlib.Path(workdir).resolve()
    except Exception as e:
        return None, f"invalid workdir: {e}"

    try:
        resolved.relative_to(root)
    except ValueError:
        return None, f"workdir is outside workspace root: {resolved}; workspace_root={root}"

    if not resolved.exists() or not resolved.is_dir():
        return None, f"workdir must be an existing directory: {resolved}"

    return str(resolved), None


def _cleanup_jobs_locked() -> None:
    now = time.time()
    stale_ids: list[str] = []
    for job_id, record in _JOBS.items():
        if record.finished_at is None:
            continue
        if (now - record.finished_at) > _TERMINAL_JOB_TTL_SEC:
            stale_ids.append(job_id)

    for job_id in stale_ids:
        record = _JOBS.pop(job_id)
        try:
            os.remove(record.log_path)
        except Exception:
            pass

    if len(_JOBS) <= _MAX_JOB_RECORDS:
        return

    terminal = sorted(
        (
            (job_id, r)
            for job_id, r in _JOBS.items()
            if r.finished_at is not None or r.process.poll() is not None
        ),
        key=lambda pair: pair[1].finished_at or pair[1].started_at,
    )
    while len(_JOBS) > _MAX_JOB_RECORDS and terminal:
        drop_id, record = terminal.pop(0)
        _JOBS.pop(drop_id, None)
        try:
            os.remove(record.log_path)
        except Exception:
            pass


def _running_jobs_count_locked() -> int:
    return sum(1 for record in _JOBS.values() if record.process.poll() is None)


def _tail_log(log_path: str, max_bytes: int = _MAX_OUTPUT_TAIL_BYTES) -> tuple[str, int, float | None]:
    try:
        size = os.path.getsize(log_path)
    except OSError:
        return "", 0, None

    try:
        with open(log_path, "rb") as fp:
            if size > max_bytes:
                fp.seek(size - max_bytes)
            chunk = fp.read(max_bytes)
        mtime = os.path.getmtime(log_path) if size > 0 else None
    except OSError:
        return "", size, None

    return chunk.decode("utf-8", errors="replace"), size, mtime


def _final_status(record: _JobRecord) -> str:
    exit_code = record.process.poll()
    if exit_code is None:
        return "running"
    if record.cancelled:
        return "cancelled"
    if exit_code == 0:
        return "succeeded"
    return "failed"


def _activity_for(record: _JobRecord, now: float, file_size: int) -> str:
    if record.process.poll() is not None:
        return "quiet"

    if file_size > record.last_observed_size:
        return "active"

    reference_ts = record.last_output_at or record.started_at
    silence = max(0.0, now - reference_ts)
    if silence >= _JOB_STALLED_THRESHOLD_SEC:
        return "stalled"
    if silence >= _JOB_QUIET_THRESHOLD_SEC:
        return "quiet"
    return "active"


def _recommended_poll_after(activity: str, terminal: bool) -> int | None:
    if terminal:
        return None
    if activity == "active":
        return 2
    if activity == "quiet":
        return 5
    return 10


def _refresh_record_locked(record: _JobRecord) -> dict:
    now = time.time()
    output_tail, file_size, last_output_at = _tail_log(record.log_path)
    if last_output_at is not None:
        record.last_output_at = last_output_at
    activity = _activity_for(record, now, file_size)
    record.last_observed_size = file_size
    record.tail_cache = output_tail

    exit_code = record.process.poll()
    terminal = exit_code is not None
    if terminal and record.finished_at is None:
        record.finished_at = now

    status = _final_status(record)
    return {
        "job_id": record.job_id,
        "status": status,
        "activity": activity,
        "duration_sec": max(0.0, now - record.started_at),
        "last_output_at": record.last_output_at,
        "output_tail": record.tail_cache,
        "recommended_poll_after_s": _recommended_poll_after(activity, terminal),
        "terminal": terminal,
        **({"exit_code": exit_code} if terminal else {}),
    }


def _start_job(
    command: str,
    workdir: str | None = None,
    *,
    workspace_root: pathlib.Path | str | None = None,
) -> str:
    if not isinstance(command, str) or not command.strip():
        return _err("invalid_argument", "command must be a non-empty string")
    if workdir is not None and not isinstance(workdir, str):
        return _err("invalid_argument", "workdir must be a string or null")

    block_reason = _is_blocked(command)
    if block_reason:
        return _err("invalid_argument", f"blocked command: {block_reason}")
    if _is_detached_command(command):
        return _err("invalid_argument", "background/detached operator '&' is not allowed in run_job")

    resolved_workdir, err = _resolve_workdir(workdir, workspace_root=workspace_root)
    if err is not None:
        return _err("invalid_argument", err)

    with _LOCK:
        _cleanup_jobs_locked()
        if _running_jobs_count_locked() >= _MAX_RUNNING_JOBS:
            return _err("too_many_running_jobs", f"running jobs exceed limit: {_MAX_RUNNING_JOBS}")

        job_id = f"job_{uuid.uuid4().hex[:12]}"
        started_at = time.time()
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                delete=False,
                prefix="agent_job_",
                suffix=".log",
            ) as fp:
                log_path = fp.name

            with open(log_path, "ab") as log_fp:
                popen_kwargs: dict = {
                    "shell": True,
                    "stdout": log_fp,
                    "stderr": subprocess.STDOUT,
                    "cwd": resolved_workdir,
                }
                if os.name == "posix":
                    popen_kwargs["preexec_fn"] = os.setsid
                process = subprocess.Popen(command, **popen_kwargs)
        except Exception as e:
            return _err("spawn_failed", str(e))

        record = _JobRecord(
            job_id=job_id,
            process=process,
            started_at=started_at,
            log_path=log_path,
        )
        _JOBS[job_id] = record
        payload = _refresh_record_locked(record)

    return _ok(payload)


def _status_job(job_id: str) -> str:
    if not isinstance(job_id, str) or not job_id:
        return _err("invalid_argument", "job_id must be a non-empty string")

    with _LOCK:
        _cleanup_jobs_locked()
        record = _JOBS.get(job_id)
        if record is None:
            return _err("job_not_found", f"job not found: {job_id}")
        payload = _refresh_record_locked(record)

    return _ok(payload)


def _cancel_job(job_id: str, force: bool = False) -> str:
    if not isinstance(job_id, str) or not job_id:
        return _err("invalid_argument", "job_id must be a non-empty string")
    if not isinstance(force, bool):
        return _err("invalid_argument", "force must be a boolean")

    with _LOCK:
        _cleanup_jobs_locked()
        record = _JOBS.get(job_id)
        if record is None:
            return _err("job_not_found", f"job not found: {job_id}")
        if record.process.poll() is not None:
            if record.finished_at is None:
                record.finished_at = time.time()
            return _err("already_finished", f"job already finished: {job_id}")
        pid = record.process.pid
        process = record.process

    used_signal = "SIGKILL" if force else "SIGTERM"
    sig_value = signal.SIGKILL if force else signal.SIGTERM
    try:
        if os.name == "posix":
            os.killpg(os.getpgid(pid), sig_value)
        elif force:
            process.kill()
        else:
            process.terminate()

        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            if not force:
                used_signal = "SIGKILL"
                if os.name == "posix":
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                else:
                    process.kill()
            process.wait(timeout=2.0)
    except Exception as e:
        return _err("cancel_failed", f"failed to cancel job: {e}")

    with _LOCK:
        record.cancelled = True
        record.cancel_signal = used_signal
        record.finished_at = time.time()
        payload = _refresh_record_locked(record)
        payload["message"] = f"cancelled with {used_signal}"

    return _ok(payload)


def run_job(
    operation: str,
    job_id: str | None = None,
    command: str | None = None,
    workdir: str | None = None,
    force: bool = False,
    *,
    workspace_root: pathlib.Path | str | None = None,
) -> str:
    if operation == "start":
        return _start_job(command or "", workdir=workdir, workspace_root=workspace_root)
    if operation == "status":
        return _status_job(job_id or "")
    if operation == "cancel":
        return _cancel_job(job_id or "", force=force)
    return _err("invalid_argument", "operation must be one of: start, status, cancel")


def register_job_tools(workspace_root: pathlib.Path | str | None = None) -> None:
    def _handler(
        operation: str,
        job_id: str | None = None,
        command: str | None = None,
        workdir: str | None = None,
        force: bool = False,
    ) -> str:
        return run_job(
            operation=operation,
            job_id=job_id,
            command=command,
            workdir=workdir,
            force=force,
            workspace_root=workspace_root,
        )

    register(
        name="run_job",
        description=(
            "管理后台 shell 任务。"
            "operation=start 启动长任务，operation=status 查询高信号状态，"
            "operation=cancel 取消任务。"
        ),
        parameters={
            "operation": {
                "type": "string",
                "description": "操作类型：start | status | cancel",
            },
            "job_id": {
                "type": ["string", "null"],
                "description": "status/cancel 时必填的任务 ID",
            },
            "command": {
                "type": ["string", "null"],
                "description": "start 时必填的 shell 命令",
            },
            "workdir": {
                "type": ["string", "null"],
                "description": "start 时可选的工作目录",
            },
            "force": {
                "type": "boolean",
                "description": "cancel 时是否直接强制终止",
            },
        },
        required=["operation"],
        handler=_handler,
    )
