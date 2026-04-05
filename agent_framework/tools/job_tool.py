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


_MAX_JOB_RECORDS = 256
_TERMINAL_JOB_TTL_SEC = 3600.0
_MAX_RUNNING_JOBS = 16
_MAX_JOB_LOG_BYTES = 4000


_JOBS: dict[str, _JobRecord] = {}
_LOCK = threading.RLock()


def _ok(payload: dict) -> str:
    data = {"ok": True}
    data.update(payload)
    return json.dumps(data, ensure_ascii=False)


def _err(error: str, message: str) -> str:
    return json.dumps({"ok": False, "error": error, "message": message}, ensure_ascii=False)


def _get_job(job_id: str) -> _JobRecord | None:
    with _LOCK:
        return _JOBS.get(job_id)


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


def _finalize_completed_jobs_locked() -> None:
    """Mark completed processes with finished_at timestamp."""
    for record in _JOBS.values():
        if record.finished_at is None and record.process.poll() is not None:
            record.finished_at = time.time()


def _running_jobs_count_locked() -> int:
    running = 0
    for record in _JOBS.values():
        if record.process.poll() is None:
            running += 1
    return running


def start_job(
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
        return _err(
            "invalid_argument", "background/detached operator '&' is not allowed in start_job"
        )

    resolved_workdir, err = _resolve_workdir(workdir, workspace_root=workspace_root)
    if err is not None:
        return _err("invalid_argument", err)

    with _LOCK:
        _cleanup_jobs_locked()
        _finalize_completed_jobs_locked()
        if _running_jobs_count_locked() >= _MAX_RUNNING_JOBS:
            return _err(
                "too_many_running_jobs",
                f"running jobs exceed limit: {_MAX_RUNNING_JOBS}",
            )

        job_id = f"job_{uuid.uuid4().hex[:12]}"
        started_at = time.time()

        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", delete=False, prefix="agent_job_", suffix=".log"
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

    return _ok(
        {
            "job_id": job_id,
            "pid": process.pid,
            "status": "running",
            "started_at": started_at,
            "log_path": log_path,
        }
    )


def poll_job(job_id: str) -> str:
    if not isinstance(job_id, str) or not job_id:
        return _err("invalid_argument", "job_id must be a non-empty string")

    with _LOCK:
        _cleanup_jobs_locked()
        record = _JOBS.get(job_id)
        if record is None:
            return _err("job_not_found", f"job not found: {job_id}")

        exit_code = record.process.poll()
        status = "running"
        if exit_code is not None:
            status = "cancelled" if record.cancelled else "exited"
            if record.finished_at is None:
                record.finished_at = time.time()

        duration_sec = max(0.0, time.time() - record.started_at)
    return _ok(
        {
            "job_id": record.job_id,
            "status": status,
            "pid": record.process.pid,
            "exit_code": exit_code,
            "duration_sec": duration_sec,
        }
    )


def read_job_log(job_id: str, offset: int = 0, limit: int = 4000) -> str:
    if not isinstance(job_id, str) or not job_id:
        return _err("invalid_argument", "job_id must be a non-empty string")
    if not isinstance(offset, int) or offset < 0:
        return _err("invalid_argument", "offset must be an integer >= 0")
    if not isinstance(limit, int) or limit < 0:
        return _err("invalid_argument", "limit must be an integer >= 0")

    effective_limit = min(limit, _MAX_JOB_LOG_BYTES)

    with _LOCK:
        _cleanup_jobs_locked()
        record = _JOBS.get(job_id)
        if record is None:
            return _err("job_not_found", f"job not found: {job_id}")
        log_path = record.log_path

    try:
        with open(log_path, "rb") as fp:
            fp.seek(offset)
            chunk = fp.read(effective_limit)
            next_offset = offset + len(chunk)
            file_size = fp.seek(0, os.SEEK_END)
    except Exception as e:
        return _err("io_error", f"failed to read log: {e}")

    content = chunk.decode("utf-8", errors="replace")
    truncated = effective_limit > 0 and next_offset < file_size
    return _ok(
        {
            "job_id": record.job_id,
            "offset": offset,
            "limit": effective_limit,
            "next_offset": next_offset,
            "bytes_read": len(chunk),
            "eof": next_offset >= file_size,
            "truncated": truncated,
            "content": content,
            "preview": content[:200],
            **(
                {
                    "continuation": {
                        "tool": "read_job_log",
                        "job_id": job_id,
                        "offset": next_offset,
                        "limit": effective_limit,
                    }
                }
                if truncated
                else {}
            ),
        }
    )

def cancel_job(job_id: str, force: bool = False) -> str:
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

    return _ok(
        {
            "job_id": record.job_id,
            "status": "cancelled",
            "signal": used_signal,
            "exit_code": process.poll(),
        }
    )


def register_job_tools(workspace_root: pathlib.Path | str | None = None) -> None:
    def _start_handler(command: str, workdir: str | None = None) -> str:
        return start_job(command, workdir=workdir, workspace_root=workspace_root)

    register(
        name="start_job",
        description="Start a managed background shell job and return its job metadata.",
        parameters={
            "command": {"type": "string", "description": "Shell command to run in background"},
            "workdir": {
                "type": ["string", "null"],
                "description": "Optional working directory for command execution",
            },
        },
        required=["command"],
        handler=_start_handler,
    )
    register(
        name="poll_job",
        description="Poll a managed background job status and exit information.",
        parameters={
            "job_id": {"type": "string", "description": "Job id returned by start_job"},
        },
        required=["job_id"],
        handler=poll_job,
    )
    register(
        name="read_job_log",
        description="Read incremental bytes from managed job log by UTF-8 byte offset.",
        parameters={
            "job_id": {"type": "string", "description": "Job id returned by start_job"},
            "offset": {
                "type": "integer",
                "description": "UTF-8 byte offset to start reading log content",
            },
            "limit": {"type": "integer", "description": "Maximum bytes to read from log"},
        },
        required=["job_id"],
        handler=read_job_log,
    )
    register(
        name="cancel_job",
        description="Cancel a managed background job with SIGTERM or SIGKILL.",
        parameters={
            "job_id": {"type": "string", "description": "Job id returned by start_job"},
            "force": {
                "type": "boolean",
                "description": "If true, use SIGKILL directly; otherwise use SIGTERM then escalate",
            },
        },
        required=["job_id"],
        handler=cancel_job,
    )
