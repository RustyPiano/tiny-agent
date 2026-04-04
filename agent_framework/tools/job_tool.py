import json
import os
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass

from agent_framework.tools.registry import register


@dataclass
class _JobRecord:
    job_id: str
    process: subprocess.Popen
    started_at: float
    log_path: str
    cancelled: bool = False
    cancel_signal: str | None = None


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


def start_job(command: str, workdir: str | None = None) -> str:
    if not isinstance(command, str) or not command.strip():
        return _err("invalid_argument", "command must be a non-empty string")
    if workdir is not None and not isinstance(workdir, str):
        return _err("invalid_argument", "workdir must be a string or null")

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
                "cwd": workdir,
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
    with _LOCK:
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

    record = _get_job(job_id)
    if record is None:
        return _err("job_not_found", f"job not found: {job_id}")

    exit_code = record.process.poll()
    status = "running"
    if exit_code is not None:
        status = "cancelled" if record.cancelled else "exited"

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

    record = _get_job(job_id)
    if record is None:
        return _err("job_not_found", f"job not found: {job_id}")

    try:
        with open(record.log_path, "rb") as fp:
            fp.seek(offset)
            chunk = fp.read(limit)
            next_offset = offset + len(chunk)
            file_size = fp.seek(0, os.SEEK_END)
    except Exception as e:
        return _err("invalid_argument", f"failed to read log: {e}")

    return _ok(
        {
            "job_id": record.job_id,
            "offset": offset,
            "next_offset": next_offset,
            "eof": next_offset >= file_size,
            "content": chunk.decode("utf-8", errors="replace"),
        }
    )


def cancel_job(job_id: str, force: bool = False) -> str:
    if not isinstance(job_id, str) or not job_id:
        return _err("invalid_argument", "job_id must be a non-empty string")
    if not isinstance(force, bool):
        return _err("invalid_argument", "force must be a boolean")

    record = _get_job(job_id)
    if record is None:
        return _err("job_not_found", f"job not found: {job_id}")

    if record.process.poll() is not None:
        return _err("already_finished", f"job already finished: {job_id}")

    used_signal = "SIGKILL" if force else "SIGTERM"
    sig_value = signal.SIGKILL if force else signal.SIGTERM

    try:
        if os.name == "posix":
            os.killpg(os.getpgid(record.process.pid), sig_value)
        elif force:
            record.process.kill()
        else:
            record.process.terminate()

        try:
            record.process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            if not force:
                used_signal = "SIGKILL"
                if os.name == "posix":
                    os.killpg(os.getpgid(record.process.pid), signal.SIGKILL)
                else:
                    record.process.kill()
            record.process.wait(timeout=2.0)
    except Exception as e:
        return _err("invalid_argument", f"failed to cancel job: {e}")

    with _LOCK:
        record.cancelled = True
        record.cancel_signal = used_signal

    return _ok(
        {
            "job_id": record.job_id,
            "status": "cancelled",
            "signal": used_signal,
            "exit_code": record.process.poll(),
        }
    )


def register_job_tools() -> None:
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
        handler=start_job,
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
