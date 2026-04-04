# sessions/store.py
import json
import pathlib

from agent_framework import _config as config
from agent_framework.sessions.migrations import migrate

SCHEMA_VERSION = 1


def _sessions_root() -> pathlib.Path:
    root = pathlib.Path(config.SESSIONS_DIR).resolve()
    root.mkdir(exist_ok=True)
    return root


def _validate_session_id(session_id: str) -> str:
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id 必须是非空字符串")
    if any(part in session_id for part in ("/", "\\", "..", "\x00")):
        raise ValueError(f"非法 session_id: {session_id!r}")
    return session_id


def _path(session_id: str) -> pathlib.Path:
    safe_id = _validate_session_id(session_id)
    root = _sessions_root()
    target = (root / f"{safe_id}.json").resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"session_id 路径越界: {session_id!r}") from exc
    return target


def save(session_id: str, messages: list[dict], provider_type: str = "unknown") -> None:
    data = {
        "schema_version": SCHEMA_VERSION,
        "provider": provider_type,
        "messages": messages,
    }
    _path(session_id).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load(session_id: str) -> tuple[list[dict], str]:
    """返回 (messages, provider_type) 元组。如果文件不存在或损坏，返回 ([], "")。"""
    p = _path(session_id)
    if not p.exists():
        return [], ""
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return [], ""
    # 兼容旧格式（纯列表）
    if isinstance(data, list):
        data = {"schema_version": 0, "provider": "unknown", "messages": data}
    # 执行迁移
    data = migrate(data)
    return data.get("messages", []), data.get("provider", "unknown")


def delete(session_id: str) -> bool:
    p = _path(session_id)
    if p.exists():
        p.unlink()
        return True
    return False


def list_sessions() -> list[str]:
    return [p.stem for p in pathlib.Path(config.SESSIONS_DIR).glob("*.json")]
