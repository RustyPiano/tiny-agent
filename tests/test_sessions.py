# tests/test_sessions.py
import json
import pathlib

import pytest

from agent_framework import _config as config, main
from agent_framework._config import AgentSettings
from agent_framework.core.agent import _load_history_with_provider_check
from agent_framework.core.logging import RunContext
from agent_framework.llm.base import BaseLLMProvider, LLMResponse
from agent_framework.sessions.migrations import migrate
from agent_framework.sessions.store import SCHEMA_VERSION, delete, list_sessions, load, save


@pytest.fixture(autouse=True)
def _patch_sessions_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SESSIONS_DIR", str(tmp_path / "sessions"))


class _SingleResponseProvider(BaseLLMProvider):
    def __init__(self, text: str):
        self._response = LLMResponse(
            text=text,
            tool_calls=[],
            stop_reason="end_turn",
            assistant_message={"role": "assistant", "content": text},
        )

    def chat(self, messages, system, tools, max_tokens=16000) -> LLMResponse:
        return self._response

    def format_tool_result(self, tool_call_id: str, content: str) -> dict:
        return {"type": "tool_result", "tool_use_id": tool_call_id, "content": content}

    def tool_results_as_message(self, results: list[dict]) -> list[dict]:
        return [{"role": "user", "content": results}]


class TestMigrations:
    def test_migrate_adds_schema_version(self):
        data = {"messages": [{"role": "user", "content": "hi"}], "provider": "anthropic"}
        result = migrate(data)
        assert result["schema_version"] == 1

    def test_migrate_noop_when_current(self):
        data = {"schema_version": 1, "messages": [], "provider": "x"}
        result = migrate(data)
        assert result == data

    def test_migrate_preserves_fields(self):
        data = {"messages": [1, 2], "provider": "openai"}
        result = migrate(data)
        assert result["messages"] == [1, 2]
        assert result["provider"] == "openai"


class TestStore:
    def test_save_includes_schema_version(self, tmp_path):
        save("test1", [{"role": "user", "content": "hello"}], "anthropic")
        raw = json.loads(pathlib.Path(config.SESSIONS_DIR, "test1.json").read_text())
        assert raw["schema_version"] == SCHEMA_VERSION
        assert raw["provider"] == "anthropic"

    def test_round_trip(self):
        msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        save("rt", msgs, "openai")
        loaded_msgs, provider = load("rt")
        assert loaded_msgs == msgs
        assert provider == "openai"

    def test_load_old_list_format(self, tmp_path):
        p = pathlib.Path(config.SESSIONS_DIR)
        p.mkdir(exist_ok=True)
        (p / "old.json").write_text(
            json.dumps([{"role": "user", "content": "legacy"}]),
            encoding="utf-8",
        )
        msgs, provider = load("old")
        assert msgs == [{"role": "user", "content": "legacy"}]
        assert provider == "unknown"

    def test_load_missing_returns_empty(self):
        msgs, provider = load("nonexistent")
        assert msgs == []
        assert provider == ""

    def test_load_corrupt_returns_empty(self, tmp_path):
        p = pathlib.Path(config.SESSIONS_DIR)
        p.mkdir(exist_ok=True)
        (p / "bad.json").write_text("not json {{{", encoding="utf-8")
        msgs, provider = load("bad")
        assert msgs == []
        assert provider == ""

    def test_delete(self):
        save("delme", [], "x")
        assert delete("delme") is True
        assert delete("delme") is False

    def test_list_sessions(self):
        save("a", [], "x")
        save("b", [], "y")
        names = list_sessions()
        assert "a" in names
        assert "b" in names

    def test_save_rejects_path_traversal_session_id(self):
        with pytest.raises(ValueError):
            save("../../escape", [], "x")

    def test_load_rejects_path_traversal_session_id(self):
        with pytest.raises(ValueError):
            load("../escape")

    def test_delete_rejects_path_traversal_session_id(self):
        with pytest.raises(ValueError):
            delete("../escape")

    def test_runtime_persistence_uses_settings_sessions_dir(self, tmp_path, monkeypatch):
        main.bootstrap(AgentSettings(workspace_root=tmp_path))
        runtime_sessions = tmp_path / "runtime-sessions"
        global_sessions = tmp_path / "global-sessions"
        monkeypatch.setattr(config, "SESSIONS_DIR", str(global_sessions))

        from agent_framework.core.agent import run

        result = run(
            "remember this",
            provider=_SingleResponseProvider("done"),
            session_id="runtime-session",
            settings=AgentSettings(
                workspace_root=tmp_path,
                sessions_dir=runtime_sessions,
            ),
        )

        assert result == "done"
        assert (runtime_sessions / "runtime-session.json").exists()
        assert not (global_sessions / "runtime-session.json").exists()

    def test_load_history_with_provider_check_uses_settings_sessions_dir(self, tmp_path, monkeypatch):
        runtime_sessions = tmp_path / "runtime-sessions"
        runtime_sessions.mkdir()
        global_sessions = tmp_path / "global-sessions"
        global_sessions.mkdir()
        monkeypatch.setattr(config, "SESSIONS_DIR", str(global_sessions))
        (runtime_sessions / "session-a.json").write_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "provider": "anthropic",
                    "messages": [{"role": "user", "content": "from runtime"}],
                }
            ),
            encoding="utf-8",
        )

        history = _load_history_with_provider_check(
            "session-a",
            "anthropic",
            RunContext(session_id="session-a"),
            settings=AgentSettings(
                workspace_root=tmp_path,
                sessions_dir=runtime_sessions,
            ),
        )

        assert history == [{"role": "user", "content": "from runtime"}]

    def test_save_creates_nested_settings_sessions_dir_parents(self, tmp_path):
        nested_sessions = tmp_path / "nested" / "runtime" / "sessions"

        save(
            "nested-session",
            [{"role": "user", "content": "hello"}],
            "anthropic",
            sessions_dir=nested_sessions,
        )

        assert (nested_sessions / "nested-session.json").exists()
