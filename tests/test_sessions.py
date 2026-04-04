# tests/test_sessions.py
import json
import pathlib

import pytest

import config
from sessions.migrations import migrate
from sessions.store import SCHEMA_VERSION, delete, list_sessions, load, save


@pytest.fixture(autouse=True)
def _patch_sessions_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SESSIONS_DIR", str(tmp_path / "sessions"))


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
