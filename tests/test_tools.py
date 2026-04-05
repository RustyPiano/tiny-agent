# tests/test_tools.py
import json
import os
import re
import time
from pathlib import Path

import pytest

from agent_framework import _config as config
from agent_framework._config import AgentSettings
from agent_framework.main import bootstrap
from agent_framework.skills import discover_skills
from agent_framework.tools import bash_tool
from agent_framework.tools import grep_tool, list_dir_tool, registry
from agent_framework.tools.bash_tool import run_bash
from agent_framework.tools.file_tools import read_file, write_file
from agent_framework.tools.skill_tool import use_skill


def _reset_tools() -> None:
    registry._TOOLS.clear()
    registry.clear_before_tool_call_hooks()


def test_write_and_read(tmp_path):
    p = str(tmp_path / "hello.txt")
    assert "ok" in write_file(p, "hello world")
    assert read_file(p) == "hello world"


def test_read_line_range(tmp_path):
    p = str(tmp_path / "lines.txt")
    write_file(p, "a\nb\nc\nd\ne")
    result = read_file(p, start_line=2, end_line=4)
    assert "b" in result and "d" in result and "a" not in result


def test_read_line_range_rejects_non_positive_start(tmp_path):
    p = str(tmp_path / "lines.txt")
    write_file(p, "a\nb\nc")
    result = read_file(p, start_line=0, end_line=2)
    assert "[error]" in result


def test_read_line_range_rejects_non_positive_end(tmp_path):
    p = str(tmp_path / "lines.txt")
    write_file(p, "a\nb\nc")
    result = read_file(p, start_line=1, end_line=0)
    assert "[error]" in result


def test_read_line_range_rejects_end_before_start(tmp_path):
    p = str(tmp_path / "lines.txt")
    write_file(p, "a\nb\nc")
    result = read_file(p, start_line=3, end_line=2)
    assert "[error]" in result


def test_write_append(tmp_path):
    p = str(tmp_path / "log.txt")
    write_file(p, "line1\n")
    write_file(p, "line2\n", mode="append")
    assert read_file(p) == "line1\nline2\n"


def test_write_file_rejects_oversized_content(tmp_path):
    p = str(tmp_path / "too_large.txt")
    oversized = "a" * (config.MAX_WRITE_FILE_CHARS + 1)
    result = write_file(p, oversized)

    assert "[error]" in result
    assert str(len(oversized)) in result
    assert str(config.MAX_WRITE_FILE_CHARS) in result
    assert "edit_file" in result
    assert "chunk" in result.lower()
    assert not Path(p).exists()


def test_read_nonexistent():
    assert "[error]" in read_file("/nonexistent/path.txt")


def test_bootstrap_registers_file_tools_with_settings_workspace_root(monkeypatch, tmp_path):
    _reset_tools()
    settings_root = tmp_path / "runtime-root"
    global_root = tmp_path / "global-root"
    settings_root.mkdir()
    global_root.mkdir()
    monkeypatch.setattr(config, "WORKSPACE_ROOT", global_root)

    bootstrap(settings=AgentSettings(workspace_root=settings_root))

    result = registry.execute(
        "write_file",
        {"path": str(settings_root / "hello.txt"), "content": "hello"},
    )

    assert "[ok]" in result
    assert (settings_root / "hello.txt").read_text(encoding="utf-8") == "hello"


def test_bootstrap_registers_list_dir_tool_with_settings_workspace_root(monkeypatch, tmp_path):
    _reset_tools()
    settings_root = tmp_path / "runtime-root"
    global_root = tmp_path / "global-root"
    settings_root.mkdir()
    global_root.mkdir()
    (settings_root / "visible.txt").write_text("x", encoding="utf-8")
    monkeypatch.setattr(config, "WORKSPACE_ROOT", global_root)

    bootstrap(settings=AgentSettings(workspace_root=settings_root))

    result = registry.execute("list_dir", {"path": str(settings_root)})

    assert "visible.txt" in result


def test_bootstrap_registers_edit_file_tool_with_settings_workspace_root(monkeypatch, tmp_path):
    _reset_tools()
    settings_root = tmp_path / "runtime-root"
    global_root = tmp_path / "global-root"
    settings_root.mkdir()
    global_root.mkdir()
    target = settings_root / "editable.txt"
    target.write_text("old value", encoding="utf-8")
    monkeypatch.setattr(config, "WORKSPACE_ROOT", global_root)

    bootstrap(settings=AgentSettings(workspace_root=settings_root))

    result = registry.execute(
        "edit_file",
        {
            "path": str(target),
            "old_str": "old",
            "new_str": "new",
        },
    )

    assert "[ok]" in result
    assert target.read_text(encoding="utf-8") == "new value"


def test_bootstrap_registers_grep_tool_with_settings_workspace_root(monkeypatch, tmp_path):
    _reset_tools()
    settings_root = tmp_path / "runtime-root"
    global_root = tmp_path / "global-root"
    settings_root.mkdir()
    global_root.mkdir()
    target = settings_root / "hello.txt"
    target.write_text("alpha beta\n", encoding="utf-8")
    monkeypatch.setattr(config, "WORKSPACE_ROOT", global_root)

    bootstrap(settings=AgentSettings(workspace_root=settings_root))

    result = registry.execute("grep", {"pattern": "alpha", "path": str(target)})

    assert "hello.txt" in result


def test_bootstrap_registers_run_bash_with_settings_workspace_root(monkeypatch, tmp_path):
    _reset_tools()
    settings_root = tmp_path / "runtime-root"
    global_root = tmp_path / "global-root"
    settings_root.mkdir()
    global_root.mkdir()
    observed: dict = {}
    monkeypatch.setattr(config, "WORKSPACE_ROOT", global_root)

    class _FakePopen:
        returncode = 0

        def __init__(self, *args, **kwargs):
            observed["cwd"] = kwargs.get("cwd")

        def communicate(self, timeout=None):
            _ = timeout
            return ("ok\n", "")

    monkeypatch.setattr(bash_tool.subprocess, "Popen", _FakePopen)
    bootstrap(settings=AgentSettings(workspace_root=settings_root))

    result = registry.execute("run_bash", {"command": "pwd"})

    assert result == "ok\n"
    assert observed["cwd"] == str(settings_root)


def test_list_dir_truncates_with_continuation_metadata(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(list_dir_tool, "_MAX_LIST_DIR_ENTRIES", 3, raising=False)

    root = tmp_path / "demo"
    root.mkdir()
    for name in ["a.txt", "b.txt", "c.txt", "d.txt", "e.txt"]:
        (root / name).write_text(name, encoding="utf-8")

    result = list_dir_tool.list_dir(str(root))
    data = json.loads(result)

    assert data["ok"] is True
    assert data["truncated"] is True
    assert data["next_offset"] == 3
    assert data["preview"] == ["a.txt", "b.txt", "c.txt"]
    assert data["continuation"]["tool"] == "list_dir"
    assert data["continuation"]["offset"] == 3


def test_list_dir_supports_follow_up_pagination(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(list_dir_tool, "_MAX_LIST_DIR_ENTRIES", 3, raising=False)

    root = tmp_path / "demo"
    root.mkdir()
    for name in ["a.txt", "b.txt", "c.txt", "d.txt", "e.txt"]:
        (root / name).write_text(name, encoding="utf-8")

    first = json.loads(list_dir_tool.list_dir(str(root), offset=0, limit=3))
    second = json.loads(
        list_dir_tool.list_dir(str(root), offset=first["next_offset"], limit=3)
    )

    assert first["truncated"] is True
    assert first["preview"] == ["a.txt", "b.txt", "c.txt"]
    assert second["preview"] == ["d.txt", "e.txt"]
    assert second["truncated"] is False
    assert "continuation" not in second


def test_grep_truncates_with_continuation_metadata(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(grep_tool, "_MAX_GREP_MATCHES", 2, raising=False)

    root = tmp_path / "demo"
    root.mkdir()
    for idx in range(4):
        (root / f"file-{idx}.txt").write_text(f"alpha {idx}\n", encoding="utf-8")

    result = grep_tool.grep("alpha", str(root))
    data = json.loads(result)

    assert data["ok"] is True
    assert data["truncated"] is True
    assert data["next_offset"] == 2
    assert len(data["preview"]) == 2
    assert data["continuation"]["tool"] == "grep"
    assert data["continuation"]["offset"] == 2


def test_grep_supports_follow_up_pagination(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(grep_tool, "_MAX_GREP_MATCHES", 2, raising=False)

    root = tmp_path / "demo"
    root.mkdir()
    for idx in range(4):
        (root / f"file-{idx}.txt").write_text(f"alpha {idx}\n", encoding="utf-8")

    first = json.loads(grep_tool.grep("alpha", str(root), offset=0, limit=2))
    second = json.loads(
        grep_tool.grep("alpha", str(root), offset=first["next_offset"], limit=2)
    )

    assert first["truncated"] is True
    assert len(first["preview"]) == 2
    assert len(second["preview"]) == 2
    assert second["truncated"] is False


def test_grep_truncates_wide_match_text(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(grep_tool, "_MAX_GREP_MATCHES", 5, raising=False)
    monkeypatch.setattr(grep_tool, "_MAX_GREP_LINE_CHARS", 20, raising=False)

    root = tmp_path / "demo"
    root.mkdir()
    (root / "wide.txt").write_text("alpha " + ("x" * 200) + "\n", encoding="utf-8")

    result = grep_tool.grep("alpha", str(root))
    data = json.loads(result)

    assert data["ok"] is True
    assert data["truncated"] is False
    assert data["preview_truncated"] is True
    assert len(data["preview"]) == 1
    assert "...[截断" in data["preview"][0]
    assert "continuation" not in data


def test_bash_echo():
    assert "hello" in run_bash("echo hello")


def test_bash_supports_shell_operators():
    result = run_bash("echo one && echo two")
    assert "one" in result
    assert "two" in result


def test_bash_stderr():
    result = run_bash("ls /nonexistent_path_xyz")
    assert "[stderr]" in result or "No such" in result


def test_bash_timeout():
    result = run_bash("sleep 10", timeout=1)
    assert "[timeout]" in result


def test_bash_blocked():
    result = run_bash("rm -rf /")
    assert "[blocked]" in result


def test_bash_rejects_detached_command_with_job_guidance():
    result = run_bash("sleep 1 &")

    assert "[blocked]" in result
    assert "run_job" in result


def test_bash_rejects_background_operator_not_only_at_end():
    result = run_bash("sleep 30 & echo done")

    assert "[blocked]" in result
    assert "run_job" in result


def test_bash_allows_fd_redirection_2_to_1_not_detached():
    result = run_bash("python3 -c \"import sys; sys.stderr.write('redir-ok\\n')\" 2>&1")

    assert "[blocked]" not in result
    assert "redir-ok" in result


def test_detached_parser_allows_non_background_ampersand_forms():
    assert bash_tool._is_detached_command("echo hi >&2") is False
    assert bash_tool._is_detached_command("ls /nonexistent |& cat") is False


def test_bash_missing_timeout_binary_returns_platform_hint(monkeypatch):
    monkeypatch.setattr(bash_tool.shutil, "which", lambda _: None)
    monkeypatch.setattr(bash_tool.sys, "platform", "darwin")

    result = run_bash("timeout 3 sleep 1")

    assert "[error]" in result
    assert "timeout" in result
    assert "brew install coreutils" in result


def test_bash_missing_timeout_binary_returns_cross_platform_hint(monkeypatch):
    monkeypatch.setattr(bash_tool.shutil, "which", lambda _: None)
    monkeypatch.setattr(bash_tool.sys, "platform", "linux")

    result = run_bash("timeout 3 sleep 1")

    assert "[error]" in result
    assert "timeout" in result
    assert "coreutils" in result


def test_bash_default_timeout_is_fixed_short_timeout(monkeypatch):
    observed_timeouts: list[int | None] = []

    class _FakePopen:
        def __init__(self, *args, **kwargs):
            observed_timeouts.append(None)
            self.returncode = 0

        def communicate(self, timeout=None):
            observed_timeouts[-1] = timeout
            return ("ok", "")

    monkeypatch.setattr(bash_tool.subprocess, "Popen", _FakePopen)

    run_bash("npm install")
    run_bash("pytest tests/test_tools.py")
    run_bash("echo hello")

    assert observed_timeouts == [30, 30, 30]


def test_bash_timeout_kills_child_process_tree(tmp_path):
    if os.name != "posix":
        return

    pidfile = tmp_path / "child.pid"
    command = (
        "python3 -c \"import pathlib,subprocess,time; "
        "p=subprocess.Popen(['python3','-c','import time; time.sleep(30)']); "
        f"pathlib.Path(r'{pidfile}').write_text(str(p.pid)); "
        "time.sleep(30)\""
    )

    result = run_bash(
        command,
        timeout=1,
        workdir=str(tmp_path),
        workspace_root=tmp_path,
    )

    assert "[timeout]" in result
    for _ in range(20):
        if pidfile.exists():
            break
        time.sleep(0.1)
    assert pidfile.exists() is True

    child_pid = int(pidfile.read_text(encoding="utf-8"))
    time.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


def test_bash_truncation_includes_saved_full_output_path(monkeypatch):
    monkeypatch.setattr(bash_tool, "OUTPUT_TRUNCATE", 10)

    class _FakePopen:
        returncode = 0

        def __init__(self, *args, **kwargs):
            pass

        def communicate(self, timeout=None):
            return ("x" * 50, "")

    monkeypatch.setattr(bash_tool.subprocess, "Popen", _FakePopen)

    result = run_bash("echo oversize")

    assert "[输出已截断" in result
    assert "完整输出已保存到:" in result
    match = re.search(r"完整输出已保存到:\s*(\S+)", result)
    assert match is not None
    assert Path(match.group(1)).exists()


def test_bash_explicit_timeout_uses_legacy_behavior(monkeypatch):
    observed_timeouts: list[int | None] = []

    class _FakePopen:
        def __init__(self, *args, **kwargs):
            observed_timeouts.append(None)
            self.returncode = 0

        def communicate(self, timeout=None):
            observed_timeouts[-1] = timeout
            return ("", "")

    monkeypatch.setattr(bash_tool.subprocess, "Popen", _FakePopen)

    result = run_bash("true", timeout=30)

    assert observed_timeouts == [30]
    assert "[ok] 命令执行完毕，退出码 0，无输出" in result


def test_use_skill_returns_prompt_text(tmp_path):
    project_skills = tmp_path / ".agents" / "skills"
    (project_skills / "coding").mkdir(parents=True, exist_ok=True)
    (project_skills / "coding" / "SKILL.md").write_text(
        "---\nname: coding\ndescription: coding skill\n---\n编码规范",
        encoding="utf-8",
    )
    discover_skills(project_dir=project_skills, global_dir=Path("/nonexistent"))
    result = use_skill("coding")
    assert "编码规范" in result


def test_use_skill_unknown_returns_error():
    result = use_skill("unknown_skill")
    assert "[error]" in result
