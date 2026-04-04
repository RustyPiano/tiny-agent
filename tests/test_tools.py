# tests/test_tools.py
import re
from pathlib import Path

from agent_framework import _config as config
from agent_framework.skills import discover_skills
from agent_framework.tools import bash_tool
from agent_framework.tools.bash_tool import run_bash
from agent_framework.tools.file_tools import read_file, write_file
from agent_framework.tools.skill_tool import use_skill


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
    assert "start_job" in result
    assert "poll_job" in result
    assert "read_job_log" in result
    assert "cancel_job" in result


def test_bash_rejects_background_operator_not_only_at_end():
    result = run_bash("sleep 30 & echo done")

    assert "[blocked]" in result
    assert "start_job" in result


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


def test_bash_adaptive_timeout_for_long_commands(monkeypatch):
    observed_timeouts: list[int | None] = []

    def fake_run(*args, **kwargs):
        observed_timeouts.append(kwargs.get("timeout"))
        return bash_tool.subprocess.CompletedProcess(
            args=args[0], returncode=0, stdout="ok", stderr=""
        )

    monkeypatch.setattr(bash_tool.subprocess, "run", fake_run)

    run_bash("npm install")
    run_bash("pytest tests/test_tools.py")
    run_bash("echo hello")

    assert observed_timeouts[0] == 300
    assert observed_timeouts[1] == 180
    assert observed_timeouts[2] == 30


def test_bash_truncation_includes_saved_full_output_path(monkeypatch):
    monkeypatch.setattr(bash_tool, "OUTPUT_TRUNCATE", 10)

    def fake_run(*args, **kwargs):
        return bash_tool.subprocess.CompletedProcess(
            args=args[0], returncode=0, stdout="x" * 50, stderr=""
        )

    monkeypatch.setattr(bash_tool.subprocess, "run", fake_run)

    result = run_bash("echo oversize")

    assert "[输出已截断" in result
    assert "完整输出已保存到:" in result
    match = re.search(r"完整输出已保存到:\s*(\S+)", result)
    assert match is not None
    assert Path(match.group(1)).exists()


def test_bash_explicit_timeout_uses_legacy_behavior(monkeypatch):
    observed_timeouts: list[int | None] = []

    def fake_run(*args, **kwargs):
        observed_timeouts.append(kwargs.get("timeout"))
        return bash_tool.subprocess.CompletedProcess(
            args=args[0], returncode=0, stdout="", stderr=""
        )

    monkeypatch.setattr(bash_tool.subprocess, "run", fake_run)

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
