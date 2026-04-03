# tests/test_tools.py
from pathlib import Path

from skills import discover_skills
from tools.bash_tool import run_bash
from tools.file_tools import read_file, write_file
from tools.skill_tool import use_skill


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


def test_read_nonexistent():
    assert "[error]" in read_file("/nonexistent/path.txt")


def test_bash_echo():
    assert "hello" in run_bash("echo hello")


def test_bash_stderr():
    result = run_bash("ls /nonexistent_path_xyz")
    assert "[stderr]" in result or "No such" in result


def test_bash_timeout():
    result = run_bash("sleep 10", timeout=1)
    assert "[timeout]" in result


def test_bash_blocked():
    result = run_bash("rm -rf /")
    assert "[blocked]" in result


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
