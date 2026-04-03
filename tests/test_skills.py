# tests/test_skills.py
from pathlib import Path

from core.prompt_builder import build_system_prompt
from skills.registry import (
    clear_skills,
    discover_skills,
    get_skill_metadata,
    get_skill_prompt,
    load,
    render_available_skills,
)
from tools.skill_tool import use_skill


def _write_skill(root: Path, folder_name: str, *, name: str, description: str, body: str) -> None:
    skill_dir = root / folder_name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "\n".join(
            [
                "---",
                f"name: {name}",
                f"description: {description}",
                "---",
                body,
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_discover_skills_from_project_and_global(tmp_path):
    global_root = tmp_path / "global"
    project_root = tmp_path / "project"

    _write_skill(
        global_root,
        "global_only",
        name="global_only",
        description="Global only skill",
        body="global body",
    )
    _write_skill(
        project_root,
        "project_only",
        name="project_only",
        description="Project only skill",
        body="project body",
    )

    summary = discover_skills(project_dir=project_root, global_dir=global_root)

    assert summary["discovered"] == 2
    assert summary["loaded"] == 2
    assert summary["overridden"] == 0
    assert summary["total"] == 2

    names = [item["name"] for item in get_skill_metadata()]
    assert "global_only" in names
    assert "project_only" in names


def test_project_skills_override_global_by_name(tmp_path):
    global_root = tmp_path / "global"
    project_root = tmp_path / "project"

    _write_skill(
        global_root,
        "same_name",
        name="shared",
        description="Global description",
        body="global prompt body",
    )
    _write_skill(
        project_root,
        "same_name",
        name="shared",
        description="Project description",
        body="project prompt body",
    )

    summary = discover_skills(project_dir=project_root, global_dir=global_root)

    assert summary["overridden"] == 1
    assert summary["total"] == 1
    assert get_skill_prompt("shared") == "project prompt body"


def test_render_available_skills_contains_name_and_description(tmp_path):
    global_root = tmp_path / "global"
    project_root = tmp_path / "project"

    _write_skill(
        global_root,
        "alpha",
        name="alpha",
        description="Alpha description",
        body="alpha body",
    )

    discover_skills(project_dir=project_root, global_dir=global_root)
    text = render_available_skills()

    assert "## Available Skills" in text
    assert "alpha: Alpha description" in text


def test_use_skill_returns_discovered_body(tmp_path):
    global_root = tmp_path / "global"
    project_root = tmp_path / "project"

    _write_skill(
        project_root,
        "coding",
        name="coding",
        description="Coding skill",
        body="编码规范：保持函数简洁。",
    )

    discover_skills(project_dir=project_root, global_dir=global_root)
    result = use_skill("coding")

    assert "编码规范" in result


def test_use_skill_unknown_returns_error():
    clear_skills()
    result = use_skill("unknown_skill")
    assert "[error]" in result


def test_prompt_builder_includes_available_skills_section(tmp_path):
    global_root = tmp_path / "global"
    project_root = tmp_path / "project"

    _write_skill(
        global_root,
        "safe_ops",
        name="safe_ops",
        description="Safety operations",
        body="avoid destructive operations",
    )

    discover_skills(project_dir=project_root, global_dir=global_root)
    prompt = build_system_prompt()

    assert "## Available Skills" in prompt
    assert "safe_ops: Safety operations" in prompt


def test_load_keeps_backward_compatible_preload_behavior(tmp_path):
    global_root = tmp_path / "global"
    project_root = tmp_path / "project"

    _write_skill(
        project_root,
        "a",
        name="a",
        description="A",
        body="AAA",
    )
    _write_skill(
        project_root,
        "b",
        name="b",
        description="B",
        body="BBB",
    )

    discover_skills(project_dir=project_root, global_dir=global_root)
    text = load(["a", "missing", "b"])
    assert text == "AAA\n\nBBB"


def test_discover_skills_counts_failed_for_empty_skill_body(tmp_path):
    global_root = tmp_path / "global"
    project_root = tmp_path / "project"

    _write_skill(
        global_root,
        "broken",
        name="broken",
        description="Broken skill",
        body="",
    )

    summary = discover_skills(project_dir=project_root, global_dir=global_root)

    assert summary["discovered"] == 1
    assert summary["loaded"] == 0
    assert summary["failed"] >= 1
    assert summary["total"] == 0
    assert summary["failure_details"]
    assert summary["failure_details"][0]["reason"] == "empty_prompt"
    assert summary["failure_details"][0]["path"].endswith("broken/SKILL.md")
