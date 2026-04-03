# tests/test_skills.py
from skills import load_builtin_skills
from skills.registry import list_skills, load


def test_builtin_skills_load():
    load_builtin_skills()
    names = list_skills()
    assert "coding" in names
    assert "safe_ops" in names
    assert "project_explorer" in names


def test_skill_content_nonempty():
    load_builtin_skills()
    text = load(["coding"])
    assert len(text) > 20


def test_unknown_skill_ignored():
    load_builtin_skills()
    text = load(["nonexistent_skill"])
    assert text == ""
