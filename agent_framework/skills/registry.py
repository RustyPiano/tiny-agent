from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict


class SkillEntry(TypedDict):
    name: str
    description: str
    prompt: str
    source: str
    path: str


_SKILLS: dict[str, SkillEntry] = {}
_MAX_FAILURE_DETAILS = 20


def clear_skills() -> None:
    _SKILLS.clear()


def discover_skills(project_dir: Path, global_dir: Path) -> dict[str, Any]:
    clear_skills()

    discovered = 0
    loaded = 0
    overridden = 0
    failed = 0
    failure_details: list[dict[str, str]] = []

    global_discovered, global_count_failed, global_count_details = _count_candidate_skills(
        global_dir
    )
    discovered += global_discovered
    failed += global_count_failed
    for detail in global_count_details:
        _append_failure_detail(failure_details, detail["path"], detail["reason"])

    global_loaded, global_load_failed, global_load_details = _load_from_root(
        global_dir, source="global"
    )
    loaded += global_loaded
    failed += global_load_failed
    for detail in global_load_details:
        _append_failure_detail(failure_details, detail["path"], detail["reason"])

    project_discovered, project_count_failed, project_count_details = _count_candidate_skills(
        project_dir
    )
    discovered += project_discovered
    failed += project_count_failed
    for detail in project_count_details:
        _append_failure_detail(failure_details, detail["path"], detail["reason"])

    before_project_load = len(_SKILLS)
    project_loaded, project_load_failed, project_load_details = _load_from_root(
        project_dir, source="project"
    )
    loaded += project_loaded
    failed += project_load_failed
    for detail in project_load_details:
        _append_failure_detail(failure_details, detail["path"], detail["reason"])

    after_project_load = len(_SKILLS)
    overridden = before_project_load + project_loaded - after_project_load

    return {
        "discovered": discovered,
        "loaded": loaded,
        "overridden": overridden,
        "failed": failed,
        "total": len(_SKILLS),
        "failure_details": failure_details,
    }


def list_skills() -> list[str]:
    return sorted(_SKILLS.keys())


def load(skill_names: list[str]) -> str:
    parts = [entry["prompt"] for n in skill_names if (entry := _SKILLS.get(n))]
    return "\n\n".join(parts)


def get_skill_metadata() -> list[dict[str, str]]:
    return [
        {"name": entry["name"], "description": entry["description"]}
        for entry in sorted(_SKILLS.values(), key=lambda item: item["name"])
    ]


def get_skill_prompt(name: str) -> str | None:
    entry = _SKILLS.get(name)
    if entry is None:
        return None
    return entry["prompt"]


def render_available_skills() -> str:
    metadata = get_skill_metadata()
    if not metadata:
        return ""

    lines = [
        "## Available Skills",
        "以下是可用技能；当需要某个技能的完整指令时，调用工具 `use_skill(name)`：",
    ]
    for item in metadata:
        lines.append(f"- {item['name']}: {item['description']}")
    return "\n".join(lines)


def _count_candidate_skills(root_dir: Path) -> tuple[int, int, list[dict[str, str]]]:
    if not root_dir.exists() or not root_dir.is_dir():
        return 0, 0, []

    count = 0
    failed = 0
    failure_details: list[dict[str, str]] = []

    try:
        entries = list(root_dir.iterdir())
    except OSError:
        _append_failure_detail(failure_details, str(root_dir), "iterdir_error")
        return 0, 1, failure_details

    for entry in entries:
        try:
            is_dir = entry.is_dir()
        except OSError:
            failed += 1
            _append_failure_detail(failure_details, str(entry), "entry_stat_error")
            continue

        if not is_dir:
            continue

        try:
            has_skill_file = (entry / "SKILL.md").is_file()
        except OSError:
            failed += 1
            _append_failure_detail(
                failure_details,
                str(entry / "SKILL.md"),
                "skill_file_stat_error",
            )
            continue

        if has_skill_file:
            count += 1
    return count, failed, failure_details


def _load_from_root(root_dir: Path, source: str) -> tuple[int, int, list[dict[str, str]]]:
    if not root_dir.exists() or not root_dir.is_dir():
        return 0, 0, []

    loaded = 0
    failed = 0
    failure_details: list[dict[str, str]] = []

    try:
        entries = sorted(root_dir.iterdir(), key=lambda item: item.name)
    except OSError:
        _append_failure_detail(failure_details, str(root_dir), "iterdir_error")
        return 0, 1, failure_details

    for skill_dir in entries:
        try:
            is_dir = skill_dir.is_dir()
        except OSError:
            failed += 1
            _append_failure_detail(failure_details, str(skill_dir), "entry_stat_error")
            continue

        if not is_dir:
            continue

        skill_file = skill_dir / "SKILL.md"
        try:
            has_skill_file = skill_file.is_file()
        except OSError:
            failed += 1
            _append_failure_detail(failure_details, str(skill_file), "skill_file_stat_error")
            continue

        if not has_skill_file:
            continue

        parsed, reason = _parse_skill_file(skill_file, fallback_name=skill_dir.name)
        if parsed is None:
            failed += 1
            _append_failure_detail(failure_details, str(skill_file), reason)
            continue
        _SKILLS[parsed["name"]] = SkillEntry(
            name=parsed["name"],
            description=parsed["description"],
            prompt=parsed["prompt"],
            source=source,
            path=str(skill_file),
        )
        loaded += 1
    return loaded, failed, failure_details


def _parse_skill_file(skill_file: Path, fallback_name: str) -> tuple[dict[str, str] | None, str]:
    try:
        content = skill_file.read_text(encoding="utf-8")
    except Exception:
        return None, "read_error"

    metadata, body = _parse_frontmatter(content)
    name = metadata.get("name", fallback_name).strip() or fallback_name
    description = metadata.get("description", "").strip() or _infer_description(name, body)
    prompt = body.strip()
    if not prompt:
        return None, "empty_prompt"
    return {"name": name, "description": description, "prompt": prompt}, ""


def _append_failure_detail(details: list[dict[str, str]], path: str, reason: str) -> None:
    if len(details) >= _MAX_FAILURE_DETAILS:
        return
    details.append({"path": path, "reason": reason})


def _parse_frontmatter(content: str) -> tuple[dict[str, str], str]:
    lines = content.splitlines()
    if len(lines) < 3 or lines[0].strip() != "---":
        return {}, content

    end_index = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_index = i
            break

    if end_index == -1:
        return {}, content

    metadata: dict[str, str] = {}
    for line in lines[1:end_index]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip()

    body = "\n".join(lines[end_index + 1 :]).lstrip()
    return metadata, body


def _infer_description(name: str, prompt_text: str) -> str:
    for line in prompt_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
        if stripped:
            return stripped
    return f"Skill: {name}"
