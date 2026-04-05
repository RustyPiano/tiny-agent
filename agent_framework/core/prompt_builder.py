# core/prompt_builder.py
from agent_framework._config import BASE_SYSTEM_PROMPT
from agent_framework.skills.registry import load as load_skills
from agent_framework.skills.registry import render_available_skills
from agent_framework.tools import registry as tool_registry


def _render_allowed_tools(tool_schemas: list[dict] | None = None) -> str:
    schemas = tool_schemas if tool_schemas is not None else tool_registry.get_schemas()
    tool_names = sorted({schema["name"] for schema in schemas})
    tool_names.append("NONE")
    lines = ["## 允许的工具（运行时注册）", "你只允许调用以下 action 名称："]
    lines.extend(f"- {name}" for name in tool_names)
    return "\n".join(lines)


def build_system_prompt(
    skill_names: list[str] | None = None,
    enable_subagent_flow: bool = False,
    tool_schemas: list[dict] | None = None,
) -> str:
    parts = [BASE_SYSTEM_PROMPT]
    parts.append(_render_allowed_tools(tool_schemas))

    if enable_subagent_flow:
        parts.append(
            "## Subagent Flow Discipline\n"
            "- Follow deterministic phase progression: "
            "implement -> spec_review -> quality_review.\n"
            "- Report status using only DONE, DONE_WITH_CONCERNS, NEEDS_CONTEXT, or BLOCKED.\n"
            "- Surface blockers/context needs immediately; do not skip phases."
        )

    available_skills = render_available_skills()
    if available_skills:
        parts.append(available_skills)

    if skill_names:
        skill_text = load_skills(skill_names)
        if skill_text:
            parts.append("## Preloaded Skills\n" + skill_text)
    return "\n\n".join(parts)
