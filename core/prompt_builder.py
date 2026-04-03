# core/prompt_builder.py
from config import BASE_SYSTEM_PROMPT
from skills.registry import load as load_skills


def build_system_prompt(skill_names: list[str] | None = None) -> str:
    parts = [BASE_SYSTEM_PROMPT]
    if skill_names:
        skill_text = load_skills(skill_names)
        if skill_text:
            parts.append(skill_text)
    return "\n\n".join(parts)
