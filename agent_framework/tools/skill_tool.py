# tools/skill_tool.py
from agent_framework.skills.registry import get_skill_prompt
from agent_framework.tools.registry import register


def use_skill(name: str) -> str:
    prompt = get_skill_prompt(name)
    if prompt is None:
        return f"[error] 未知 skill: {name}"
    return prompt


def register_skill_tool() -> None:
    register(
        name="use_skill",
        description=(
            "加载某个技能的完整指令文本。"
            "当 system prompt 中 available_skills 显示某个技能可用时，"
            "调用此工具获取该技能的完整内容。"
        ),
        parameters={
            "name": {"type": "string", "description": "技能名称（来自 available_skills）"},
        },
        required=["name"],
        handler=use_skill,
    )
