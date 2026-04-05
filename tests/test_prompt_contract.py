from __future__ import annotations

import re

import pytest

from agent_framework._config import (
    AgentSettings,
    BASE_SYSTEM_PROMPT,
    MAX_FILE_READ_LINES,
    MAX_HISTORY_RECORDS,
    MAX_MEMORY_LINES,
    MAX_TURNS,
    SYSTEM_PROMPT_DYNAMIC_BOUNDARY,
)
from agent_framework.core.prompt_builder import build_system_prompt
from agent_framework.main import bootstrap
from agent_framework.tools import registry

EXPECTED_BUILTIN_TOOL_NAMES = {
    "read_file",
    "write_file",
    "edit_file",
    "run_bash",
    "run_job",
    "grep",
    "list_dir",
    "use_skill",
    "summarize",
    "finish",
}


@pytest.fixture(autouse=True)
def _clear_tool_registry() -> None:
    registry._TOOLS.clear()


def _section_body(title: str) -> str:
    pattern = rf"## {re.escape(title)}\n(.*?)(?:\n## |\Z)"
    match = re.search(pattern, BASE_SYSTEM_PROMPT, flags=re.S)
    assert match is not None, f"missing section: {title}"
    return match.group(1)


def test_output_protocol_section_contains_required_contract_items() -> None:
    section = _section_body("输出协议（严格 ReAct JSON）")
    required_items = [
        '"thought"',
        '"action"',
        '"action_input"',
        '"action_input": object | "NONE"',
        '当 action 为运行时允许的工具名时，"action_input" 必须为 object',
        '当 action 为 "NONE" 时，"action_input" 必须且只能为 "NONE"',
    ]
    for item in required_items:
        assert item in section, f"missing output protocol clause: {item}"


def test_dynamic_boundary_section_contains_marker() -> None:
    section = _section_body("动态上下文边界")
    assert SYSTEM_PROMPT_DYNAMIC_BOUNDARY in section


def test_static_prompt_no_longer_contains_hand_maintained_tool_whitelist_section() -> None:
    assert "## 工具白名单（仅调用白名单工具）" not in BASE_SYSTEM_PROMPT


def test_prompt_uses_explicit_runtime_tool_schemas_not_module_global_registry() -> None:
    registry.register(
        name="global_only_tool",
        description="Only present in the module-global registry.",
        parameters={},
        required=[],
        handler=lambda: "ok",
    )

    prompt = build_system_prompt(
        tool_schemas=[
            {
                "name": "explicit_tool",
                "description": "Provided directly to prompt builder.",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            }
        ]
    )

    assert "- explicit_tool" in prompt
    assert "- global_only_tool" not in prompt


def test_prompt_renders_allowed_tools_from_registered_schemas() -> None:
    bootstrap(settings=AgentSettings())

    tool_schemas = registry.get_schemas()
    prompt = build_system_prompt(tool_schemas=tool_schemas)
    pattern = r"## 允许的工具（运行时注册）\n(.*?)(?:\n## |\Z)"
    match = re.search(pattern, prompt, flags=re.S)
    assert match is not None, "missing runtime tool section"

    section = match.group(1)
    listed_actions = {
        line.strip()[2:] for line in section.splitlines() if line.strip().startswith("- ")
    }
    expected_actions = {schema["name"] for schema in registry.get_schemas()}
    expected_actions.add("NONE")

    assert expected_actions == listed_actions


def test_bootstrap_preserves_expected_builtin_tool_names() -> None:
    bootstrap(settings=AgentSettings())

    registered_tools = {schema["name"] for schema in registry.get_schemas()}
    assert EXPECTED_BUILTIN_TOOL_NAMES <= registered_tools


def test_bootstrap_resets_registry_to_builtin_tools_and_clears_hooks() -> None:
    registry.register(
        name="stray_tool",
        description="leftover tool",
        parameters={},
        required=[],
        handler=lambda: "ok",
    )
    registry.register_before_tool_call(lambda name, inputs: inputs)

    bootstrap(settings=AgentSettings())

    registered_tools = {schema["name"] for schema in registry.get_schemas()}
    assert "stray_tool" not in registered_tools
    assert registry._BEFORE_TOOL_CALL_HOOKS == []


def test_context_discipline_section_uses_config_limit_constants() -> None:
    section = _section_body("上下文纪律")
    assert f"最多 {MAX_TURNS} 轮" in section
    assert f"最多 {MAX_HISTORY_RECORDS} 条" in section
    assert f"最多 {MAX_MEMORY_LINES} 行" in section
    assert f"最多 {MAX_FILE_READ_LINES} 行" in section


def test_context_discipline_section_directs_long_tasks_to_run_job() -> None:
    section = _section_body("上下文纪律")
    assert "`run_bash` 仅用于前台短任务" in section
    assert "预计可能超过短任务窗口的命令必须使用 `run_job`" in section
    assert "`run_bash` 超时后，不要通过增大 `timeout` 重试" in section


def test_security_section_has_non_bypassable_guards() -> None:
    section = _section_body("安全规则")
    required_phrases = [
        "禁止调用当前运行时允许集合之外的工具或伪造工具",
        "禁止越权访问工作空间之外路径",
        "禁止执行高风险、不可逆或提权意图命令",
    ]
    for phrase in required_phrases:
        assert phrase in section, f"missing security guard: {phrase}"


def test_prompt_includes_subagent_flow_marker_when_enabled() -> None:
    prompt = build_system_prompt(enable_subagent_flow=True)
    assert "## Subagent Flow Discipline" in prompt


def test_prompt_omits_subagent_flow_marker_when_disabled() -> None:
    prompt = build_system_prompt(enable_subagent_flow=False)
    assert "## Subagent Flow Discipline" not in prompt
