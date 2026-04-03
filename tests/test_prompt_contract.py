from __future__ import annotations

import re

from config import (
    BASE_SYSTEM_PROMPT,
    MAX_FILE_READ_LINES,
    MAX_HISTORY_RECORDS,
    MAX_MEMORY_LINES,
    MAX_TURNS,
    SYSTEM_PROMPT_DYNAMIC_BOUNDARY,
)


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
        '当 action 为白名单工具名时，"action_input" 必须为 object',
        '当 action 为 "NONE" 时，"action_input" 必须且只能为 "NONE"',
    ]
    for item in required_items:
        assert item in section, f"missing output protocol clause: {item}"


def test_dynamic_boundary_section_contains_marker() -> None:
    section = _section_body("动态上下文边界")
    assert SYSTEM_PROMPT_DYNAMIC_BOUNDARY in section


def test_tool_whitelist_section_includes_complete_expected_actions() -> None:
    section = _section_body("工具白名单（仅调用白名单工具）")
    expected_actions = {
        "read_file",
        "write_file",
        "edit_file",
        "run_bash",
        "grep",
        "list_dir",
        "use_skill",
        "summarize",
        "finish",
        "NONE",
    }
    listed_actions = {
        line.strip()[2:] for line in section.splitlines() if line.strip().startswith("- ")
    }
    assert expected_actions == listed_actions


def test_context_discipline_section_uses_config_limit_constants() -> None:
    section = _section_body("上下文纪律")
    assert f"最多 {MAX_TURNS} 轮" in section
    assert f"最多 {MAX_HISTORY_RECORDS} 条" in section
    assert f"最多 {MAX_MEMORY_LINES} 行" in section
    assert f"最多 {MAX_FILE_READ_LINES} 行" in section


def test_security_section_has_non_bypassable_guards() -> None:
    section = _section_body("安全规则")
    required_phrases = [
        "禁止调用白名单之外的工具或伪造工具",
        "禁止越权访问工作空间之外路径",
        "禁止执行高风险、不可逆或提权意图命令",
    ]
    for phrase in required_phrases:
        assert phrase in section, f"missing security guard: {phrase}"
