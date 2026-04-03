# tests/test_providers.py
"""provider 格式转换单元测试，不需要真实 API key"""

from llm.factory import create_provider
from llm.openai_provider import OpenAIProvider, _to_openai_tool


def test_to_openai_tool():
    schema = {
        "name": "run_bash",
        "description": "run shell command",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
        },
    }
    result = _to_openai_tool(schema)
    assert result["type"] == "function"
    assert result["function"]["name"] == "run_bash"
    assert "command" in result["function"]["parameters"]["properties"]


def test_anthropic_format_tool_result():
    from llm.anthropic_provider import AnthropicProvider

    p = AnthropicProvider.__new__(AnthropicProvider)
    r = p.format_tool_result("call_abc", "hello output")
    assert r["type"] == "tool_result"
    assert r["tool_use_id"] == "call_abc"
    assert r["content"] == "hello output"


def test_openai_format_tool_result():
    p = OpenAIProvider.__new__(OpenAIProvider)
    r = p.format_tool_result("call_xyz", "result text")
    assert r["role"] == "tool"
    assert r["tool_call_id"] == "call_xyz"


def test_anthropic_tool_results_as_message():
    from llm.anthropic_provider import AnthropicProvider

    p = AnthropicProvider.__new__(AnthropicProvider)
    results = [
        {"type": "tool_result", "tool_use_id": "1", "content": "a"},
        {"type": "tool_result", "tool_use_id": "2", "content": "b"},
    ]
    msgs = p.tool_results_as_message(results)
    assert isinstance(msgs, list)
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert len(msgs[0]["content"]) == 2


def test_openai_tool_results_as_message():
    p = OpenAIProvider.__new__(OpenAIProvider)
    results = [
        {"role": "tool", "tool_call_id": "1", "content": "a"},
        {"role": "tool", "tool_call_id": "2", "content": "b"},
    ]
    msgs = p.tool_results_as_message(results)
    assert isinstance(msgs, list)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "tool"
    assert msgs[1]["role"] == "tool"


def test_anthropic_block_to_dict_keeps_tool_result_fields():
    from llm.anthropic_provider import _block_to_dict

    class _ToolResultBlock:
        type = "tool_result"
        tool_use_id = "toolu_123"
        content = "done"
        is_error = False

    result = _block_to_dict(_ToolResultBlock())
    assert result["type"] == "tool_result"
    assert result["tool_use_id"] == "toolu_123"
    assert result["content"] == "done"
    assert result["is_error"] is False


def test_factory_anthropic():
    p = create_provider({"type": "anthropic", "model": "claude-opus-4-6"})
    assert hasattr(p, "chat")


def test_factory_openai():
    p = create_provider({"type": "openai", "model": "gpt-4o", "api_key": "test-key"})
    assert hasattr(p, "chat")


def test_factory_unknown():
    import pytest

    with pytest.raises(ValueError, match="未知 provider"):
        create_provider({"type": "unknown_llm", "model": "x"})
